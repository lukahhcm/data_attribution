from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from .config import dump_config, parse_config_args
from .data import build_data
from .distributed import (
    all_reduce_sum,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    seed_everything,
    unwrap,
    wrap_ddp,
)
from .io import append_jsonl
from .models import make_model, maybe_convert_sync_batchnorm
from .runtime import make_loader, output_directory, set_loader_epoch, topk_count
from .training import evaluate, make_optimizer, make_scheduler


def _selected_indices(config: dict, selection_run: str | None, num_examples: int) -> torch.Tensor:
    if selection_run is not None:
        path = Path(selection_run) / "selected_indices.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        selected = torch.load(path, map_location="cpu", weights_only=True).long()
    else:
        method = str(config["experiment"]["method"]).lower()
        if method == "full":
            selected = torch.arange(num_examples)
        elif method == "uniform":
            generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
            count = topk_count(num_examples, float(config["selection"]["retention"]))
            selected = torch.randperm(num_examples, generator=generator)[:count]
        else:
            raise ValueError("Without --selection-run, evaluator method must be full or uniform")
    if selected.ndim != 1 or selected.unique().numel() != selected.numel():
        raise ValueError("selected indices must be a unique one-dimensional tensor")
    if selected.min() < 0 or selected.max() >= num_examples:
        raise ValueError("selected indices are outside the candidate dataset")
    return selected


def run(config: dict, selection_run: str | None) -> Path:
    context = initialize_distributed()
    seed = int(config["experiment"].get("seed", 1))
    seed_everything(seed, context.rank)
    bundle = build_data(config["dataset"])
    selected = _selected_indices(config, selection_run, len(bundle.candidate))
    selected_dataset = Subset(bundle.candidate, selected.tolist())

    if selection_run is not None:
        output_dir = Path(selection_run) / "evaluator"
    else:
        output_dir = output_directory(config) / "evaluator"
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, output_dir / "config.yaml")
        torch.save(selected, output_dir / "selected_indices.pt")
    barrier()

    evaluator_config = config["evaluator"]
    workers = int(config["runtime"].get("workers", 4))
    pin_memory = bool(config["runtime"].get("pin_memory", True))
    batch_size = int(evaluator_config["batch_size"])
    train_loader = make_loader(
        selected_dataset, context, batch_size, True, seed, workers, pin_memory
    )
    development_loader = make_loader(
        bundle.development, context, batch_size, False, seed, workers, pin_memory
    )
    test_loader = make_loader(
        bundle.test, context, batch_size, False, seed, workers, pin_memory
    )

    model = make_model(config["model"], bundle.num_classes).to(context.device)
    model = maybe_convert_sync_batchnorm(
        model,
        bool(config["runtime"].get("sync_batchnorm", False)),
        context.enabled,
    )
    model = wrap_ddp(model, context)
    optimizer = make_optimizer(model, evaluator_config["optimizer"])
    epochs = int(evaluator_config["epochs"])
    scheduler = make_scheduler(
        optimizer, evaluator_config.get("scheduler", {"name": "none"}), epochs
    )
    metrics_path = output_dir / "metrics.jsonl"
    best_dev_loss = float("inf")
    best_state = None
    autocast_enabled = bool(config["runtime"].get("amp", True)) and context.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    for epoch in range(1, epochs + 1):
        set_loader_epoch(train_loader, epoch)
        model.train()
        loss_sum = 0.0
        count_sum = 0
        for _, _, train_view, target in train_loader:
            train_view = train_view.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=context.device.type, enabled=autocast_enabled):
                loss = F.cross_entropy(model(train_view), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item()) * target.numel()
            count_sum += target.numel()
        if scheduler is not None:
            scheduler.step()
        loss_count = torch.tensor(
            [loss_sum, count_sum], device=context.device, dtype=torch.float64
        )
        all_reduce_sum(loss_count)
        dev_metrics = evaluate(model, development_loader, context.device)
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_state = copy.deepcopy(unwrap(model).state_dict())
        if context.is_main:
            append_jsonl(
                metrics_path,
                {
                    "epoch": epoch,
                    "train_loss": float(
                        (loss_count[0] / loss_count[1].clamp_min(1)).item()
                    ),
                    "dev_loss": dev_metrics["loss"],
                    "dev_accuracy": dev_metrics["accuracy"],
                },
            )

    if best_state is None:
        raise RuntimeError("Evaluator produced no checkpoint")
    unwrap(model).load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, context.device)
    if context.is_main:
        torch.save(best_state, output_dir / "best_model.pt")
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **test_metrics,
                    "selected_count": int(selected.numel()),
                    "selected_noise_fraction": float(bundle.corrupted_mask[selected.numpy()].mean()),
                },
                handle,
                indent=2,
            )
    barrier()
    cleanup_distributed()
    return output_dir


def main() -> None:
    config, args = parse_config_args("Train an independent evaluator on selected data")
    run(config, args.selection_run)


if __name__ == "__main__":
    main()
