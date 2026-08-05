from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import dump_config, parse_config_args
from .data import build_data
from .distributed import cleanup_distributed, initialize_distributed, seed_everything
from .io import append_jsonl
from .models import make_model
from .runtime import make_loader, output_directory, set_loader_epoch
from .training import evaluate, make_optimizer


def _train_reference(config: dict, bundle, context, output_dir: Path):
    rho_config = config["rho"]
    model = make_model(config["model"], bundle.num_classes).to(context.device)
    optimizer = make_optimizer(model, rho_config["optimizer"])
    workers = int(config["runtime"].get("workers", 4))
    pin_memory = bool(config["runtime"].get("pin_memory", True))
    batch_size = int(rho_config.get("reference_batch_size", 128))
    train_loader = make_loader(
        bundle.validation, context, batch_size, True, 101, workers, pin_memory
    )
    dev_loader = make_loader(
        bundle.development, context, batch_size, False, 102, workers, pin_memory
    )
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, int(rho_config["reference_epochs"]) + 1):
        set_loader_epoch(train_loader, epoch)
        model.train()
        for _, _, train_view, target in train_loader:
            train_view = train_view.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_view), target)
            loss.backward()
            optimizer.step()
        dev_metrics = evaluate(model, dev_loader, context.device)
        if dev_metrics["loss"] < best_loss:
            best_loss = dev_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Reference model produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    torch.save(best_state, output_dir / "reference_model.pt")
    return model


def run(config: dict) -> Path:
    context = initialize_distributed()
    if context.world_size != 1:
        raise RuntimeError(
            "Original RHO is intentionally single-GPU per run to preserve global 320-to-32 semantics. "
            "Parallelize seeds/configurations across GPUs instead."
        )
    seed = int(config["experiment"].get("seed", 1))
    seed_everything(seed)
    method = str(config["experiment"]["method"]).lower()
    if method not in {"original_rho", "uniform_online"}:
        raise ValueError("RHO entry point supports original_rho or uniform_online")
    output_dir = output_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_dir / "config.yaml")

    bundle = build_data(config["dataset"])
    reference = _train_reference(config, bundle, context, output_dir)
    target_model = make_model(config["model"], bundle.num_classes).to(context.device)
    optimizer = make_optimizer(target_model, config["rho"]["optimizer"])
    workers = int(config["runtime"].get("workers", 4))
    pin_memory = bool(config["runtime"].get("pin_memory", True))
    candidate_batch_size = int(config["rho"]["candidate_batch_size"])
    selected_batch_size = int(config["rho"]["selected_batch_size"])
    if selected_batch_size > candidate_batch_size:
        raise ValueError("selected_batch_size cannot exceed candidate_batch_size")
    train_loader = make_loader(
        bundle.candidate,
        context,
        candidate_batch_size,
        True,
        seed,
        workers,
        pin_memory,
    )
    dev_loader = make_loader(
        bundle.development,
        context,
        candidate_batch_size,
        False,
        seed,
        workers,
        pin_memory,
    )
    test_loader = make_loader(
        bundle.test,
        context,
        candidate_batch_size,
        False,
        seed,
        workers,
        pin_memory,
    )
    selection_counts = torch.zeros(len(bundle.candidate), dtype=torch.int64)
    best_dev_loss = float("inf")
    best_state = None

    for epoch in range(1, int(config["rho"]["target_epochs"]) + 1):
        set_loader_epoch(train_loader, epoch)
        for global_index, score_view, train_view, target in train_loader:
            score_view = score_view.to(context.device, non_blocking=True)
            train_view = train_view.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            actual_selected = min(selected_batch_size, target.numel())
            if method == "uniform_online":
                chosen = torch.randperm(target.numel(), device=context.device)[:actual_selected]
            else:
                selection_mode = str(config["rho"].get("selection_mode", "train")).lower()
                score_view_name = str(config["rho"].get("score_view", "train")).lower()
                if score_view_name == "train":
                    rho_input = train_view
                elif score_view_name == "deterministic":
                    rho_input = score_view
                else:
                    raise ValueError("rho.score_view must be train or deterministic")
                if selection_mode == "train":
                    target_model.train()
                elif selection_mode == "eval":
                    target_model.eval()
                else:
                    raise ValueError("rho.selection_mode must be train or eval")
                with torch.no_grad():
                    current_loss = F.cross_entropy(
                        target_model(rho_input), target, reduction="none"
                    )
                    reference_loss = F.cross_entropy(
                        reference(rho_input), target, reduction="none"
                    )
                    chosen = torch.topk(
                        current_loss - reference_loss, k=actual_selected, largest=True
                    ).indices
            selection_counts[global_index[chosen.cpu()]] += 1
            target_model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(target_model(train_view[chosen]), target[chosen])
            loss.backward()
            optimizer.step()

        dev_metrics = evaluate(target_model, dev_loader, context.device)
        append_jsonl(
            output_dir / "metrics.jsonl",
            {
                "epoch": epoch,
                "dev_loss": dev_metrics["loss"],
                "dev_accuracy": dev_metrics["accuracy"],
            },
        )
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_state = copy.deepcopy(target_model.state_dict())

    if best_state is None:
        raise RuntimeError("Target model produced no checkpoint")
    target_model.load_state_dict(best_state)
    test_metrics = evaluate(target_model, test_loader, context.device)
    torch.save(best_state, output_dir / "best_target_model.pt")
    torch.save(selection_counts, output_dir / "selection_counts.pt")
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        total_selections = selection_counts.sum().clamp_min(1)
        corrupted = torch.as_tensor(bundle.corrupted_mask, dtype=torch.bool)
        noisy_selections = selection_counts[corrupted].sum()
        ever_selected = selection_counts > 0
        json.dump(
            {
                **test_metrics,
                "total_selection_events": int(selection_counts.sum().item()),
                "unique_examples_selected": int(ever_selected.sum().item()),
                "selected_noise_event_fraction": float(
                    (noisy_selections / total_selections).item()
                ),
            },
            handle,
            indent=2,
        )
    cleanup_distributed()
    return output_dir


def main() -> None:
    config, _ = parse_config_args("Train Original RHO or uniform-online")
    run(config)


if __name__ == "__main__":
    main()
