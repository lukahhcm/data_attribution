from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import dump_config, parse_config_args
from .data import build_data, save_data_artifacts
from .distributed import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
    seed_everything,
    unwrap,
    wrap_ddp,
)
from .io import append_jsonl, load_selector_checkpoint, save_checkpoint
from .models import make_model, maybe_convert_sync_batchnorm
from .runtime import make_loader, output_directory, set_loader_epoch, topk_count
from .selection import (
    effective_sample_size,
    global_rho_vf_scores,
    selected_noise_fraction,
    spearman_rank_correlation,
    topk_jaccard,
    update_weights,
)
from .training import (
    alpha_at_epoch,
    calibrate_alpha,
    make_optimizer,
    make_scheduler,
    round_boundaries,
    train_selector_epoch,
)


def _save_selection(
    output_dir: Path,
    selected: torch.Tensor,
    omega: torch.Tensor,
    score: torch.Tensor,
    method: str,
) -> None:
    torch.save(selected.cpu(), output_dir / "selected_indices.pt")
    torch.save(omega.cpu(), output_dir / "omega.pt")
    torch.save(score.cpu(), output_dir / "final_score.pt")
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"method": method, "selected_count": int(selected.numel())},
            handle,
            indent=2,
        )


def run(config: dict) -> Path:
    context = initialize_distributed()
    seed = int(config["experiment"].get("seed", 1))
    seed_everything(seed, context.rank)
    if context.device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["runtime"].get("cudnn_benchmark", True))

    output_dir = output_directory(config)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, output_dir / "config.yaml")
    barrier()

    bundle = build_data(config["dataset"])
    if context.is_main:
        save_data_artifacts(bundle, output_dir)

    workers = int(config["runtime"].get("workers", 4))
    pin_memory = bool(config["runtime"].get("pin_memory", True))
    batch_size = int(config["selector"]["batch_size"])
    validation_batch_size = int(config["selector"].get("validation_batch_size", batch_size))
    score_batch_size = int(config["selector"].get("score_batch_size", batch_size))

    train_loader = make_loader(
        bundle.candidate, context, batch_size, True, seed, workers, pin_memory
    )
    validation_loader = make_loader(
        bundle.validation,
        context,
        validation_batch_size,
        True,
        seed + 17,
        workers,
        pin_memory,
    )
    score_loader = make_loader(
        bundle.candidate,
        context,
        score_batch_size,
        False,
        seed,
        workers,
        pin_memory,
    )

    theta_hat = make_model(config["model"], bundle.num_classes).to(context.device)
    theta_tilde = copy.deepcopy(theta_hat).to(context.device)
    sync_bn = bool(config["runtime"].get("sync_batchnorm", False))
    theta_hat = maybe_convert_sync_batchnorm(theta_hat, sync_bn, context.enabled)
    theta_tilde = maybe_convert_sync_batchnorm(theta_tilde, sync_bn, context.enabled)
    theta_hat = wrap_ddp(theta_hat, context)
    theta_tilde = wrap_ddp(theta_tilde, context)

    optimizer_hat = make_optimizer(theta_hat, config["selector"]["optimizer"])
    optimizer_tilde = make_optimizer(theta_tilde, config["selector"]["optimizer"])
    epochs = int(config["selector"]["epochs"])
    scheduler_hat = make_scheduler(
        optimizer_hat, config["selector"].get("scheduler", {"name": "none"}), epochs
    )
    scheduler_tilde = make_scheduler(
        optimizer_tilde, config["selector"].get("scheduler", {"name": "none"}), epochs
    )
    amp_enabled = bool(config["runtime"].get("amp", True)) and context.device.type == "cuda"
    scaler_hat = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scaler_tilde = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    alpha_config = config["selection"]["alpha"]
    if str(alpha_config.get("scale", "fixed")).lower() == "gradient_ratio":
        alpha_reference = calibrate_alpha(
            theta_tilde,
            next(iter(train_loader)),
            next(iter(validation_loader)),
            context.device,
        )
        alpha_min = float(alpha_config.get("min_multiplier", 0.1)) * alpha_reference
        alpha_max = float(alpha_config.get("max_multiplier", 3.0)) * alpha_reference
    else:
        alpha_reference = 1.0
        alpha_min = float(alpha_config["min"])
        alpha_max = float(alpha_config["max"])

    num_examples = len(bundle.candidate)
    omega = torch.ones(num_examples, device=context.device, dtype=torch.float32)
    method = str(config["experiment"]["method"]).lower()
    if method not in {"rho_vf_one_step", "iterative_vf"}:
        raise ValueError("selector method must be rho_vf_one_step or iterative_vf")
    rounds = 1 if method == "rho_vf_one_step" else int(config["selection"]["rounds"])
    boundaries = round_boundaries(epochs, rounds)
    boundary_set = set(boundaries)
    retention = float(config["selection"]["retention"])
    selected_count = topk_count(num_examples, retention)
    omega_lr = float(config["selection"]["omega_lr"])
    metrics_path = output_dir / "selector_metrics.jsonl"
    first_topk = None
    first_score = None
    latest_score = torch.zeros_like(omega)
    round_index = 0
    start_epoch = 0
    resume_path = config["experiment"].get("resume")
    if resume_path:
        checkpoint = load_selector_checkpoint(
            Path(resume_path),
            theta_hat,
            theta_tilde,
            optimizer_hat,
            optimizer_tilde,
            scheduler_hat,
            scheduler_tilde,
            scaler_hat,
            scaler_tilde,
        )
        omega = checkpoint["omega"].to(context.device)
        start_epoch = int(checkpoint["epoch"])
        round_index = int(checkpoint["round_index"])
        if checkpoint.get("latest_score") is not None:
            latest_score = checkpoint["latest_score"].to(context.device)
        if checkpoint.get("first_topk") is not None:
            first_topk = checkpoint["first_topk"].to(context.device)
        first_round_path = output_dir / "round_001.pt"
        if first_round_path.exists():
            first_score = torch.load(
                first_round_path, map_location=context.device, weights_only=False
            )["score"]
    start_time = time.perf_counter()

    for epoch in range(start_epoch + 1, epochs + 1):
        set_loader_epoch(train_loader, epoch)
        set_loader_epoch(validation_loader, epoch)
        alpha = alpha_at_epoch(
            epoch - 1,
            epochs,
            alpha_min,
            alpha_max,
            float(alpha_config.get("ramp_fraction", 0.5)),
            str(alpha_config.get("schedule", "linear")),
        )
        epoch_metrics = train_selector_epoch(
            theta_hat,
            theta_tilde,
            optimizer_hat,
            optimizer_tilde,
            train_loader,
            validation_loader,
            omega,
            alpha,
            context.device,
            amp_enabled,
            scaler_hat,
            scaler_tilde,
        )
        if scheduler_hat is not None:
            scheduler_hat.step()
        if scheduler_tilde is not None:
            scheduler_tilde.step()

        if epoch in boundary_set:
            round_index += 1
            scores = global_rho_vf_scores(
                theta_hat, theta_tilde, score_loader, num_examples, context.device
            )
            latest_score = scores.score
            if method == "iterative_vf":
                omega = update_weights(omega, latest_score, omega_lr)
                ranking_value = omega
            else:
                ranking_value = latest_score
            selected = torch.topk(ranking_value, k=selected_count, largest=True).indices
            if first_topk is None:
                first_topk = selected.clone()
            if first_score is None:
                first_score = latest_score.clone()
            observed_labels = torch.as_tensor(
                bundle.candidate_labels, device=selected.device
            )
            clean_labels = torch.as_tensor(
                bundle.clean_candidate_labels, device=selected.device
            )
            observed_class_counts = torch.bincount(
                observed_labels[selected], minlength=bundle.num_classes
            ).cpu().tolist()
            clean_class_counts = torch.bincount(
                clean_labels[selected], minlength=bundle.num_classes
            ).cpu().tolist()
            record = {
                "epoch": epoch,
                "round": round_index,
                "alpha": alpha,
                "alpha_reference": alpha_reference,
                "hat_loss": epoch_metrics.hat_loss,
                "tilde_train_loss": epoch_metrics.tilde_train_loss,
                "validation_loss": epoch_metrics.validation_loss,
                "effective_sample_size": effective_sample_size(omega),
                "selected_noise_fraction": selected_noise_fraction(
                    selected, bundle.corrupted_mask
                ),
                "topk_jaccard_with_first": topk_jaccard(first_topk, selected),
                "score_spearman_with_first": spearman_rank_correlation(
                    first_score, latest_score
                ),
                "selected_observed_class_counts": observed_class_counts,
                "selected_clean_class_counts": clean_class_counts,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if context.is_main:
                append_jsonl(metrics_path, record)
                torch.save(
                    {
                        "score": scores.score.cpu(),
                        "loss_hat": scores.loss_hat.cpu(),
                        "loss_tilde": scores.loss_tilde.cpu(),
                        "selected": selected.cpu(),
                    },
                    output_dir / f"round_{round_index:03d}.pt",
                )
                save_checkpoint(
                    output_dir / "selector_checkpoint.pt",
                    theta_hat,
                    theta_tilde,
                    optimizer_hat,
                    optimizer_tilde,
                    omega,
                    epoch,
                    round_index,
                    alpha,
                    scheduler_hat,
                    scheduler_tilde,
                    scaler_hat,
                    scaler_tilde,
                    latest_score,
                    first_topk,
                )

    ranking_value = omega if method == "iterative_vf" else latest_score
    selected = torch.topk(ranking_value, k=selected_count, largest=True).indices
    if context.is_main:
        _save_selection(output_dir, selected, omega, latest_score, method)
        torch.save(unwrap(theta_hat).state_dict(), output_dir / "theta_hat.pt")
        torch.save(unwrap(theta_tilde).state_dict(), output_dir / "theta_tilde.pt")
    barrier()
    cleanup_distributed()
    return output_dir


def main() -> None:
    config, _ = parse_config_args("Train a RHO/VF-1 or iterative VF selector")
    run(config)


if __name__ == "__main__":
    main()
