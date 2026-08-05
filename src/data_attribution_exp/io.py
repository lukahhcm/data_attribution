from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .distributed import unwrap


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    theta_hat: torch.nn.Module,
    theta_tilde: torch.nn.Module,
    optimizer_hat: torch.optim.Optimizer,
    optimizer_tilde: torch.optim.Optimizer,
    omega: torch.Tensor,
    epoch: int,
    round_index: int,
    alpha: float,
    scheduler_hat: Any = None,
    scheduler_tilde: Any = None,
    scaler_hat: Any = None,
    scaler_tilde: Any = None,
    latest_score: torch.Tensor | None = None,
    first_topk: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "theta_hat": unwrap(theta_hat).state_dict(),
            "theta_tilde": unwrap(theta_tilde).state_dict(),
            "optimizer_hat": optimizer_hat.state_dict(),
            "optimizer_tilde": optimizer_tilde.state_dict(),
            "omega": omega.detach().cpu(),
            "epoch": epoch,
            "round_index": round_index,
            "alpha": alpha,
            "scheduler_hat": scheduler_hat.state_dict() if scheduler_hat else None,
            "scheduler_tilde": scheduler_tilde.state_dict() if scheduler_tilde else None,
            "scaler_hat": scaler_hat.state_dict() if scaler_hat else None,
            "scaler_tilde": scaler_tilde.state_dict() if scaler_tilde else None,
            "latest_score": latest_score.detach().cpu() if latest_score is not None else None,
            "first_topk": first_topk.detach().cpu() if first_topk is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_selector_checkpoint(
    path: Path,
    theta_hat: torch.nn.Module,
    theta_tilde: torch.nn.Module,
    optimizer_hat: torch.optim.Optimizer,
    optimizer_tilde: torch.optim.Optimizer,
    scheduler_hat: Any = None,
    scheduler_tilde: Any = None,
    scaler_hat: Any = None,
    scaler_tilde: Any = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(theta_hat).load_state_dict(checkpoint["theta_hat"])
    unwrap(theta_tilde).load_state_dict(checkpoint["theta_tilde"])
    optimizer_hat.load_state_dict(checkpoint["optimizer_hat"])
    optimizer_tilde.load_state_dict(checkpoint["optimizer_tilde"])
    for object_, key in (
        (scheduler_hat, "scheduler_hat"),
        (scheduler_tilde, "scheduler_tilde"),
        (scaler_hat, "scaler_hat"),
        (scaler_tilde, "scaler_tilde"),
    ):
        if object_ is not None and checkpoint.get(key) is not None:
            object_.load_state_dict(checkpoint[key])
    return checkpoint
