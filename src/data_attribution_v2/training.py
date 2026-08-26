from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributed import all_reduce_sum

def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    name = str(config.get("name", "sgd")).lower()
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(config["lr"]),
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            nesterov=bool(config.get("nesterov", False)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(config["lr"]),
            betas=tuple(config.get("betas", [0.9, 0.999])),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(
    optimizer: torch.optim.Optimizer, config: dict, epochs: int
):
    name = str(config.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        warmup_epochs = int(config.get("warmup_epochs", 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=float(config.get("eta_min", 0.0)),
        )
        if warmup_epochs == 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(config.get("warmup_start_factor", 0.1)),
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs]
        )
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config["milestones"]),
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


@torch.no_grad()
def classification_counts(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum(dtype=torch.float64)
    count = torch.tensor(targets.numel(), device=targets.device, dtype=torch.float64)
    return correct, count


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.float64)
    count_sum = torch.zeros((), device=device, dtype=torch.float64)
    for _, score_view, _, target in loader:
        score_view = score_view.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(score_view)
        loss_sum += F.cross_entropy(logits, target, reduction="sum").double()
        correct, count = classification_counts(logits, target)
        correct_sum += correct
        count_sum += count
    all_reduce_sum(loss_sum)
    all_reduce_sum(correct_sum)
    all_reduce_sum(count_sum)
    return {
        "loss": float((loss_sum / count_sum.clamp_min(1)).item()),
        "accuracy": float((correct_sum / count_sum.clamp_min(1)).item()),
    }
