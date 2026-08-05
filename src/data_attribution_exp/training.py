from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed import all_reduce_sum, reduce_mean, unwrap


def cycle_loader(loader: Iterable):
    while True:
        yield from loader


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


def alpha_at_epoch(
    epoch: int,
    epochs: int,
    alpha_min: float,
    alpha_max: float,
    ramp_fraction: float,
    schedule: str,
) -> float:
    ramp_epochs = max(1, int(round(epochs * ramp_fraction)))
    progress = min(1.0, max(0.0, epoch / ramp_epochs))
    schedule = schedule.lower()
    if schedule == "constant":
        return float(alpha_max)
    if schedule == "linear":
        return float(alpha_min + progress * (alpha_max - alpha_min))
    if schedule == "geometric":
        if alpha_min <= 0:
            raise ValueError("geometric alpha schedule requires alpha_min > 0")
        return float(alpha_min * (alpha_max / alpha_min) ** progress)
    raise ValueError(f"Unsupported alpha schedule: {schedule}")


def round_boundaries(epochs: int, rounds: int) -> list[int]:
    if rounds < 1 or rounds > epochs:
        raise ValueError("rounds must be in [1, epochs]")
    values = [math.ceil((index + 1) * epochs / rounds) for index in range(rounds)]
    values[-1] = epochs
    if len(set(values)) != rounds:
        raise ValueError("round boundaries are not unique; reduce rounds")
    return values


def simplex_projection(vector: torch.Tensor, total: float) -> torch.Tensor:
    """Euclidean projection onto {x >= 0, sum(x) = total}."""
    if vector.ndim != 1:
        raise ValueError("simplex_projection expects a one-dimensional tensor")
    if total <= 0:
        raise ValueError("simplex total must be positive")
    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - total
    indices = torch.arange(1, len(vector) + 1, device=vector.device, dtype=vector.dtype)
    condition = sorted_values - cumulative / indices > 0
    rho = torch.nonzero(condition, as_tuple=False)[-1, 0]
    theta = cumulative[rho] / indices[rho]
    return torch.clamp(vector - theta, min=0)


@torch.no_grad()
def classification_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def calibrate_alpha(
    model: torch.nn.Module,
    train_batch,
    validation_batch,
    device: torch.device,
) -> float:
    """Estimate ||grad f|| / ||grad g|| on the initial model."""
    module = unwrap(model)
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    _, _, train_view, train_target = train_batch
    _, validation_view, _, validation_target = validation_batch
    train_view = train_view.to(device, non_blocking=True)
    train_target = train_target.to(device, non_blocking=True)
    validation_view = validation_view.to(device, non_blocking=True)
    validation_target = validation_target.to(device, non_blocking=True)

    train_loss = F.cross_entropy(module(train_view), train_target)
    train_grads = torch.autograd.grad(train_loss, parameters)

    validation_loss = F.cross_entropy(module(validation_view), validation_target)
    validation_grads = torch.autograd.grad(validation_loss, parameters)

    def global_norm_sq(gradients) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for gradient in gradients:
            global_gradient = gradient.detach().float().clone()
            if dist.is_initialized():
                dist.all_reduce(global_gradient, op=dist.ReduceOp.SUM)
                global_gradient /= dist.get_world_size()
            total += global_gradient.square().sum()
        return total

    train_norm_sq = global_norm_sq(train_grads)
    validation_norm_sq = global_norm_sq(validation_grads)
    return float(torch.sqrt(validation_norm_sq / train_norm_sq.clamp_min(1e-24)).item())


@dataclass
class EpochMetrics:
    hat_loss: float
    tilde_train_loss: float
    validation_loss: float


def train_selector_epoch(
    theta_hat: torch.nn.Module,
    theta_tilde: torch.nn.Module,
    optimizer_hat: torch.optim.Optimizer,
    optimizer_tilde: torch.optim.Optimizer,
    train_loader,
    validation_loader,
    omega: torch.Tensor,
    alpha: float,
    device: torch.device,
    amp: bool,
    scaler_hat,
    scaler_tilde,
) -> EpochMetrics:
    theta_hat.train()
    theta_tilde.train()
    validation_iterator = cycle_loader(validation_loader)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    autocast_enabled = amp and device.type == "cuda"

    for index, _, train_view, target in train_loader:
        _, validation_view, _, validation_target = next(validation_iterator)
        index = index.to(device, non_blocking=True)
        train_view = train_view.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        validation_view = validation_view.to(device, non_blocking=True)
        validation_target = validation_target.to(device, non_blocking=True)
        batch_weight = omega[index].detach()

        optimizer_hat.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            hat_loss_vector = F.cross_entropy(theta_hat(train_view), target, reduction="none")
            hat_objective = (batch_weight * hat_loss_vector).mean()
        scaler_hat.scale(hat_objective).backward()
        scaler_hat.step(optimizer_hat)
        scaler_hat.update()

        optimizer_tilde.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            tilde_loss_vector = F.cross_entropy(
                theta_tilde(train_view), target, reduction="none"
            )
            tilde_train_objective = (batch_weight * tilde_loss_vector).mean()
            validation_objective = F.cross_entropy(
                theta_tilde(validation_view), validation_target
            )
            tilde_objective = validation_objective + alpha * tilde_train_objective
        scaler_tilde.scale(tilde_objective).backward()
        scaler_tilde.step(optimizer_tilde)
        scaler_tilde.update()

        batch_size = target.numel()
        totals[0] += hat_objective.detach().double() * batch_size
        totals[1] += tilde_train_objective.detach().double() * batch_size
        totals[2] += validation_objective.detach().double() * validation_target.numel()
        totals[3] += batch_size
        totals[4] += validation_target.numel()

    all_reduce_sum(totals)
    count = totals[3].clamp_min(1)
    return EpochMetrics(
        hat_loss=float((totals[0] / count).item()),
        tilde_train_loss=float((totals[1] / count).item()),
        validation_loss=float((totals[2] / totals[4].clamp_min(1)).item()),
    )
