from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .f2sa import weighted_candidate_risk


@contextmanager
def inner_optimization_mode(model: torch.nn.Module):
    """Train parameters, including BN affine terms, without changing BN buffers."""
    was_training = model.training
    bn_training = {
        module: module.training
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }
    model.train(True)
    for module in bn_training:
        module.eval()
    try:
        yield
    finally:
        for module, training in bn_training.items():
            module.train(training)
        model.train(was_training)


@dataclass(frozen=True)
class InnerStepMetrics:
    hat_objective: float
    tilde_candidate_objective: float
    tilde_validation_objective: float


def f2sa_inner_step(
    theta_hat: torch.nn.Module,
    theta_tilde: torch.nn.Module,
    optimizer_hat: torch.optim.Optimizer,
    optimizer_tilde: torch.optim.Optimizer,
    candidate_batch,
    validation_batch,
    omega: torch.Tensor,
    penalty: float,
    device: torch.device,
    *,
    amp: bool = False,
) -> InnerStepMetrics:
    """Advance both finite-penalty inner branches by one stochastic step."""
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    index, _, candidate_view, candidate_labels = candidate_batch
    _, validation_view, _, validation_labels = validation_batch
    index = index.to(device, non_blocking=True)
    candidate_view = candidate_view.to(device, non_blocking=True)
    candidate_labels = candidate_labels.to(device, non_blocking=True)
    validation_view = validation_view.to(device, non_blocking=True)
    validation_labels = validation_labels.to(device, non_blocking=True)
    batch_weights = omega[index].detach()
    autocast_enabled = amp and device.type == "cuda"

    optimizer_hat.zero_grad(set_to_none=True)
    with inner_optimization_mode(theta_hat):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            hat_losses = F.cross_entropy(
                theta_hat(candidate_view), candidate_labels, reduction="none"
            )
            hat_objective = weighted_candidate_risk(hat_losses, batch_weights)
        hat_objective.backward()
    optimizer_hat.step()

    optimizer_tilde.zero_grad(set_to_none=True)
    with inner_optimization_mode(theta_tilde):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            tilde_losses = F.cross_entropy(
                theta_tilde(candidate_view), candidate_labels, reduction="none"
            )
            tilde_candidate = weighted_candidate_risk(tilde_losses, batch_weights)
            tilde_validation = F.cross_entropy(
                theta_tilde(validation_view), validation_labels
            )
            tilde_objective = tilde_validation + penalty * tilde_candidate
        tilde_objective.backward()
    optimizer_tilde.step()

    return InnerStepMetrics(
        float(hat_objective.detach().item()),
        float(tilde_candidate.detach().item()),
        float(tilde_validation.detach().item()),
    )


@torch.no_grad()
def per_example_losses(
    model: torch.nn.Module,
    loader: Iterable,
    num_examples: int,
    device: torch.device,
    *,
    amp: bool = False,
) -> torch.Tensor:
    """Score each stable candidate index exactly once with frozen BN buffers."""
    losses = torch.empty(num_examples, dtype=torch.float32, device=device)
    seen = torch.zeros(num_examples, dtype=torch.bool, device=device)
    was_training = model.training
    model.eval()
    try:
        for index, score_view, _, labels in loader:
            index = index.to(device)
            if index.min().item() < 0 or index.max().item() >= num_examples:
                raise ValueError("candidate index lies outside the score vector")
            if seen[index].any():
                raise ValueError("global score loader contains duplicate candidate indices")
            score_view = score_view.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                batch_losses = F.cross_entropy(model(score_view), labels, reduction="none")
            losses[index] = batch_losses.float()
            seen[index] = True
    finally:
        model.train(was_training)
    if not seen.all():
        raise ValueError("global score loader did not cover every candidate exactly once")
    return losses


@torch.no_grad()
def global_vf_score(
    theta_hat: torch.nn.Module,
    theta_tilde: torch.nn.Module,
    loader: Iterable,
    num_examples: int,
    device: torch.device,
    *,
    amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_hat = per_example_losses(theta_hat, loader, num_examples, device, amp=amp)
    loss_tilde = per_example_losses(theta_tilde, loader, num_examples, device, amp=amp)
    return loss_hat - loss_tilde, loss_hat, loss_tilde
