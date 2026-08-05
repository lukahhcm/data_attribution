from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .distributed import all_reduce_sum
from .training import simplex_projection


@dataclass
class ScoreResult:
    score: torch.Tensor
    loss_hat: torch.Tensor
    loss_tilde: torch.Tensor


@torch.no_grad()
def global_rho_vf_scores(
    theta_hat: torch.nn.Module,
    theta_tilde: torch.nn.Module,
    loader,
    num_examples: int,
    device: torch.device,
) -> ScoreResult:
    theta_hat.eval()
    theta_tilde.eval()
    hat_sum = torch.zeros(num_examples, device=device, dtype=torch.float64)
    tilde_sum = torch.zeros_like(hat_sum)
    counts = torch.zeros_like(hat_sum)

    for index, score_view, _, target in loader:
        index = index.to(device, non_blocking=True)
        score_view = score_view.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        hat_loss = F.cross_entropy(theta_hat(score_view), target, reduction="none").double()
        tilde_loss = F.cross_entropy(
            theta_tilde(score_view), target, reduction="none"
        ).double()
        hat_sum.scatter_add_(0, index, hat_loss)
        tilde_sum.scatter_add_(0, index, tilde_loss)
        counts.scatter_add_(0, index, torch.ones_like(hat_loss))

    all_reduce_sum(hat_sum)
    all_reduce_sum(tilde_sum)
    all_reduce_sum(counts)
    if torch.any(counts == 0):
        missing = int((counts == 0).sum().item())
        raise RuntimeError(f"Distributed scoring missed {missing} candidate examples")
    loss_hat = (hat_sum / counts).float()
    loss_tilde = (tilde_sum / counts).float()
    return ScoreResult(loss_hat - loss_tilde, loss_hat, loss_tilde)


def standardized_score(score: torch.Tensor) -> torch.Tensor:
    centered = score - score.mean()
    return centered / centered.square().mean().sqrt().clamp_min(1e-8)


@torch.no_grad()
def update_weights(
    omega: torch.Tensor,
    score: torch.Tensor,
    learning_rate: float,
) -> torch.Tensor:
    proposal = omega + learning_rate * standardized_score(score)
    return simplex_projection(proposal, total=float(len(omega)))


def selected_noise_fraction(selected: torch.Tensor, corrupted_mask: np.ndarray) -> float:
    indices = selected.detach().cpu().numpy()
    return float(np.asarray(corrupted_mask, dtype=bool)[indices].mean())


def effective_sample_size(weights: torch.Tensor) -> float:
    numerator = weights.sum().square()
    denominator = weights.square().sum().clamp_min(1e-12)
    return float((numerator / denominator).item())


def topk_jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    first_set = set(first.detach().cpu().tolist())
    second_set = set(second.detach().cpu().tolist())
    union = first_set | second_set
    return len(first_set & second_set) / max(1, len(union))


def spearman_rank_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Spearman correlation without a SciPy dependency.

    Reducible-loss scores are continuous in normal use, so ties are vanishingly
    rare. Stable sorting keeps this diagnostic deterministic if ties occur.
    """
    if first.numel() != second.numel():
        raise ValueError("rank correlation inputs must have equal length")
    if first.numel() < 2:
        return 1.0
    first_cpu = first.detach().float().cpu()
    second_cpu = second.detach().float().cpu()
    rank_first = torch.empty_like(first_cpu)
    rank_second = torch.empty_like(second_cpu)
    rank_first[torch.argsort(first_cpu, stable=True)] = torch.arange(
        first_cpu.numel(), dtype=first_cpu.dtype
    )
    rank_second[torch.argsort(second_cpu, stable=True)] = torch.arange(
        second_cpu.numel(), dtype=second_cpu.dtype
    )
    rank_first -= rank_first.mean()
    rank_second -= rank_second.mean()
    denominator = rank_first.norm() * rank_second.norm()
    if denominator <= 0:
        return 0.0
    return float(torch.dot(rank_first, rank_second).div(denominator).item())
