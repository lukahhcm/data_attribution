from __future__ import annotations

from dataclasses import dataclass

import torch


def weighted_candidate_risk(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Fixed-denominator risk n^{-1} sum_i omega_i loss_i."""
    if losses.shape != weights.shape:
        raise ValueError("losses and weights must have equal shape")
    return (losses * weights).mean()


def vf_score(loss_hat: torch.Tensor, loss_tilde: torch.Tensor) -> torch.Tensor:
    """Negative finite-penalty outer-gradient direction, up to alpha/n."""
    if loss_hat.shape != loss_tilde.shape:
        raise ValueError("loss vectors must have equal shape")
    return loss_hat - loss_tilde


def f2sa_coordinate_gradient(
    loss_hat: torch.Tensor, loss_tilde: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Coordinate gradient of the minimized F2SA penalty."""
    if loss_hat.shape != loss_tilde.shape or loss_hat.ndim != 1:
        raise ValueError("loss vectors must have equal one-dimensional shape")
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    return penalty * (loss_tilde - loss_hat) / loss_hat.numel()


@dataclass(frozen=True)
class PenaltySchedule:
    initial: float
    power: float = 0.0

    def __post_init__(self) -> None:
        if self.initial <= 0:
            raise ValueError("initial penalty must be positive")
        if self.power < 0:
            raise ValueError("penalty power must be nonnegative")

    def at_round(self, round_index: int) -> float:
        if round_index < 0:
            raise ValueError("round_index must be nonnegative")
        return float(self.initial * (1 + round_index) ** self.power)


@dataclass(frozen=True)
class InnerStopConfig:
    relative_objective_tolerance: float = 1e-3
    score_cosine_threshold: float = 0.99
    topk_jaccard_threshold: float = 0.95
    patience: int = 3
    maximum_passes: int = 50


@dataclass(frozen=True)
class InnerObservation:
    hat_objective: float
    tilde_objective: float
    score: torch.Tensor
    selected: torch.Tensor


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.float().norm() * second.float().norm()
    if denominator.item() == 0:
        return 1.0 if torch.equal(first, second) else 0.0
    return float(torch.dot(first.float(), second.float()).div(denominator).item())


def jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    left = set(first.detach().cpu().tolist())
    right = set(second.detach().cpu().tolist())
    return len(left & right) / max(1, len(left | right))


class InnerConvergenceTracker:
    """Preregistered stopping rule based only on inner/validation information."""

    def __init__(self, config: InnerStopConfig) -> None:
        self.config = config
        self.previous: InnerObservation | None = None
        self.stable_checks = 0

    @staticmethod
    def _relative_change(current: float, previous: float) -> float:
        return abs(current - previous) / max(1e-12, abs(previous))

    def update(self, observation: InnerObservation, pass_index: int) -> bool:
        if pass_index < 1:
            raise ValueError("pass_index is one-based")
        if self.previous is not None:
            stable = (
                self._relative_change(
                    observation.hat_objective, self.previous.hat_objective
                )
                <= self.config.relative_objective_tolerance
                and self._relative_change(
                    observation.tilde_objective, self.previous.tilde_objective
                )
                <= self.config.relative_objective_tolerance
                and cosine(observation.score, self.previous.score)
                >= self.config.score_cosine_threshold
                and jaccard(observation.selected, self.previous.selected)
                >= self.config.topk_jaccard_threshold
            )
            self.stable_checks = self.stable_checks + 1 if stable else 0
        self.previous = observation
        return (
            self.stable_checks >= self.config.patience
            or pass_index >= self.config.maximum_passes
        )
