from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Parameterization = Literal["projected", "sigmoid"]


def capped_simplex_projection(vector: torch.Tensor, total: float) -> torch.Tensor:
    """Project a vector onto {0 <= w <= 1, sum(w) = total}."""
    if vector.ndim != 1:
        raise ValueError("capped_simplex_projection expects a vector")
    if not 0 < total <= vector.numel():
        raise ValueError("total must be in (0, len(vector)]")
    work = vector.double()
    lower = float((work - 1.0).min().item())
    upper = float(work.max().item())
    for _ in range(80):
        threshold = (lower + upper) / 2.0
        current = torch.clamp(work - threshold, 0.0, 1.0).sum().item()
        if current > total:
            lower = threshold
        else:
            upper = threshold
    projected = torch.clamp(work - (lower + upper) / 2.0, 0.0, 1.0)
    return projected.to(vector.dtype)


def standardized_direction(score: torch.Tensor) -> torch.Tensor:
    if score.ndim != 1:
        raise ValueError("score must be a vector")
    centered = score - score.mean()
    return centered / centered.square().mean().sqrt().clamp_min(1e-12)


def budgeted_sigmoid(logits: torch.Tensor, total: float) -> torch.Tensor:
    """Map logits to (0,1)^n while enforcing an exact sum through a shared shift."""
    if logits.ndim != 1:
        raise ValueError("logits must be a vector")
    if not 0 < total < logits.numel():
        raise ValueError("sigmoid parameterization requires total in (0, len(logits))")
    work = logits.double()
    lower = -80.0 - float(work.max().item())
    upper = 80.0 - float(work.min().item())
    for _ in range(100):
        shift = (lower + upper) / 2.0
        current = torch.sigmoid(work + shift).sum().item()
        if current < total:
            lower = shift
        else:
            upper = shift
    weights = torch.sigmoid(work + (lower + upper) / 2.0)
    return weights.to(logits.dtype)


def budgeted_sigmoid_logit_direction(
    weights: torch.Tensor, negative_weight_gradient: torch.Tensor
) -> torch.Tensor:
    """Apply the chain rule for w_i=sigmoid(a_i+c(a)), sum_i w_i=B.

    ``negative_weight_gradient`` is the descent score -dL/dw. The shared shift
    creates a dense centering term obtained by implicit differentiation.
    """
    if weights.shape != negative_weight_gradient.shape or weights.ndim != 1:
        raise ValueError("weights and score must be equally sized vectors")
    sensitivity = weights * (1.0 - weights)
    normalizer = sensitivity.sum().clamp_min(1e-12)
    mean = (sensitivity * negative_weight_gradient).sum() / normalizer
    return sensitivity * (negative_weight_gradient - mean)


@dataclass
class OmegaState:
    """The persistent continuous outer state; Top-B is only a deployed view."""

    budget: int
    parameterization: Parameterization
    weights: torch.Tensor
    logits: torch.Tensor | None = None

    @classmethod
    def uniform(
        cls,
        num_examples: int,
        budget: int,
        parameterization: Parameterization = "projected",
        *,
        device: torch.device | str | None = None,
    ) -> "OmegaState":
        if not 0 < budget < num_examples:
            raise ValueError("budget must be strictly between zero and num_examples")
        weights = torch.full(
            (num_examples,), budget / num_examples, dtype=torch.float32, device=device
        )
        if parameterization == "projected":
            return cls(budget, parameterization, weights)
        if parameterization == "sigmoid":
            logits = torch.zeros(num_examples, dtype=torch.float32, device=device)
            return cls(budget, parameterization, budgeted_sigmoid(logits, budget), logits)
        raise ValueError(f"Unknown parameterization: {parameterization}")

    def validate(self, atol: float = 1e-5) -> None:
        if self.weights.ndim != 1:
            raise ValueError("weights must be a vector")
        if self.weights.min().item() < -atol or self.weights.max().item() > 1.0 + atol:
            raise ValueError("weights violate [0,1] bounds")
        if abs(self.weights.sum().item() - self.budget) > atol:
            raise ValueError("weights violate the fixed budget")
        if self.parameterization == "sigmoid" and self.logits is None:
            raise ValueError("sigmoid state requires logits")

    def selected(self) -> torch.Tensor:
        """Return deterministic Top-B indices without mutating continuous state."""
        return torch.argsort(self.weights, descending=True, stable=True)[: self.budget]

    def step(self, score: torch.Tensor, step_size: float) -> "OmegaState":
        """Take a descent step using a score defined as -dL/domega."""
        if score.shape != self.weights.shape:
            raise ValueError("score shape must match omega")
        direction = standardized_direction(score.detach().to(self.weights))
        if self.parameterization == "projected":
            weights = capped_simplex_projection(
                self.weights + step_size * direction, float(self.budget)
            )
            result = OmegaState(self.budget, self.parameterization, weights)
        else:
            assert self.logits is not None
            logit_direction = budgeted_sigmoid_logit_direction(self.weights, direction)
            logit_direction = standardized_direction(logit_direction)
            logits = self.logits + step_size * logit_direction
            result = OmegaState(
                self.budget,
                self.parameterization,
                budgeted_sigmoid(logits, float(self.budget)),
                logits,
            )
        result.validate()
        return result

    def effective_sample_size(self) -> float:
        numerator = self.weights.sum().square()
        denominator = self.weights.square().sum().clamp_min(1e-12)
        return float((numerator / denominator).item())
