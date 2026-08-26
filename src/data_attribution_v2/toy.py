from __future__ import annotations

import torch


def quadratic_solutions(
    weights: torch.Tensor,
    centers: torch.Tensor,
    validation_center: torch.Tensor,
    penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form inner solutions for a scalar strongly-convex gate problem."""
    if weights.shape != centers.shape or weights.ndim != 1:
        raise ValueError("weights and centers must be equal-sized vectors")
    n = weights.numel()
    weighted_mass = weights.sum()
    weighted_center = (weights * centers).sum()
    theta_hat = weighted_center / weighted_mass
    theta_tilde = (
        validation_center + (penalty / n) * weighted_center
    ) / (1.0 + (penalty / n) * weighted_mass)
    return theta_hat, theta_tilde


def per_example_quadratic_loss(theta: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return 0.5 * (theta - centers).square()


def minimized_penalty_value(
    weights: torch.Tensor,
    centers: torch.Tensor,
    validation_center: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    theta_hat, theta_tilde = quadratic_solutions(
        weights, centers, validation_center, penalty
    )
    validation = 0.5 * (theta_tilde - validation_center).square()
    g_tilde = (weights * per_example_quadratic_loss(theta_tilde, centers)).mean()
    g_hat = (weights * per_example_quadratic_loss(theta_hat, centers)).mean()
    return validation + penalty * (g_tilde - g_hat)
