#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_attribution_v2.f2sa import f2sa_coordinate_gradient, vf_score  # noqa: E402
from data_attribution_v2.toy import (  # noqa: E402
    minimized_penalty_value,
    per_example_quadratic_loss,
    quadratic_solutions,
)


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.dot(first, second) / (first.norm() * second.norm()))


def finite_difference(
    weights: torch.Tensor,
    centers: torch.Tensor,
    validation_center: torch.Tensor,
    penalty: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    result = torch.empty_like(weights)
    for index in range(weights.numel()):
        offset = torch.zeros_like(weights)
        offset[index] = epsilon
        upper = minimized_penalty_value(
            weights + offset, centers, validation_center, penalty
        )
        lower = minimized_penalty_value(
            weights - offset, centers, validation_center, penalty
        )
        result[index] = (upper - lower) / (2 * epsilon)
    return result


def main() -> None:
    weights = torch.tensor([0.15, 0.35, 0.65, 0.85], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([-1.4, -0.2, 0.9, 1.8], dtype=torch.float64)
    validation_center = torch.tensor(0.3, dtype=torch.float64)
    penalty = 4.0

    value = minimized_penalty_value(weights, centers, validation_center, penalty)
    autograd = torch.autograd.grad(value, weights)[0]
    finite = finite_difference(weights.detach(), centers, validation_center, penalty)
    theta_hat, theta_tilde = quadratic_solutions(
        weights.detach(), centers, validation_center, penalty
    )
    loss_hat = per_example_quadratic_loss(theta_hat, centers)
    loss_tilde = per_example_quadratic_loss(theta_tilde, centers)
    loss_difference = f2sa_coordinate_gradient(loss_hat, loss_tilde, penalty)
    negative_direction = vf_score(loss_hat, loss_tilde)

    relative_error = float((autograd - finite).norm() / autograd.norm())
    report = {
        "autograd_vs_finite_cosine": cosine(autograd, finite),
        "autograd_vs_loss_difference_cosine": cosine(autograd, loss_difference),
        "negative_direction_vs_negative_gradient_cosine": cosine(
            negative_direction, -autograd
        ),
        "finite_difference_relative_error": relative_error,
        "passed": bool(
            cosine(autograd, finite) > 0.999
            and cosine(autograd, loss_difference) > 0.999
            and cosine(negative_direction, -autograd) > 0.999
            and relative_error < 1e-6
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
