import torch

from data_attribution_v2.f2sa import PenaltySchedule, f2sa_coordinate_gradient, vf_score
from data_attribution_v2.toy import (
    minimized_penalty_value,
    per_example_quadratic_loss,
    quadratic_solutions,
)


def test_loss_difference_matches_exact_outer_gradient() -> None:
    weights = torch.tensor([0.2, 0.7, 0.4, 0.9], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([-1.5, -0.1, 0.8, 2.0], dtype=torch.float64)
    validation_center = torch.tensor(0.35, dtype=torch.float64)
    penalty = 3.0

    objective = minimized_penalty_value(weights, centers, validation_center, penalty)
    exact_gradient = torch.autograd.grad(objective, weights)[0]
    theta_hat, theta_tilde = quadratic_solutions(
        weights.detach(), centers, validation_center, penalty
    )
    loss_hat = per_example_quadratic_loss(theta_hat, centers)
    loss_tilde = per_example_quadratic_loss(theta_tilde, centers)
    loss_difference_gradient = f2sa_coordinate_gradient(
        loss_hat, loss_tilde, penalty
    )

    assert torch.allclose(exact_gradient, loss_difference_gradient, atol=1e-10)
    assert torch.allclose(
        vf_score(loss_hat, loss_tilde),
        -(weights.numel() / penalty) * exact_gradient,
        atol=1e-10,
    )


def test_penalty_schedule_uses_round_index() -> None:
    schedule = PenaltySchedule(initial=2.0, power=0.5)
    assert schedule.at_round(0) == 2.0
    assert schedule.at_round(3) == 4.0
