import torch

from data_attribution_v2.omega import (
    OmegaState,
    budgeted_sigmoid,
    budgeted_sigmoid_logit_direction,
    capped_simplex_projection,
)


def test_capped_simplex_projection_enforces_budget_and_bounds() -> None:
    vector = torch.tensor([-3.0, -0.2, 0.4, 1.7, 8.0])
    projected = capped_simplex_projection(vector, total=2.0)

    assert torch.all(projected >= 0)
    assert torch.all(projected <= 1)
    assert torch.isclose(projected.sum(), torch.tensor(2.0), atol=1e-6)


def test_uniform_state_and_projected_step_preserve_constraints() -> None:
    state = OmegaState.uniform(20, budget=4, parameterization="projected")
    assert torch.allclose(state.weights, torch.full((20,), 0.2))

    updated = state.step(torch.arange(20, dtype=torch.float32), step_size=0.3)
    updated.validate()
    assert updated.selected().tolist() == [19, 18, 17, 16]


def test_budgeted_sigmoid_enforces_exact_sum() -> None:
    logits = torch.linspace(-4, 4, 31)
    weights = budgeted_sigmoid(logits, total=7.0)

    assert torch.all(weights > 0)
    assert torch.all(weights < 1)
    assert torch.isclose(weights.sum(), torch.tensor(7.0), atol=1e-5)


def test_budgeted_sigmoid_chain_rule_matches_directional_difference() -> None:
    logits = torch.tensor([-1.1, -0.2, 0.7, 1.4], dtype=torch.float64)
    score = torch.tensor([0.8, -0.4, 0.1, -0.3], dtype=torch.float64)
    weights = budgeted_sigmoid(logits, total=1.5)
    logit_direction = budgeted_sigmoid_logit_direction(weights, score)

    epsilon = 1e-5
    perturbed = budgeted_sigmoid(logits + epsilon * logit_direction, total=1.5)
    finite_difference = ((perturbed - weights) * score).sum() / epsilon

    assert finite_difference.item() > 0
    assert abs(logit_direction.sum().item()) < 1e-10
