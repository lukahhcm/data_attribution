import copy

import torch

from data_attribution_v2.rho import faithful_batch_choice, practical_rho_scores
from data_attribution_v2.selector import f2sa_inner_step


def _classifier() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 6),
        torch.nn.BatchNorm1d(6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 2),
    )


class DropoutClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.dropout = torch.nn.Dropout(p=0.9)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(inputs))


def _bn_buffers(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    bn = next(module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d))
    return bn.running_mean.detach().clone(), bn.running_var.detach().clone()


def test_f2sa_inner_step_learns_parameters_without_committing_bn_buffers() -> None:
    torch.manual_seed(4)
    theta_hat = _classifier()
    theta_tilde = copy.deepcopy(theta_hat)
    optimizer_hat = torch.optim.SGD(theta_hat.parameters(), lr=0.05)
    optimizer_tilde = torch.optim.SGD(theta_tilde.parameters(), lr=0.05)
    candidate_x = torch.randn(8, 4)
    candidate_y = torch.arange(8) % 2
    validation_x = torch.randn(8, 4)
    validation_y = (torch.arange(8) + 1) % 2
    indices = torch.arange(8)
    omega = torch.full((8,), 0.5)
    before_buffers = _bn_buffers(theta_hat)
    before_parameter = next(theta_hat.parameters()).detach().clone()

    f2sa_inner_step(
        theta_hat,
        theta_tilde,
        optimizer_hat,
        optimizer_tilde,
        (indices, candidate_x, candidate_x, candidate_y),
        (indices, validation_x, validation_x, validation_y),
        omega,
        penalty=2.0,
        device=torch.device("cpu"),
    )

    after_buffers = _bn_buffers(theta_hat)
    assert torch.equal(before_buffers[0], after_buffers[0])
    assert torch.equal(before_buffers[1], after_buffers[1])
    assert not torch.equal(before_parameter, next(theta_hat.parameters()).detach())


def test_rho_batch_statistics_do_not_persist_and_choice_is_stable() -> None:
    torch.manual_seed(8)
    target = _classifier()
    irreducible = copy.deepcopy(target)
    views = torch.randn(16, 4)
    labels = torch.arange(16) % 2
    before = _bn_buffers(target)

    scores, current, irreducible_loss = practical_rho_scores(
        target, irreducible, views, labels, use_batch_statistics=True
    )
    after = _bn_buffers(target)

    assert scores.shape == current.shape == irreducible_loss.shape == (16,)
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    assert faithful_batch_choice(torch.tensor([1.0, 1.0, 0.0]), 2).tolist() == [0, 1]


def test_rho_scoring_does_not_enable_dropout() -> None:
    torch.manual_seed(11)
    target = DropoutClassifier()
    irreducible = copy.deepcopy(target)
    target.train(True)
    irreducible.train(True)
    views = torch.randn(12, 4)
    labels = torch.arange(12) % 2

    first = practical_rho_scores(target, irreducible, views, labels)[0]
    second = practical_rho_scores(target, irreducible, views, labels)[0]

    assert torch.equal(first, second)
    assert target.training and irreducible.training
