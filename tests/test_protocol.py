import pytest
import torch

from data_attribution_v2.protocol import (
    Method,
    ProtocolSpec,
    selection_update_epochs,
    shared_initial_subset,
)


def test_update_epochs_exclude_unused_terminal_update() -> None:
    assert selection_update_epochs(30, 10, None) == [10]
    assert selection_update_epochs(30, 10, 5) == [10, 15, 20, 25]


def test_shared_initial_subset_is_deterministic() -> None:
    first = shared_initial_subset(100, 10, seed=7)
    second = shared_initial_subset(100, 10, seed=7)
    third = shared_initial_subset(100, 10, seed=8)

    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert first.unique().numel() == 10


def test_one_shot_rejects_finite_update_interval() -> None:
    with pytest.raises(ValueError):
        ProtocolSpec(Method.VF_ONE_SHOT, 30, 10, 5, 0.1)
