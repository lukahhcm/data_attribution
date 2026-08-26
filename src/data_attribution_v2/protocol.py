from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class Method(StrEnum):
    UNIFORM_FIXED = "uniform_fixed"
    UNIFORM_ROUND = "uniform_round"
    RHO_ONE_SHOT = "rho_one_shot"
    VF_ONE_SHOT = "vf_one_shot"
    RHO_MULTI = "rho_multi"
    VF_MULTI = "vf_multi"


def selection_update_epochs(
    total_epochs: int, warm_start_epochs: int, update_interval: int | None
) -> list[int]:
    """Epoch boundaries after which a new subset is selected.

    ``None`` is the one-shot/infinite interval. The first update always occurs
    after the shared warm start. Updates at ``total_epochs`` are excluded
    because no real target-training block would consume them.
    """
    if not 0 < warm_start_epochs < total_epochs:
        raise ValueError("warm_start_epochs must be in (0, total_epochs)")
    if update_interval is None:
        return [warm_start_epochs]
    if update_interval <= 0:
        raise ValueError("update_interval must be positive or None")
    return list(range(warm_start_epochs, total_epochs, update_interval))


def shared_initial_subset(num_examples: int, budget: int, seed: int) -> torch.Tensor:
    if not 0 < budget < num_examples:
        raise ValueError("budget must be in (0, num_examples)")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_examples, generator=generator)[:budget]


@dataclass(frozen=True)
class ProtocolSpec:
    method: Method
    total_epochs: int
    warm_start_epochs: int
    update_interval: int | None
    retention: float

    def __post_init__(self) -> None:
        if not 0 < self.retention < 1:
            raise ValueError("retention must be in (0,1)")
        if self.method in {Method.RHO_ONE_SHOT, Method.VF_ONE_SHOT}:
            if self.update_interval is not None:
                raise ValueError("one-shot methods require update_interval=None")

    def update_epochs(self) -> list[int]:
        return selection_update_epochs(
            self.total_epochs, self.warm_start_epochs, self.update_interval
        )
