from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def selection_batch_mode(model: torch.nn.Module, use_batch_statistics: bool = True):
    """Use large-batch BN statistics without enabling dropout or committing buffers."""
    training_states = {module: module.training for module in model.modules()}
    buffers = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    model.eval()
    if use_batch_statistics:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train(True)
    try:
        yield
    finally:
        current = dict(model.named_buffers())
        for name, value in buffers.items():
            current[name].copy_(value)
        for module, training in training_states.items():
            module.training = training


@torch.no_grad()
def practical_rho_scores(
    target: torch.nn.Module,
    irreducible_model: torch.nn.Module,
    score_views: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_batch_statistics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faithful large-batch current-loss minus frozen-IL scoring."""
    with selection_batch_mode(target, use_batch_statistics):
        current_loss = F.cross_entropy(target(score_views), labels, reduction="none")
    with selection_batch_mode(irreducible_model, use_batch_statistics):
        irreducible_loss = F.cross_entropy(
            irreducible_model(score_views), labels, reduction="none"
        )
    return current_loss - irreducible_loss, current_loss, irreducible_loss


def faithful_batch_choice(scores: torch.Tensor, selected_batch_size: int) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be a vector")
    if not 0 < selected_batch_size <= scores.numel():
        raise ValueError("selected_batch_size must fit the candidate batch")
    return torch.argsort(scores, descending=True, stable=True)[:selected_batch_size]
