import numpy as np
import torch

from data_attribution_exp.data import stratified_split, symmetric_label_noise
from data_attribution_exp.selection import (
    spearman_rank_correlation,
    standardized_score,
    update_weights,
)
from data_attribution_exp.training import round_boundaries, simplex_projection


def test_symmetric_noise_changes_exactly_requested_labels():
    labels = np.arange(1000) % 10
    noisy, mask = symmetric_label_noise(labels, num_classes=10, rate=0.1, seed=7)
    assert mask.sum() == 100
    assert np.all(noisy[mask] != labels[mask])
    assert np.all(noisy[~mask] == labels[~mask])


def test_stratified_split_has_requested_disjoint_sizes():
    labels = np.repeat(np.arange(10), 100)
    parts = stratified_split(labels, (800, 100, 100), seed=3)
    assert tuple(map(len, parts)) == (800, 100, 100)
    joined = np.concatenate(parts)
    assert len(np.unique(joined)) == len(labels)


def test_simplex_projection_is_nonnegative_and_preserves_sum():
    vector = torch.tensor([-2.0, 0.5, 3.0, 8.0])
    projected = simplex_projection(vector, total=4.0)
    assert torch.all(projected >= 0)
    assert torch.isclose(projected.sum(), torch.tensor(4.0), atol=1e-6)


def test_weight_update_preserves_uniform_sum_and_follows_ranking():
    omega = torch.ones(5)
    score = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    updated = update_weights(omega, score, learning_rate=0.1)
    assert torch.isclose(updated.sum(), torch.tensor(5.0), atol=1e-6)
    assert torch.equal(torch.argsort(updated), torch.argsort(score))


def test_round_boundaries_are_even_and_end_at_epoch_budget():
    assert round_boundaries(100, 10) == list(range(10, 101, 10))
    assert round_boundaries(7, 3) == [3, 5, 7]


def test_spearman_rank_correlation_tracks_order():
    first = torch.tensor([3.0, -1.0, 2.0, 0.0])
    assert np.isclose(spearman_rank_correlation(first, first), 1.0)
    assert np.isclose(spearman_rank_correlation(first, -first), -1.0)
