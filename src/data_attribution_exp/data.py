from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class IndexedViews(Dataset):
    """A dataset with stable local indices and separate scoring/training views."""

    def __init__(
        self,
        base: Dataset,
        source_indices: np.ndarray,
        labels: np.ndarray,
        score_transform: Callable,
        train_transform: Callable,
    ) -> None:
        self.base = base
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.score_transform = score_transform
        self.train_transform = train_transform
        if len(self.source_indices) != len(self.labels):
            raise ValueError("source_indices and labels must have the same length")

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, local_index: int):
        source_index = int(self.source_indices[local_index])
        image, _ = self.base[source_index]
        score_view = self.score_transform(image)
        train_view = self.train_transform(image)
        return int(local_index), score_view, train_view, int(self.labels[local_index])


@dataclass
class DataBundle:
    candidate: IndexedViews
    validation: IndexedViews
    development: IndexedViews
    test: IndexedViews
    num_classes: int
    input_shape: tuple[int, ...]
    split_indices: dict[str, np.ndarray]
    clean_candidate_labels: np.ndarray
    candidate_labels: np.ndarray
    corrupted_mask: np.ndarray


def _targets(dataset: Dataset) -> np.ndarray:
    values = getattr(dataset, "targets")
    if isinstance(values, torch.Tensor):
        return values.cpu().numpy().astype(np.int64)
    return np.asarray(values, dtype=np.int64)


def stratified_split(
    labels: np.ndarray,
    sizes: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sum(sizes) != len(labels):
        raise ValueError(f"Split sizes {sizes} do not sum to {len(labels)}")
    rng = np.random.default_rng(seed)
    class_indices = {c: rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)}
    fractions = np.asarray(sizes, dtype=np.float64) / len(labels)
    buckets = [[], [], []]
    for indices in class_indices.values():
        first = int(round(len(indices) * fractions[0]))
        second = int(round(len(indices) * fractions[1]))
        buckets[0].extend(indices[:first])
        buckets[1].extend(indices[first : first + second])
        buckets[2].extend(indices[first + second :])

    # Rounding can miss the requested global sizes. Move random items between
    # buckets without changing the disjointness invariant.
    for target_bucket in range(2):
        while len(buckets[target_bucket]) < sizes[target_bucket]:
            donor = max(range(3), key=lambda i: len(buckets[i]) - sizes[i])
            buckets[target_bucket].append(buckets[donor].pop())
        while len(buckets[target_bucket]) > sizes[target_bucket]:
            receiver = min(range(3), key=lambda i: len(buckets[i]) - sizes[i])
            buckets[receiver].append(buckets[target_bucket].pop())
    arrays = tuple(rng.permutation(np.asarray(bucket, dtype=np.int64)) for bucket in buckets)
    assert tuple(map(len, arrays)) == sizes
    return arrays  # type: ignore[return-value]


def symmetric_label_noise(
    labels: np.ndarray,
    num_classes: int,
    rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= rate < 1.0:
        raise ValueError("noise rate must be in [0, 1)")
    noisy = np.asarray(labels, dtype=np.int64).copy()
    mask = np.zeros(len(noisy), dtype=bool)
    if rate == 0.0:
        return noisy, mask
    rng = np.random.default_rng(seed)
    count = int(round(rate * len(noisy)))
    chosen = rng.choice(len(noisy), size=count, replace=False)
    offsets = rng.integers(1, num_classes, size=count)
    noisy[chosen] = (noisy[chosen] + offsets) % num_classes
    mask[chosen] = True
    assert np.all(noisy[chosen] != labels[chosen])
    return noisy, mask


def _mnist_transforms():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return transform, transform


def _cifar_transforms(name: str):
    stats = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }
    mean, std = stats[name]
    score = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return score, train


def build_data(config: dict, download: bool = True) -> DataBundle:
    name = str(config["name"]).lower()
    root = str(Path(config.get("root", "./data")).expanduser())
    split_seed = int(config.get("split_seed", 1))
    noise_seed = int(config.get("noise_seed", split_seed))
    noise_rate = float(config.get("noise_rate", 0.0))

    if name == "mnist":
        train_base = datasets.MNIST(root, train=True, download=download, transform=None)
        test_base = datasets.MNIST(root, train=False, download=download, transform=None)
        split_sizes = (50_000, 5_000, 5_000)
        num_classes = 10
        input_shape = (1, 28, 28)
        score_transform, train_transform = _mnist_transforms()
    elif name == "cifar10":
        train_base = datasets.CIFAR10(root, train=True, download=download, transform=None)
        test_base = datasets.CIFAR10(root, train=False, download=download, transform=None)
        split_sizes = (40_000, 5_000, 5_000)
        num_classes = 10
        input_shape = (3, 32, 32)
        score_transform, train_transform = _cifar_transforms(name)
    elif name == "cifar100":
        train_base = datasets.CIFAR100(root, train=True, download=download, transform=None)
        test_base = datasets.CIFAR100(root, train=False, download=download, transform=None)
        split_sizes = (40_000, 5_000, 5_000)
        num_classes = 100
        input_shape = (3, 32, 32)
        score_transform, train_transform = _cifar_transforms(name)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    all_labels = _targets(train_base)
    candidate_idx, validation_idx, development_idx = stratified_split(
        all_labels, split_sizes, split_seed
    )
    clean_candidate_labels = all_labels[candidate_idx]
    candidate_labels, corrupted_mask = symmetric_label_noise(
        clean_candidate_labels, num_classes, noise_rate, noise_seed
    )

    candidate = IndexedViews(
        train_base, candidate_idx, candidate_labels, score_transform, train_transform
    )
    validation = IndexedViews(
        train_base,
        validation_idx,
        all_labels[validation_idx],
        score_transform,
        train_transform,
    )
    development = IndexedViews(
        train_base,
        development_idx,
        all_labels[development_idx],
        score_transform,
        score_transform,
    )
    test_indices = np.arange(len(test_base), dtype=np.int64)
    test_labels = _targets(test_base)
    test = IndexedViews(
        test_base, test_indices, test_labels, score_transform, score_transform
    )
    return DataBundle(
        candidate=candidate,
        validation=validation,
        development=development,
        test=test,
        num_classes=num_classes,
        input_shape=input_shape,
        split_indices={
            "candidate": candidate_idx,
            "validation": validation_idx,
            "development": development_idx,
        },
        clean_candidate_labels=clean_candidate_labels,
        candidate_labels=candidate_labels,
        corrupted_mask=corrupted_mask,
    )


def save_data_artifacts(bundle: DataBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "data_indices_and_noise.npz",
        **bundle.split_indices,
        clean_candidate_labels=bundle.clean_candidate_labels,
        candidate_labels=bundle.candidate_labels,
        corrupted_mask=bundle.corrupted_mask,
    )
