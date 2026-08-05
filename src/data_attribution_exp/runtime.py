from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .distributed import DistributedContext, distributed_sampler


def output_directory(config: dict) -> Path:
    root = Path(config["experiment"].get("output_root", "outputs"))
    dataset = str(config["dataset"]["name"])
    method = str(config["experiment"]["method"])
    noise = float(config["dataset"].get("noise_rate", 0.0))
    retention = float(config["selection"].get("retention", 1.0))
    seed = int(config["experiment"].get("seed", 1))
    base = root / dataset / f"noise_{noise:g}" / f"retention_{retention:g}" / method
    run_name = config["experiment"].get("run_name")
    if run_name:
        base = base / str(run_name)
    return base / f"seed_{seed}"


def per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"global batch size {global_batch_size} must be divisible by world size {world_size}"
        )
    return global_batch_size // world_size


def make_loader(
    dataset,
    context: DistributedContext,
    global_batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    sampler = distributed_sampler(dataset, context, shuffle=shuffle, seed=seed)
    return DataLoader(
        dataset,
        batch_size=per_rank_batch_size(global_batch_size, context.world_size),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=drop_last,
    )


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def topk_count(num_examples: int, retention: float) -> int:
    if not 0 < retention <= 1:
        raise ValueError("retention must be in (0, 1]")
    return max(1, min(num_examples, int(round(num_examples * retention))))
