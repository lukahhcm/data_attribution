from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def initialize_distributed(timeout_minutes: int = 30) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def wrap_ddp(module: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if not context.enabled:
        return module
    kwargs = {"find_unused_parameters": False, "broadcast_buffers": True}
    if context.device.type == "cuda":
        kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return DistributedDataParallel(module, **kwargs)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_tensor(tensor: torch.Tensor, source: int = 0) -> torch.Tensor:
    if dist.is_initialized():
        dist.broadcast(tensor, src=source)
    return tensor


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    all_reduce_sum(value)
    if dist.is_initialized():
        value /= dist.get_world_size()
    return value


def distributed_sampler(dataset, context: DistributedContext, shuffle: bool, seed: int):
    if not context.enabled:
        return None
    return torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )

