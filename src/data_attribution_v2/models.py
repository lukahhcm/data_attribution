from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class MNISTMLP(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 300) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RHOPaperMNISTMLP(nn.Module):
    """Three-layer 512-unit ReLU MLP used by the released RHO-Loss code."""

    def __init__(self, num_classes: int = 10, hidden: int = 512) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        width = 28 * 28
        for _ in range(3):
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def cifar_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_model(config: dict, num_classes: int) -> nn.Module:
    name = str(config["name"]).lower()
    if name in {"mnist_mlp", "mlp"}:
        return MNISTMLP(num_classes=num_classes, hidden=int(config.get("hidden", 300)))
    if name in {"rho_paper_mnist_mlp", "mnist_mlp512"}:
        return RHOPaperMNISTMLP(
            num_classes=num_classes, hidden=int(config.get("hidden", 512))
        )
    if name in {"resnet18", "cifar_resnet18"}:
        return cifar_resnet18(num_classes)
    raise ValueError(f"Unsupported model: {name}")


def maybe_convert_sync_batchnorm(
    model: nn.Module, enabled: bool, distributed: bool
) -> nn.Module:
    if enabled and distributed:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model
