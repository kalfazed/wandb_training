"""Small BEV CNN detector: ResNet-style backbone + CenterHead."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + x)


class BEVBackbone(nn.Module):
    """Lightweight 2D CNN on a multi-channel BEV tensor.

    We keep full BEV resolution (no stride-2 downsampling) so the heatmap
    head matches the target grid one-to-one — easier to debug in exercise 01.
    A production detector would downsample and build targets at ``output_stride``.
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = ConvBNReLU(in_channels, c, kernel=3, stride=1)
        self.body = nn.Sequential(
            ResBlock(c),
            ResBlock(c),
            ResBlock(c),
            ResBlock(c),
        )
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.stem(x))


class CenterHead(nn.Module):
    """Predict heatmap logits + dense regression map."""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        mid = max(in_channels // 2, 32)
        self.shared = nn.Sequential(
            ConvBNReLU(in_channels, mid),
            ResBlock(mid),
        )
        self.heatmap = nn.Conv2d(mid, num_classes, kernel_size=1)
        self.regression = nn.Conv2d(mid, 8, kernel_size=1)

        # Focal loss works better when the background logit starts negative.
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        return self.heatmap(feat), self.regression(feat)


class BEVDetector(nn.Module):
    """End-to-end BEV detector for exercise 01."""

    def __init__(self, in_channels: int = 5, num_classes: int = 1, base_channels: int = 32):
        super().__init__()
        self.backbone = BEVBackbone(in_channels, base_channels)
        self.head = CenterHead(self.backbone.out_channels, num_classes)

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(bev)
        heatmap_logits, reg = self.head(feat)
        return {"heatmap": heatmap_logits, "reg": reg}
