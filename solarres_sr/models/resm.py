from __future__ import annotations

import torch
import torch.nn as nn

from .common import ChannelAttention, PixelShuffleUpsampler, default_conv


class EdgeAwareGate(nn.Module):
    """Learn a soft high-frequency gate over residual solar features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.GELU(),
            default_conv(channels, channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class RESMBlock(nn.Module):
    """Residual Edge-aware Solar Module block."""

    def __init__(self, channels: int, expansion: int = 2, reduction: int = 16, res_scale: float = 0.2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.body = nn.Sequential(
            default_conv(channels, hidden, 3),
            nn.GELU(),
            default_conv(hidden, channels, 3),
            ChannelAttention(channels, reduction=reduction),
            EdgeAwareGate(channels),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class RESMNet(nn.Module):
    """Residual Edge-aware Solar Module Network for image super-resolution.

    RESM is designed as a recruiter-friendly, solar-specific architecture:
    compact enough to train quickly, but expressive enough to emphasize
    filamentary and edge-like structure common in solar magnetograms.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale: int = 4,
        num_features: int = 64,
        num_blocks: int = 10,
        expansion: int = 2,
        reduction: int = 16,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.head = default_conv(in_channels, num_features, 3)
        self.body = nn.Sequential(
            *[
                RESMBlock(
                    num_features,
                    expansion=expansion,
                    reduction=reduction,
                    res_scale=res_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.body_tail = default_conv(num_features, num_features, 3)
        self.upsampler = PixelShuffleUpsampler(num_features, scale, activation=nn.GELU)
        self.tail = default_conv(num_features, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)
        deep = self.body_tail(self.body(shallow))
        return self.tail(self.upsampler(shallow + deep))
