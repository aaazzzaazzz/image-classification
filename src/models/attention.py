# 注意力模块：SE、CBAM、ECA，可直接插入卷积残差块。


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation（通道注意力）。

    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


class ResidualSEBlock(nn.Module):
    """零初始化残差 SE：gamma=0 时等价于恒等映射，训练中再学习门控强度。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se.fc(self.se.avg_pool(x))
        return x * (1.0 + self.gamma * (scale - 1.0))


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力分支（同时使用平均池化与最大池化）。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力分支。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module（通道 + 空间）。

    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ECABlock(nn.Module):
    """ECA-Net 的轻量通道注意力（1D 卷积，无降维）。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


def build_attention(name: str | None, channels: int, reduction: int = 16) -> nn.Module | None:
    if not name:
        return None
    name = str(name).lower()
    if name in ("none", "identity", "null"):
        return None
    if name == "se":
        return SEBlock(channels, reduction)
    if name in ("se_res", "residual_se"):
        return ResidualSEBlock(channels, reduction)
    if name == "cbam":
        return CBAM(channels, reduction)
    if name == "eca":
        return ECABlock(channels)
    raise ValueError(f"未知注意力类型: {name}")
