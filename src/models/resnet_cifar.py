# 适用于 CIFAR-10（32×32）的 ResNet 系列，可选嵌入 SE / CBAM / ECA 注意力。


from __future__ import annotations

import torch
import torch.nn as nn

from .attention import build_attention


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        attention: str | None = None,
        reduction: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.attn = build_attention(attention, planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attn is not None:
            out = self.attn(out)
        out = self.dropout(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        depth: int = 20,
        num_classes: int = 10,
        attention: str | None = None,
        reduction: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"depth 必须满足 6n+2，收到 {depth}")
        n = (depth - 2) // 6
        self.in_planes = 16
        self.attention = attention
        self.stem = nn.Sequential(
            conv3x3(3, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16, n, stride=1, attention=attention, reduction=reduction)
        self.layer2 = self._make_layer(32, n, stride=2, attention=attention, reduction=reduction)
        self.layer3 = self._make_layer(64, n, stride=2, attention=attention, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 残差分支最后一个 BN 的 gamma 置零，使初始网络接近恒等映射
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
        attention: str | None,
        reduction: int,
    ) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=s,
                    attention=attention,
                    reduction=reduction,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
