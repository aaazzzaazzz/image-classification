# 自定义卷积神经网络（基础要求中的 CNN 模型）。


from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """VGG 风格的轻量 CNN。

    结构：3 个 stage，每个 stage 两个 3×3 卷积 + BN + ReLU，然后 2×2 最大池化；
    最后全局平均池化 + Dropout + 全连接分类器。
    该模型作为“从零训练”的基线，参数量约 1M。
    """

    def __init__(self, num_classes: int = 10, width: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        w = int(width)
        self.features = nn.Sequential(
            conv_bn_relu(3, w),
            conv_bn_relu(w, w),
            nn.MaxPool2d(2),  # 16x16
            conv_bn_relu(w, w * 2),
            conv_bn_relu(w * 2, w * 2),
            nn.MaxPool2d(2),  # 8x8
            conv_bn_relu(w * 2, w * 4),
            conv_bn_relu(w * 4, w * 4),
            nn.MaxPool2d(2),  # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 4, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
