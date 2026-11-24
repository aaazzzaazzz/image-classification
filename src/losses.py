# 损失函数：交叉熵、标签平滑、Focal Loss。


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵（Szegedy et al., 2016）。

    将 one-hot 目标平滑为 ``(1-eps) * onehot + eps / C``，抑制过度自信，
    通常能带来 0.2%~0.5% 的泛化提升。
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing 必须在 [0, 1) 内")
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """Focal Loss（Lin et al., 2017），降低易分类样本的权重。

    ``alpha`` 为类别平衡因子（标量形式），``gamma`` 控制聚焦程度。
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * float(self.alpha)
        return loss.mean()


def build_criterion(cfg: dict) -> nn.Module:
    """根据 ``cfg["train"]`` 构建损失函数。"""
    name = str(cfg.get("loss", "ce")).lower()
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    if name in ("ls", "label_smoothing", "smooth"):
        return LabelSmoothingCrossEntropy(float(cfg.get("label_smoothing", 0.1)))
    if name == "focal":
        alpha = cfg.get("focal_alpha", 0.25)
        return FocalLoss(gamma=float(cfg.get("focal_gamma", 2.0)), alpha=alpha)
    raise ValueError(f"未知损失函数: {name}")
