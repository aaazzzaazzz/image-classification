# 模型工厂：按配置构建模型，并给出 Grad-CAM 的目标层。


from __future__ import annotations

import re
import warnings
from typing import Optional

import torch.nn as nn

from .pretrained import (
    mobilenet_v3_large_cifar,
    mobilenet_v3_small_cifar,
    resnet18_cifar,
    resnet34_cifar,
    vgg16_bn_cifar,
    vit_b_16_cifar,
)
from .resnet_cifar import ResNetCIFAR
from .simple_cnn import SimpleCNN

__all__ = [
    "build_model",
    "get_gradcam_target",
    "SimpleCNN",
    "ResNetCIFAR",
    "resnet18_cifar",
    "mobilenet_v3_small_cifar",
]


def build_model(model_cfg: dict) -> nn.Module:
    """根据 ``cfg["model"]`` 构建模型。

    ``model.name`` 支持:
        simple_cnn / resnet20 / resnet32 / resnet44 / resnet56 / resnet110
        resnet18_cifar / resnet34_cifar / mobilenet_v3_small / mobilenet_v3_large
        vgg16_bn_cifar / vit_b_16_cifar
    """
    cfg = dict(model_cfg or {})
    name = str(cfg.get("name", "resnet20")).lower()
    num_classes = int(cfg.get("num_classes", 10))
    attention = cfg.get("attention") or None
    reduction = int(cfg.get("attention_reduction", 16))
    dropout = float(cfg.get("dropout", 0.0))

    if name in ("simple_cnn", "cnn", "vgg_cnn"):
        return SimpleCNN(
            num_classes=num_classes,
            width=int(cfg.get("width", 64)),
            dropout=float(cfg.get("dropout", 0.3)),
        )

    m = re.fullmatch(r"resnet(\d+)", name)
    if m:
        depth = int(m.group(1))
        return ResNetCIFAR(
            depth=depth,
            num_classes=num_classes,
            attention=attention,
            reduction=reduction,
            dropout=dropout,
        )

    pretrained = bool(cfg.get("pretrained", False))
    try:
        if name == "resnet18_cifar":
            return resnet18_cifar(num_classes, pretrained=pretrained)
        if name == "resnet34_cifar":
            return resnet34_cifar(num_classes, pretrained=pretrained)
        if name == "mobilenet_v3_small":
            return mobilenet_v3_small_cifar(num_classes, pretrained=pretrained)
        if name == "mobilenet_v3_large":
            return mobilenet_v3_large_cifar(num_classes, pretrained=pretrained)
        if name == "vgg16_bn_cifar":
            return vgg16_bn_cifar(num_classes, pretrained=pretrained, dropout=dropout or 0.3)
        if name == "vit_b_16_cifar":
            return vit_b_16_cifar(
                num_classes,
                pretrained=pretrained,
                image_size=int(cfg.get("image_size", 224)),
            )
    except Exception as exc:  # 离线环境无法下载权重时，退化为随机初始化并给出警告
        if not pretrained:
            raise
        warnings.warn(f"加载预训练权重失败（{exc}），将使用随机初始化继续。")
        return build_model({**cfg, "pretrained": False})

    raise ValueError(f"未知模型: {name}")


def _find_last_conv(module: nn.Module) -> Optional[nn.Module]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def get_gradcam_target(model: nn.Module, model_name: str = "") -> nn.Module:
    """返回 Grad-CAM 使用的目标层（尽量选最后一个语义卷积层）。"""
    if hasattr(model, "layer4"):  # torchvision ResNet
        return model.layer4[-1]
    if hasattr(model, "layer3"):  # CIFAR ResNet
        return model.layer3[-1]
    if hasattr(model, "features"):  # SimpleCNN / MobileNet / VGG
        features = model.features
        if model_name.startswith("simple_cnn"):
            conv = _find_last_conv(features)
            return conv if conv is not None else features[-1]
        if isinstance(features, nn.Sequential) and len(features) > 0:
            return features[-1]
        return features
    conv = _find_last_conv(model)
    if conv is None:
        raise ValueError("无法自动定位 Grad-CAM 目标层，请在配置中显式指定。")
    return conv
