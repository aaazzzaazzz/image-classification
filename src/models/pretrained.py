# 基于 torchvision 的预训练模型微调（ImageNet 权重）。


from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

# 把 torchvision 的预训练权重缓存放到项目目录下，避免系统目录权限问题
_TORCH_HOME = Path(__file__).resolve().parents[2] / ".cache" / "torch"
os.environ.setdefault("TORCH_HOME", str(_TORCH_HOME))
_TORCH_HOME.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _maybe_download_weights(weights_enum, pretrained: bool):
    if not pretrained:
        return None
    return weights_enum.IMAGENET1K_V1


def resnet18_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = _maybe_download_weights(tvm.ResNet18_Weights, pretrained)
    model = tvm.resnet18(weights=weights)
    # 32×32 输入：保留 7×7 卷积但不再下采样，去掉 maxpool，特征图保持 32×32 起步
    model.conv1.stride = (1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet34_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = _maybe_download_weights(tvm.ResNet34_Weights, pretrained)
    model = tvm.resnet34(weights=weights)
    model.conv1.stride = (1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def mobilenet_v3_small_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = _maybe_download_weights(tvm.MobileNet_V3_Small_Weights, pretrained)
    model = tvm.mobilenet_v3_small(weights=weights)
    model.features[0][0].stride = (1, 1)  # 3×3 stem 不再下采样
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def mobilenet_v3_large_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = _maybe_download_weights(tvm.MobileNet_V3_Large_Weights, pretrained)
    model = tvm.mobilenet_v3_large(weights=weights)
    model.features[0][0].stride = (1, 1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def vgg16_bn_cifar(num_classes: int = 10, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = _maybe_download_weights(tvm.VGG16_BN_Weights, pretrained)
    model = tvm.vgg16_bn(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


def vit_b_16_cifar(num_classes: int = 10, pretrained: bool = True, image_size: int = 224) -> nn.Module:
    """ViT-B/16 微调版本。

    CIFAR-10 为 32×32；默认直接用 32×32 输入，并把预训练的 14×14
    positional embedding 双三次插值到 2×2，以保留 196→4 的 patch 对应关系。
    如果显式传入 image_size=224，则按原始 224×224 输入处理。
    """
    weights = _maybe_download_weights(tvm.ViT_B_16_Weights, pretrained)
    if image_size == 224:
        model = tvm.vit_b_16(weights=weights, image_size=image_size)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    model = tvm.vit_b_16(weights=None, image_size=image_size)
    if weights is not None:
        state = weights.get_state_dict()
        key = "encoder.pos_embedding"
        if key in state:
            pos = state[key]
            cls_token, patch_pos = pos[:, :1], pos[:, 1:]
            grid = int(round(patch_pos.shape[1] ** 0.5))
            patch_pos = patch_pos.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
            target = image_size // 16
            patch_pos = torch.nn.functional.interpolate(
                patch_pos, size=(target, target), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target * target, -1)
            state[key] = torch.cat([cls_token, patch_pos], dim=1)
        # 分类头改为 10 类：去掉预训练 head 的键，避免尺寸不匹配
        state = {k: v for k, v in state.items() if not k.startswith("heads.head.")}
        model.load_state_dict(state, strict=False)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
