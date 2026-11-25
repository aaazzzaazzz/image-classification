# 全部实验配置。


from __future__ import annotations

from copy import deepcopy

from .utils import deep_update

BASE = {
    "name": "base",
    "seed": 42,
    "data": {
        "root": ".",
        "val_size": 5000,
        "batch_size": 128,
        "num_workers": 4,
        "aug": "basic",
        "cutout": 0,
        "randaug_num_ops": 2,
        "randaug_magnitude": 9,
        "limit_train": 0,
        "limit_val": 0,
        "limit_test": 0,
    },
    "model": {
        "name": "resnet20",
        "num_classes": 10,
        "attention": None,
        "attention_reduction": 16,
        "dropout": 0.0,
        "pretrained": False,
    },
    "train": {
        "epochs": 60,
        "batch_size": 128,
        "optimizer": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 5e-4,
        "scheduler": "cosine",
        "warmup_epochs": 1,
        "eta_min": 0.0,
        "step_milestones": [0.5, 0.75],
        "step_gamma": 0.1,
        "loss": "ce",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "mixup": "none",
        "mixup_alpha": 0.2,
        "mixup_prob": 0.5,
        "ema_decay": 0.0,
        "grad_clip": 0.0,
        "amp": True,
        "early_stop_patience": 20,
        "early_stop_min_delta": 0.0,
        "deterministic": False,
        "test_tta": False,
        "log_interval": 100,
    },
}


def make(name: str, description: str, **overrides) -> dict:
    cfg = deepcopy(BASE)
    cfg["name"] = name
    cfg["description"] = description
    cfg = deep_update(cfg, overrides)
    return cfg


EXPERIMENTS = {
    # ------------------------------------------------------------------ #
    # 基础模型 / 模型对比
    # ------------------------------------------------------------------ #
    "simple_cnn": make(
        "simple_cnn",
        "自定义 6 层卷积 CNN + BN + Dropout，基础数据增强，60 epoch",
        model={"name": "simple_cnn", "width": 64, "dropout": 0.3},
    ),
    "resnet20_basic": make(
        "resnet20_basic",
        "CIFAR-ResNet-20 基线（RandomCrop+Flip, CE, cosine, 60 epoch）",
    ),
    "resnet18_scratch": make(
        "resnet18_scratch",
        "ResNet-18（32x32 适配、随机初始化）从零训练 60 epoch",
        model={"name": "resnet18_cifar", "pretrained": False},
        train={"warmup_epochs": 2, "epochs": 60},
    ),
    # ------------------------------------------------------------------ #
    # 预训练模型微调
    # ------------------------------------------------------------------ #
    "resnet18_pretrained": make(
        "resnet18_pretrained",
        "ImageNet 预训练 ResNet-18 微调（stem 适配 32x32），25 epoch",
        model={"name": "resnet18_cifar", "pretrained": True},
        train={"epochs": 25, "lr": 0.01, "weight_decay": 1e-4, "warmup_epochs": 2},
    ),
    "mobilenet_v3_pretrained": make(
        "mobilenet_v3_pretrained",
        "ImageNet 预训练 MobileNetV3-Small 微调，30 epoch",
        model={"name": "mobilenet_v3_small", "pretrained": True},
        train={"epochs": 30, "lr": 0.01, "weight_decay": 1e-4, "warmup_epochs": 2},
    ),
    # ------------------------------------------------------------------ #
    # 数据增强消融（固定 ResNet-20 与其余超参数）
    # ------------------------------------------------------------------ #
    "resnet20_aug_none": make(
        "resnet20_aug_none",
        "消融：无数据增强（仅归一化）",
        data={"aug": "none"},
    ),
    "resnet20_aug_cutout": make(
        "resnet20_aug_cutout",
        "消融：RandomCrop+Flip + Cutout(16)",
        data={"aug": "cutout", "cutout": 16},
    ),
    "resnet20_aug_randaugment": make(
        "resnet20_aug_randaugment",
        "消融：RandomCrop+Flip + RandAugment(n=2, m=9)",
        data={"aug": "randaugment", "randaug_num_ops": 2, "randaug_magnitude": 9},
    ),
    "resnet20_mixup": make(
        "resnet20_mixup",
        "消融：基础增强 + Mixup(alpha=0.2, p=0.5)",
        train={"mixup": "mixup", "mixup_alpha": 0.2, "mixup_prob": 0.5},
    ),
    # ------------------------------------------------------------------ #
    # 损失函数改进
    # ------------------------------------------------------------------ #
    "resnet20_label_smoothing": make(
        "resnet20_label_smoothing",
        "消融：标签平滑交叉熵（eps=0.1）",
        train={"loss": "ls", "label_smoothing": 0.1},
    ),
    "resnet20_focal": make(
        "resnet20_focal",
        "消融：Focal Loss（gamma=2, alpha=0.25）",
        train={"loss": "focal", "focal_gamma": 2.0, "focal_alpha": 0.25},
    ),
    # ------------------------------------------------------------------ #
    # 注意力机制
    # ------------------------------------------------------------------ #
    "resnet20_se": make(
        "resnet20_se",
        "SE-ResNet-20（通道注意力）",
        model={"attention": "se"},
    ),
    "resnet20_cbam": make(
        "resnet20_cbam",
        "CBAM-ResNet-20（通道+空间注意力）",
        model={"attention": "cbam"},
    ),
    "simple_cnn_se": make(
        "simple_cnn_se",
        "SimpleCNN + SE 通道注意力（每个 stage 后）",
        model={"name": "simple_cnn", "attention": "se"},
        train={"epochs": 140, "scheduler_epochs": 60, "early_stop_patience": 20},
    ),
    # ------------------------------------------------------------------ #
    # 学习率调度策略
    # ------------------------------------------------------------------ #
    "resnet20_step": make(
        "resnet20_step",
        "消融：MultiStep 学习率调度（50%/75% 处衰减 0.1）",
        train={"scheduler": "step", "step_milestones": [0.5, 0.75]},
    ),
    "resnet20_onecycle": make(
        "resnet20_onecycle",
        "消融：OneCycle 学习率调度（max_lr=0.1, pct_start=0.3）",
        train={"scheduler": "onecycle", "warmup_epochs": 0, "onecycle_pct_start": 0.3},
    ),
    "resnet20_sgdr": make(
        "resnet20_sgdr",
        "SGDR：CosineAnnealingWarmRestarts（T_0=20, T_mult=2）",
        train={
            "scheduler": "cosine_restarts",
            "restart_t0": 20,
            "restart_t_mult": 2,
            "epochs": 140,
            "scheduler_epochs": 60,
            "early_stop_patience": 20,
        },
    ),
    "resnet20_warmup5": make(
        "resnet20_warmup5",
        "Cosine + 5 epoch warmup（更长预热，降低前期震荡）",
        train={
            "warmup_epochs": 5,
            "epochs": 140,
            "scheduler_epochs": 60,
            "early_stop_patience": 20,
        },
    ),
    # ------------------------------------------------------------------ #
    # 综合改进模型（用于最终性能、Grad-CAM 与集成）
    # ------------------------------------------------------------------ #
    "resnet20_final": make(
        "resnet20_final",
        "综合改进：RandAugment + 标签平滑 + Cosine + EMA，120 epoch",
        data={"aug": "randaugment"},
        train={
            "epochs": 120,
            "loss": "ls",
            "label_smoothing": 0.1,
            "ema_decay": 0.999,
            "early_stop_patience": 40,
            "test_tta": True,
        },
    ),
    "se_resnet20_final": make(
        "se_resnet20_final",
        "综合改进：SE 注意力 + RandAugment + 标签平滑 + EMA，120 epoch",
        model={"attention": "se"},
        data={"aug": "randaugment"},
        train={
            "epochs": 120,
            "loss": "ls",
            "label_smoothing": 0.1,
            "ema_decay": 0.999,
            "early_stop_patience": 40,
            "test_tta": True,
        },
    ),
    "resnet18_pretrained_final": make(
        "resnet18_pretrained_final",
        "预训练 ResNet-18 精调：Cutout + 标签平滑 + EMA，30 epoch",
        model={"name": "resnet18_cifar", "pretrained": True},
        data={"aug": "cutout", "cutout": 8},
        train={
            "epochs": 30,
            "lr": 0.005,
            "weight_decay": 1e-4,
            "warmup_epochs": 2,
            "loss": "ls",
            "label_smoothing": 0.1,
            "ema_decay": 0.999,
            "test_tta": True,
            "early_stop_patience": 15,
        },
    ),
    "resnet20_kd": make(
        "resnet20_kd",
        "ResNet-20 + 知识蒸馏损失：教师为 95.65% 的预训练 ResNet-18，T=4、alpha=0.3",
        data={"aug": "basic"},
        train={
            "epochs": 160,
            "scheduler_epochs": 120,
            "loss": "ce",
            "ema_decay": 0.999,
            "early_stop_patience": 20,
            "distill_probs": "outputs/teacher_probs/resnet18_train_t4.npy",
            "distill_alpha": 0.7,
            "distill_temp": 4.0,
            "test_tta": True,
        },
    ),
}


def get_experiment(name: str) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"未知实验 {name}，可用: {', '.join(EXPERIMENTS)}")
    return deepcopy(EXPERIMENTS[name])
