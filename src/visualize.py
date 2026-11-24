# 可视化工具：训练曲线、混淆矩阵、误分类样本、数据增强、Grad-CAM 等。


from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import CIFAR10_MEAN, CIFAR10_STD
from .utils import configure_matplotlib, ensure_dir

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """把归一化后的 CHW 张量还原为 HWC、[0,1] 的 numpy 图像。"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def plot_training_curves(history: Sequence[dict], out_path, title: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train", color=PALETTE[0])
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val", color=PALETTE[1])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train", color=PALETTE[0])
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val", color=PALETTE[1])
    if any("ema_val_acc" in h for h in history):
        axes[1].plot(
            epochs,
            [h.get("ema_val_acc") if h.get("ema_val_acc") is not None else np.nan for h in history],
            label="val (EMA)",
            color=PALETTE[2],
            linestyle="--",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy curve")
    axes[1].legend()

    best = max(history, key=lambda h: max(h["val_acc"], h.get("ema_val_acc") or 0.0))
    best_acc = max(best["val_acc"], best.get("ema_val_acc") or 0.0)
    axes[1].axvline(best["epoch"], color="gray", linestyle=":", alpha=0.7)
    axes[1].annotate(
        f"best {best_acc:.2f}% @ {best['epoch']}",
        xy=(best["epoch"], best_acc),
        xytext=(5, -18),
        textcoords="offset points",
        fontsize=8,
        color="gray",
    )
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    out_path,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> None:
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Recall (row-normalized)"},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, out_path)


def plot_per_class_metrics(
    report: dict,
    classes: Sequence[str],
    out_path,
    title: str = "Per-class precision / recall / F1",
) -> None:
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, metric in enumerate(metrics):
        vals = [report[c][metric] * 100 for c in classes]
        ax.bar(x + (i - 1) * width, vals, width, label=metric, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(ncol=3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_per_class_accuracy(
    per_class_acc: Sequence[float],
    classes: Sequence[str],
    out_path,
    overall: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [PALETTE[0] if v >= (overall or 0) else PALETTE[3] for v in per_class_acc]
    bars = ax.bar(classes, per_class_acc, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-class accuracy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if overall is not None:
        ax.axhline(overall, color="gray", linestyle="--", label=f"overall {overall:.2f}%")
        ax.legend()
    for bar, v in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_misclassified(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str],
    out_path,
    max_items: int = 24,
    title: str = "Misclassified samples (true / predicted, confidence)",
    sort_by_confidence: bool = True,
) -> int:
    """images: (N,3,32,32) uint8；只画误分类样本，返回实际绘制数量。"""
    wrong = np.where(labels != preds)[0]
    conf = probs[np.arange(len(preds)), preds]
    if sort_by_confidence:
        wrong = wrong[np.argsort(-conf[wrong])]
    wrong = wrong[:max_items]
    if len(wrong) == 0:
        return 0
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        img = np.transpose(images[idx], (1, 2, 0))
        ax.imshow(img)
        ax.set_title(
            f"T:{classes[labels[idx]]}\nP:{classes[preds[idx]]} ({conf[idx]:.2f})",
            fontsize=7,
            color="crimson",
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)
    return len(wrong)


def plot_confidence_histogram(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    out_path,
    bins: int = 20,
) -> None:
    conf = probs[np.arange(len(preds)), preds]
    correct = labels == preds
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[correct], bins=bins, alpha=0.7, label="correct", color=PALETTE[2])
    ax.hist(conf[~correct], bins=bins, alpha=0.7, label="wrong", color=PALETTE[3])
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_top_confusions(
    cm: np.ndarray, classes: Sequence[str], out_path, k: int = 12, title: str = "Top confusion pairs"
) -> None:
    cm = np.asarray(cm)
    pairs = []
    for i in range(cm.shape[0]):
        total = max(cm[i].sum(), 1)
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], i, j, cm[i, j] / total))
    pairs.sort(reverse=True)
    pairs = pairs[:k]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pairs) + 1.2))
    labels = [f"{classes[i]} -> {classes[j]}" for _, i, j, _ in pairs]
    values = [p[0] for p in pairs]
    ax.barh(labels[::-1], values[::-1], color=PALETTE[3])
    for y, (count, _, _, ratio) in enumerate(pairs[::-1]):
        ax.text(count, y, f" {count} ({ratio * 100:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Number of samples")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out_path)


def plot_augmentations(
    image_uint8: np.ndarray,
    transform_fns: dict,
    classes: Sequence[str] | None = None,
    label: int | None = None,
    out_path=None,
) -> None:
    """可视化同一张图在不同增强策略下的效果。

    ``transform_fns``: {策略名: 可调用对象(uint8 CHW -> float CHW)}
    """
    n = len(transform_fns) + 1
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(np.transpose(image_uint8, (1, 2, 0)))
    axes[0].set_title("original" + (f"\n{classes[label]}" if classes is not None and label is not None else ""), fontsize=8)
    for ax, (name, fn) in zip(axes[1:], transform_fns.items()):
        img = fn(torch.from_numpy(image_uint8))
        ax.imshow(denormalize(img))
        ax.set_title(name, fontsize=8)
    fig.suptitle("Data augmentation examples", fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)


def plot_dataset_samples(
    images: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    out_path,
    n_per_class: int = 8,
) -> None:
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 1.2, n_classes * 1.25))
    for c in range(n_classes):
        idx = np.where(labels == c)[0][:n_per_class]
        for j in range(n_per_class):
            ax = axes[c, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(idx):
                ax.imshow(np.transpose(images[idx[j]], (1, 2, 0)))
            if j == 0:
                ax.text(
                    -0.12,
                    0.5,
                    classes[c],
                    transform=ax.transAxes,
                    fontsize=8,
                    rotation=90,
                    ha="center",
                    va="center",
                )
    fig.suptitle("CIFAR-10 samples (training set)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


def plot_bar_comparison(
    names: Sequence[str],
    values: Sequence[float],
    out_path,
    ylabel: str = "Test accuracy (%)",
    title: str = "",
    errors: Sequence[float] | None = None,
    color: str = PALETTE[0],
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7), 4.2))
    bars = ax.bar(range(len(names)), values, color=color)
    if errors is not None:
        ax.errorbar(range(len(names)), values, yerr=errors, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.2f}", ha="center", fontsize=8)
    low = min(values)
    ax.set_ylim(max(0, low - 5), 100)
    fig.tight_layout()
    _save(fig, out_path)


def plot_grouped_bars(
    groups: Sequence[str],
    series: dict,
    out_path,
    ylabel: str = "Test accuracy (%)",
    title: str = "",
) -> None:
    x = np.arange(len(groups))
    n = len(series)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.1), 4.2))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, values, width, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_feature_maps(
    feature_maps: torch.Tensor,
    out_path,
    layer_name: str = "layer3",
    max_channels: int = 16,
) -> None:
    """feature_maps: (1, C, H, W)。"""
    fm = feature_maps.detach().cpu()[0]
    c, h, w = fm.shape
    n = min(c, max_channels)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.5))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = fm[i].numpy()
            ax.imshow(img, cmap="viridis")
            ax.set_title(f"ch {i}", fontsize=7)
    fig.suptitle(f"Feature maps: {layer_name} ({c} channels, {h}x{w})", fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)
