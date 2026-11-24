# CIFAR-10 数据读取、分层划分、数据增强与 DataLoader 构建。


from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .utils import CIFAR10_CLASSES, worker_init_fn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]] = {}


# --------------------------------------------------------------------------- #
# 原始数据读取
# --------------------------------------------------------------------------- #
def _find_batch_dir(root: str | Path) -> Path:
    """在 root 及其常见子目录中定位 cifar-10-batches-py。"""
    root = Path(root)
    candidates = [
        root / "cifar-10-batches-py",
        root,
        root / "data" / "cifar-10-batches-py",
        root / "cifar-10-python" / "cifar-10-batches-py",
    ]
    for cand in candidates:
        if (cand / "data_batch_1").exists() and (cand / "test_batch").exists():
            return cand
    raise FileNotFoundError(
        f"未在 {root} 下找到 cifar-10-batches-py（需包含 data_batch_1 与 test_batch）"
    )


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        # CIFAR-10 的 pickle 由 Python 2 生成，键为 bytes
        obj = pickle.load(f, encoding="bytes")
    return obj


def _get(obj: dict, key: str):
    if key in obj:
        return obj[key]
    bkey = key.encode("utf-8")
    if bkey in obj:
        return obj[bkey]
    raise KeyError(f"batch 中缺少字段 {key}")


def load_cifar10(root: str | Path = ".") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """读取 CIFAR-10，返回 ``(x_train, y_train, x_test, y_test, classes)``。

    x 形状为 ``(N, 3, 32, 32)`` 的 uint8 数组（CIFAR 原始存储即为 CHW 顺序）。
    """
    root_path = Path(root).resolve()
    cache_key = str(root_path)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    batch_dir = _find_batch_dir(root_path)
    xs, ys = [], []
    for i in range(1, 6):
        batch = _unpickle(batch_dir / f"data_batch_{i}")
        xs.append(np.asarray(_get(batch, "data"), dtype=np.uint8))
        ys.append(np.asarray(_get(batch, "labels"), dtype=np.int64))
    x_train = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32)
    y_train = np.concatenate(ys, axis=0)

    test = _unpickle(batch_dir / "test_batch")
    x_test = np.asarray(_get(test, "data"), dtype=np.uint8).reshape(-1, 3, 32, 32)
    y_test = np.asarray(_get(test, "labels"), dtype=np.int64)

    classes = list(CIFAR10_CLASSES)
    meta_path = batch_dir / "batches.meta"
    if meta_path.exists():
        meta = _unpickle(meta_path)
        raw = _get(meta, "label_names")
        classes = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw]

    _CACHE[cache_key] = (x_train, y_train, x_test, y_test, classes)
    return _CACHE[cache_key]


def stratified_split(
    labels: np.ndarray, val_size: int = 5000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """按类别分层划分训练/验证集，保证验证集类别分布与训练集一致。"""
    if val_size <= 0:
        idx = np.arange(len(labels))
        return idx, idx[:0]
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    per_class = val_size / num_classes
    val_parts, train_parts = [], []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        cls_idx = rng.permutation(cls_idx)
        n_val = int(round(per_class))
        if c < val_size - n_val * num_classes:
            n_val += 1
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])
    val_idx = np.sort(np.concatenate(val_parts))
    train_idx = np.sort(np.concatenate(train_parts))
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# 数据增强
# --------------------------------------------------------------------------- #
class Cutout(torch.nn.Module):
    """Cutout：随机遮挡一个 length×length 的方形区域（在归一化之后使用，填充 0）。"""

    def __init__(self, length: int = 16) -> None:
        super().__init__()
        self.length = int(length)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.length <= 0:
            return img
        _, h, w = img.shape
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        half = self.length // 2
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x - half), min(w, x + half)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def extra_repr(self) -> str:
        return f"length={self.length}"


def build_transforms(cfg: dict, train: bool) -> v2.Compose:
    """根据配置构建增强流水线。

    支持的 ``data.aug``:
        - ``none``           仅 ToTensor + Normalize
        - ``basic``          RandomCrop(32, padding=4) + RandomHorizontalFlip
        - ``cutout``         basic + Cutout（由 ``data.cutout`` 控制大小）
        - ``randaugment``    basic + RandAugment（由 num_ops / magnitude 控制）
        - ``autoaugment``    basic + AutoAugment(CIFAR10 policy)
        - ``trivialaugment`` basic + TrivialAugmentWide
    """
    aug = str(cfg.get("aug", "basic")).lower()
    ops: list = []
    if train:
        if aug != "none":
            ops.append(v2.RandomCrop(32, padding=4, padding_mode="reflect"))
            ops.append(v2.RandomHorizontalFlip(p=0.5))
        if aug == "randaugment":
            ops.append(
                v2.RandAugment(
                    num_ops=int(cfg.get("randaug_num_ops", 2)),
                    magnitude=int(cfg.get("randaug_magnitude", 9)),
                    fill=128,  # 平移/旋转填充灰色，避免出现不自然的黑边
                )
            )
        elif aug == "autoaugment":
            ops.append(v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10, fill=128))
        elif aug == "trivialaugment":
            ops.append(v2.TrivialAugmentWide(fill=128))
    ops.append(v2.ToDtype(torch.float32, scale=True))
    ops.append(v2.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    if train and float(cfg.get("cutout", 0) or 0) > 0:
        ops.append(Cutout(int(cfg["cutout"])))
    return v2.Compose(ops)


class CIFAR10Dataset(Dataset):
    """只持有 uint8 张量，返回 ``(CHW float 张量, int 标签)``。"""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform: v2.Compose | None = None,
    ) -> None:
        self.images = torch.from_numpy(np.ascontiguousarray(images[indices]))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels[indices]).astype(np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        img = self.images[index]
        label = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SoftLabelDataset(Dataset):
    """在训练集上附加教师 softmax 概率，返回 ``(image, label, soft_target)``。"""

    def __init__(self, base: Dataset, probs: np.ndarray) -> None:
        if len(base) != len(probs):
            raise ValueError(f"教师概率数量 {len(probs)} 与训练集 {len(base)} 不一致")
        self.base = base
        self.probs = torch.from_numpy(np.asarray(probs, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, label, self.probs[index]


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """构建 train / val / test 三个 DataLoader。

    返回 ``(train_loader, val_loader, test_loader, meta)``，
    meta 中包含类别名、样本数与划分索引等信息。
    """
    data_cfg = dict(cfg.get("data", {}))
    root = data_cfg.get("root", ".")
    x_train, y_train, x_test, y_test, classes = load_cifar10(root)

    train_idx, val_idx = stratified_split(
        y_train,
        val_size=int(data_cfg.get("val_size", 5000)),
        seed=int(cfg.get("seed", 42)),
    )
    if data_cfg.get("limit_train"):
        # 快速调试模式：随机抽取一部分训练样本
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        train_idx = np.sort(rng.choice(train_idx, size=min(int(data_cfg["limit_train"]), len(train_idx)), replace=False))
    if data_cfg.get("limit_val"):
        val_idx = val_idx[: int(data_cfg["limit_val"])]
    test_idx = np.arange(len(y_test))
    if data_cfg.get("limit_test"):
        test_idx = test_idx[: int(data_cfg["limit_test"])]

    train_ds = CIFAR10Dataset(x_train, y_train, train_idx, build_transforms(data_cfg, train=True))
    val_ds = CIFAR10Dataset(x_train, y_train, val_idx, build_transforms(data_cfg, train=False))
    test_ds = CIFAR10Dataset(x_test, y_test, test_idx, build_transforms(data_cfg, train=False))

    # 可选：训练标签噪声（只用于补充的早停对照实验，验证集/测试集保持干净）
    noise = float(data_cfg.get("label_noise", 0.0) or 0.0)
    if noise > 0:
        rng = np.random.default_rng(int(cfg.get("seed", 42)) + 1)
        noisy_labels = y_train.copy()
        n_flip = int(round(len(train_idx) * noise))
        flip_idx = rng.choice(train_idx, size=min(n_flip, len(train_idx)), replace=False)
        noisy_labels[flip_idx] = rng.integers(0, len(classes), size=len(flip_idx))
        train_ds.labels = torch.from_numpy(np.ascontiguousarray(noisy_labels[train_idx]).astype(np.int64))

    batch_size = int(cfg.get("train", {}).get("batch_size", data_cfg.get("batch_size", 128)))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    meta = {
        "classes": classes,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "batch_size": batch_size,
        "train_index": train_idx.tolist(),
        "val_index": val_idx.tolist(),
        "test_index": test_idx.tolist(),
        "mean": list(CIFAR10_MEAN),
        "std": list(CIFAR10_STD),
    }
    return train_loader, val_loader, test_loader, meta


# --------------------------------------------------------------------------- #
# Mixup / Cutmix（batch 级增强，在训练循环中调用）
# --------------------------------------------------------------------------- #
class MixupCutmix:
    """Mixup 与 Cutmix 的统一实现。

    返回 ``(mixed_x, y_a, y_b, lam)``；若未触发则返回 ``(x, y, y, 1.0)``。
    """

    def __init__(
        self,
        mode: str = "none",
        alpha: float = 0.2,
        prob: float = 0.5,
        num_classes: int = 10,
    ) -> None:
        self.mode = str(mode).lower()
        self.alpha = float(alpha)
        self.prob = float(prob)
        self.num_classes = num_classes

    @property
    def enabled(self) -> bool:
        return self.mode in ("mixup", "cutmix", "mixup_cutmix")

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if not self.enabled or np.random.rand() > self.prob:
            return x, y, y, 1.0

        mode = self.mode
        if mode == "mixup_cutmix":
            mode = "cutmix" if np.random.rand() < 0.5 else "mixup"

        batch_size = x.size(0)
        perm = torch.randperm(batch_size, device=x.device)
        y_a, y_b = y, y[perm]

        if mode == "mixup":
            lam = float(np.random.beta(self.alpha, self.alpha))
            lam = max(lam, 1.0 - lam)  # 保证主样本占比 >= 0.5，训练更稳定
            x = lam * x + (1.0 - lam) * x[perm]
        else:  # cutmix
            lam = float(np.random.beta(self.alpha, self.alpha))
            _, _, h, w = x.shape
            cut_ratio = np.sqrt(max(0.0, 1.0 - lam))
            cut_h = int(h * cut_ratio)
            cut_w = int(w * cut_ratio)
            cy = np.random.randint(h)
            cx = np.random.randint(w)
            y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)
            x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
        return x, y_a, y_b, float(lam)
