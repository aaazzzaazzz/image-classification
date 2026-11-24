# 预计算教师模型在训练集上的 softmax 概率，供离线知识蒸馏使用。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, worker_init_fn  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import ensure_dir, get_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="outputs/checkpoints/resnet18_pretrained_best.pt")
    parser.add_argument("--out", default="outputs/teacher_probs/resnet18_train_t4.npy")
    parser.add_argument("--temp", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = {
        "seed": 42,
        "data": {"root": ".", "val_size": 5000, "aug": "none", "batch_size": args.batch_size, "num_workers": args.num_workers},
        "train": {},
    }
    train_loader, _, _, meta = build_dataloaders(cfg)
    dataset = train_loader.dataset
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    device = get_device()
    model = build_model({"name": "resnet18_cifar", "num_classes": 10, "pretrained": False}).to(device)
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    chunks = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            probs = torch.softmax(logits / args.temp, dim=1)
            chunks.append(probs.float().cpu().numpy())
    probs = np.concatenate(chunks, axis=0)
    if len(probs) != meta["num_train"]:
        raise RuntimeError(f"概率数量 {len(probs)} 与训练集 {meta['num_train']} 不一致")

    out = Path(args.out)
    ensure_dir(out.parent)
    np.save(out, probs)
    print(f"保存 {out} | shape={probs.shape} | dtype={probs.dtype}")


if __name__ == "__main__":
    main()
