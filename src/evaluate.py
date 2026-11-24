# 模型评估：单权重评估、TTA、多模型概率集成。


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import build_dataloaders
from .models import build_model
from .metrics import summarize
from .utils import DEFAULT_OUTPUT_DIR, ensure_dir, get_device, save_json
from .visualize import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_per_class_metrics,
    plot_top_confusions,
)


def _load_cfg_and_model(checkpoint_path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload.get("cfg", {})
    model = build_model(cfg.get("model", {})).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg, payload


def _test_loader(cfg: dict):
    eval_cfg = {
        **cfg,
        "data": {**cfg.get("data", {}), "num_workers": 0, "limit_train": 0, "limit_val": 1, "limit_test": 0},
        "train": {**cfg.get("train", {}), "batch_size": 256},
    }
    _, _, test_loader, meta = build_dataloaders(eval_cfg)
    return test_loader, meta["classes"]


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader, device: torch.device, tta: bool = False):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        if tta:
            probs = 0.5 * (probs + torch.softmax(model(torch.flip(images, dims=[3])).float(), dim=1))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def _save_evaluation(
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    out_dir: Path,
    extra: dict,
    make_figures: bool = True,
) -> dict:
    summary = summarize(probs, labels, classes)
    result = {
        "name": name,
        "status": "complete",
        "test_acc": summary["acc"],
        "test_acc_top5": summary["acc_top5"],
        "test_macro_f1": summary["macro_f1"],
        "test_weighted_f1": summary["weighted_f1"],
        "per_class_acc": summary["per_class_acc"],
        "per_class_report": summary["per_class_report"],
        "confusion_matrix": summary["confusion_matrix"].tolist(),
        "top_confusions": summary["top_confusions"],
        "num_test": summary["num_samples"],
        "class_names": list(classes),
        "history": [],
        **extra,
    }
    out_dir = ensure_dir(out_dir)
    save_json(result, out_dir / "results.json")
    np.save(out_dir / "test_probs.npy", probs.astype(np.float32))
    np.save(out_dir / "test_labels.npy", labels.astype(np.int64))
    if make_figures:
        fig_dir = ensure_dir(DEFAULT_OUTPUT_DIR / "figures")
        plot_confusion_matrix(
            summary["confusion_matrix"],
            classes,
            fig_dir / f"{name}_confusion.png",
            title=f"{name} (test acc {summary['acc']:.2f}%)",
        )
        plot_per_class_metrics(
            summary["per_class_report"], classes, fig_dir / f"{name}_per_class_metrics.png", title=f"{name}: per-class metrics"
        )
        plot_per_class_accuracy(
            summary["per_class_acc"], classes, fig_dir / f"{name}_per_class_acc.png", overall=summary["acc"]
        )
        plot_top_confusions(summary["confusion_matrix"], classes, fig_dir / f"{name}_top_confusions.png")
    return result


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    tta: bool = False,
    out_name: str | None = None,
    make_figures: bool = True,
) -> dict:
    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)
    model, cfg, payload = _load_cfg_and_model(checkpoint_path, device)
    test_loader, classes = _test_loader(cfg)
    probs, labels = predict_probs(model, test_loader, device, tta=tta)
    name = out_name or checkpoint_path.stem.replace("_best", "")
    out_dir = DEFAULT_OUTPUT_DIR / "evaluation" / (f"{name}_tta" if tta else name)
    extra = {
        "source_checkpoint": str(checkpoint_path),
        "tta": bool(tta),
        "best_val_acc": payload.get("best_acc"),
        "best_epoch": payload.get("epoch"),
        "used_ema": payload.get("used_ema", False),
        "num_params": sum(p.numel() for p in model.parameters()),
        "model": cfg.get("model", {}),
    }
    result = _save_evaluation(name + ("_tta" if tta else ""), probs, labels, classes, out_dir, extra, make_figures)
    print(
        f"[eval] {name} | TTA={tta} | test top1 {result['test_acc']:.2f}% | "
        f"top5 {result['test_acc_top5']:.2f}% | macro-F1 {result['test_macro_f1']:.2f}%"
    )
    return result


def ensemble_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    device: torch.device | None = None,
    tta: bool = False,
    out_name: str = "ensemble",
    weights: Sequence[float] | None = None,
    make_figures: bool = True,
) -> dict:
    """对多个模型输出的 softmax 概率做加权平均（权重默认均等）。"""
    device = device or get_device()
    paths = [Path(p) for p in checkpoint_paths]
    if not paths:
        raise ValueError("至少需要一个 checkpoint")
    weights_arr = np.asarray(weights if weights is not None else [1.0] * len(paths), dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()

    probs_sum = None
    labels_ref = None
    classes = None
    members = []
    for path, weight in zip(paths, weights_arr):
        model, cfg, payload = _load_cfg_and_model(path, device)
        if classes is None:
            test_loader, classes = _test_loader(cfg)
        probs, labels = predict_probs(model, test_loader, device, tta=tta)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError("不同模型的测试集标签不一致，请确认使用同一数据划分")
        probs_sum = probs * weight if probs_sum is None else probs_sum + probs * weight
        members.append(
            {
                "checkpoint": str(path),
                "weight": float(weight),
                "test_acc": float((probs.argmax(1) == labels).mean() * 100.0),
                "best_val_acc": payload.get("best_acc"),
                "model": cfg.get("model", {}),
                "num_params": int(sum(p.numel() for p in model.parameters())),
            }
        )
        del model
        torch.cuda.empty_cache()

    assert probs_sum is not None and labels_ref is not None and classes is not None
    ensemble_name = out_name + ("_tta" if tta else "")
    out_dir = DEFAULT_OUTPUT_DIR / "evaluation" / ensemble_name
    extra = {
        "members": members,
        "tta": bool(tta),
        "ensemble_weights": [float(w) for w in weights_arr],
        "num_params": int(sum(m.get("num_params", 0) or 0 for m in members)),
        "num_test": int(len(labels_ref)),
    }
    result = _save_evaluation(ensemble_name, probs_sum, labels_ref, classes, out_dir, extra, make_figures)
    print(
        f"[ensemble] {len(paths)} 个模型 | TTA={tta} | test top1 {result['test_acc']:.2f}% | "
        f"top5 {result['test_acc_top5']:.2f}% | macro-F1 {result['test_macro_f1']:.2f}%"
    )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="评估 CIFAR-10 模型")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="单个 checkpoint 路径")
    group.add_argument("--ensemble", nargs="+", help="多个 checkpoint 路径（概率集成）")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device) if args.device else get_device()

    if args.checkpoint:
        evaluate_checkpoint(args.checkpoint, device=device, tta=args.tta, out_name=args.name, make_figures=not args.no_figures)
    else:
        ensemble_checkpoints(
            args.ensemble,
            device=device,
            tta=args.tta,
            out_name=args.name or "ensemble",
            weights=args.weights,
            make_figures=not args.no_figures,
        )


if __name__ == "__main__":
    main()
