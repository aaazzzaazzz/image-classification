# 基于验证集的贪心集成（Greedy Ensemble Selection）。


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.evaluate import _save_evaluation, predict_probs  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import DEFAULT_OUTPUT_DIR, get_device, load_json, save_json  # noqa: E402
from src.visualize import configure_matplotlib  # noqa: E402

configure_matplotlib()

DEFAULT_CHECKPOINTS = [
    "outputs/checkpoints/resnet18_pretrained_final_best.pt",
    "outputs/checkpoints/resnet18_pretrained_best.pt",
    "outputs/checkpoints/se_resnet20_final_best.pt",
    "outputs/checkpoints/resnet20_final_best.pt",
    "outputs/checkpoints/resnet18_scratch_best.pt",
    "outputs/checkpoints/simple_cnn_best.pt",
]


def _load_model(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("cfg", {})
    model = build_model(cfg.get("model", {})).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg, payload


def _loaders(cfg: dict):
    eval_cfg = {
        **cfg,
        "data": {**cfg.get("data", {}), "num_workers": 0, "limit_train": 0, "limit_val": 0, "limit_test": 0},
        "train": {**cfg.get("train", {}), "batch_size": 256},
    }
    _, val_loader, test_loader, meta = build_dataloaders(eval_cfg)
    return val_loader, test_loader, meta["classes"]


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def greedy_selection(val_probs: list[np.ndarray], labels: np.ndarray, max_iter: int = 20):
    """返回归一化权重与非负计数。"""
    n_models = len(val_probs)
    counts = np.zeros(n_models, dtype=np.float64)
    current = np.zeros_like(val_probs[0])
    total = 0
    history = []
    best_acc, best_nll = -1.0, 1e9
    for step in range(max_iter):
        candidate_best = None
        for i in range(n_models):
            cand = (current * total + val_probs[i]) / (total + 1)
            acc = float((cand.argmax(1) == labels).mean() * 100.0)
            nll = _nll(cand, labels)
            key = (acc, -nll)
            if candidate_best is None or key > candidate_best[0]:
                candidate_best = (key, i, acc, nll, cand)
        _, idx, acc, nll, cand = candidate_best
        if acc < best_acc - 1e-9:
            break
        if acc > best_acc + 1e-9 or nll < best_nll - 1e-9:
            best_acc, best_nll = max(best_acc, acc), min(best_nll, nll)
            current = cand
            counts[idx] += 1
            total += 1
            history.append({"step": step + 1, "model_index": int(idx), "val_acc": acc, "val_nll": nll})
        else:
            break
    weights = counts / counts.sum()
    return weights, history, best_acc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="基于验证集的贪心模型集成")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--out-name", type=str, default="ensemble_greedy")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    device = torch.device(args.device) if args.device else get_device()

    paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.checkpoints]
    paths = [p for p in paths if p.exists()]
    if len(paths) < 2:
        raise SystemExit(f"可用 checkpoint 不足: {paths}")

    val_probs, test_probs, test_probs_tta = [], [], []
    labels_val = labels_test = None
    classes = None
    members = []
    for path in paths:
        model, cfg, payload = _load_model(path, device)
        val_loader, test_loader, class_names = _loaders(cfg)
        if classes is None:
            classes = class_names
        vp, vy = predict_probs(model, val_loader, device)
        tp, ty = predict_probs(model, test_loader, device)
        tp_tta, _ = predict_probs(model, test_loader, device, tta=True)
        if labels_val is None:
            labels_val, labels_test = vy, ty
        elif not (np.array_equal(labels_val, vy) and np.array_equal(labels_test, ty)):
            raise ValueError("验证/测试集顺序不一致")
        val_probs.append(vp)
        test_probs.append(tp)
        test_probs_tta.append(tp_tta)
        acc = float((tp.argmax(1) == ty).mean() * 100.0)
        members.append(
            {
                "checkpoint": str(path),
                "test_acc": acc,
                "test_acc_tta": float((tp_tta.argmax(1) == ty).mean() * 100.0),
                "best_val_acc": payload.get("best_acc"),
                "model": cfg.get("model", {}),
                "num_params": int(sum(p.numel() for p in model.parameters())),
            }
        )
        print(f"  {path.name:45s} val {float((vp.argmax(1) == vy).mean() * 100):6.2f}  test {acc:6.2f}")
        del model
        torch.cuda.empty_cache()

    weights, history, best_val = greedy_selection(val_probs, labels_val, max_iter=args.max_iter)
    active = [(i, w) for i, w in enumerate(weights) if w > 0]
    print("\n贪心选择结果（验证集）：")
    for i, w in active:
        print(f"  {paths[i].name:45s} weight {w:.3f}")
    print(f"  验证集集成准确率: {best_val:.2f}%")

    def blend(probs_list):
        out = np.zeros_like(probs_list[0])
        for i, w in active:
            out += w * probs_list[i]
        return out

    extra = {
        "method": "greedy ensemble selection on validation set",
        "members": [
            {**members[i], "weight": float(w)} for i, w in active
        ],
        "candidate_members": members,
        "greedy_history": history,
        "selection_val_acc": best_val,
        "tta": False,
    }
    result = _save_evaluation(args.out_name, blend(test_probs), labels_test, classes, DEFAULT_OUTPUT_DIR / "evaluation" / args.out_name, extra)
    print(f"[greedy] 测试 Top-1: {result['test_acc']:.2f}%  Top-5: {result['test_acc_top5']:.2f}%")

    extra_tta = {**extra, "tta": True}
    result_tta = _save_evaluation(
        args.out_name + "_tta", blend(test_probs_tta), labels_test, classes, DEFAULT_OUTPUT_DIR / "evaluation" / (args.out_name + "_tta"), extra_tta
    )
    print(f"[greedy+TTA] 测试 Top-1: {result_tta['test_acc']:.2f}%  Top-5: {result_tta['test_acc_top5']:.2f}%")

    save_json(
        {
            "weights": {str(paths[i].name): float(w) for i, w in active},
            "val_acc": best_val,
            "test_acc": result["test_acc"],
            "test_acc_tta": result_tta["test_acc"],
            "candidates": members,
        },
        DEFAULT_OUTPUT_DIR / "evaluation" / args.out_name / "search_summary.json",
    )


if __name__ == "__main__":
    main()
