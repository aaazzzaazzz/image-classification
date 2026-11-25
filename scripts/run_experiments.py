# 批量运行全部实验（按依赖顺序，支持断点续跑 / 分组运行）。


from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.evaluate import ensemble_checkpoints  # noqa: E402
from src.experiments import EXPERIMENTS  # noqa: E402
from src.train import run_experiment  # noqa: E402
from src.utils import DEFAULT_OUTPUT_DIR, ensure_dir, format_seconds, get_device, load_json, save_json  # noqa: E402

# 运行顺序：先跑基线，再跑消融，最后训练综合模型与预训练模型
ORDER = [
    "resnet20_basic",
    "simple_cnn",
    "resnet20_aug_none",
    "resnet20_aug_cutout",
    "resnet20_aug_randaugment",
    "resnet20_mixup",
    "resnet20_label_smoothing",
    "resnet20_focal",
    "resnet20_se",
    "resnet20_cbam",
    "resnet20_step",
    "resnet20_onecycle",
    "resnet18_pretrained",
    "mobilenet_v3_pretrained",
    "resnet18_scratch",
    "resnet20_final",
    "se_resnet20_final",
    "resnet18_pretrained_final",
]

GROUPS = {
    "baseline": ["resnet20_basic", "simple_cnn"],
    "models": ["simple_cnn", "resnet20_basic", "resnet18_scratch"],
    "aug": [
        "resnet20_basic",
        "resnet20_aug_none",
        "resnet20_aug_cutout",
        "resnet20_aug_randaugment",
        "resnet20_mixup",
    ],
    "loss": ["resnet20_basic", "resnet20_label_smoothing", "resnet20_focal"],
    "attention": ["resnet20_basic", "resnet20_se", "resnet20_cbam"],
    "scheduler": ["resnet20_basic", "resnet20_step", "resnet20_onecycle"],
    "pretrained": ["resnet18_pretrained", "mobilenet_v3_pretrained", "resnet18_scratch"],
    "final": ["resnet20_final", "se_resnet20_final", "resnet18_pretrained_final"],
    "kd": ["resnet20_kd"],
    "lr": ["resnet20_basic", "resnet20_step", "resnet20_onecycle", "resnet20_constant", "resnet20_constant005"],
    "all": ORDER,
}

ENSEMBLE_MEMBERS = [
    "outputs/checkpoints/resnet20_final_best.pt",
    "outputs/checkpoints/se_resnet20_final_best.pt",
    "outputs/checkpoints/resnet18_pretrained_final_best.pt",
]


def _select(args) -> list[str]:
    if args.only:
        names = args.only
    elif args.group:
        names = GROUPS[args.group]
    else:
        names = ORDER
    if args.exclude:
        names = [n for n in names if n not in set(args.exclude)]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"未知实验: {unknown}")
    return names


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="批量运行 CIFAR-10 实验")
    parser.add_argument("--group", choices=list(GROUPS), default=None)
    parser.add_argument("--only", nargs="+", default=None, help="指定实验名")
    parser.add_argument("--exclude", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="重新训练已完成的实验")
    parser.add_argument("--quick", action="store_true", help="快速冒烟测试（2 epoch、2000 样本）")
    parser.add_argument("--epochs", type=int, default=None, help="统一覆盖 epoch 数（调试用）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ensemble", action="store_true", help="全部完成后对最终模型做概率集成")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for group, names in GROUPS.items():
            print(f"\n[{group}]")
            for name in names:
                print(f"  {name:32s} {EXPERIMENTS[name].get('description', '')}")
        return

    names = _select(args)
    device = torch.device(args.device) if args.device else get_device()
    status_path = ensure_dir(DEFAULT_OUTPUT_DIR) / "run_status.json"
    status = load_json(status_path) if status_path.exists() else {"runs": {}}
    status.setdefault("runs", {})

    print("=" * 100)
    print(f"共 {len(names)} 个实验，设备: {device}")
    print("=" * 100)

    started = time.perf_counter()
    failures = []
    for index, name in enumerate(names, 1):
        cfg = EXPERIMENTS[name]
        if args.epochs is not None:
            cfg = {**cfg, "train": {**cfg["train"], "epochs": args.epochs}}
        eta_text = ""
        if status["runs"]:
            times = [v["elapsed"] for v in status["runs"].values() if v.get("status") == "complete"]
            if times:
                eta_text = f" | 平均 {format_seconds(sum(times) / len(times))}/实验"
        print(f"\n[{index}/{len(names)}] {name} — {cfg.get('description', '')}{eta_text}")
        run_start = time.perf_counter()
        try:
            result = run_experiment(cfg, device=device, quick=args.quick, force=args.force)
            status["runs"][name] = {
                "status": result.get("status", "unknown"),
                "elapsed": time.perf_counter() - run_start,
                "test_acc": result.get("test_acc"),
                "best_val_acc": result.get("best_val_acc"),
            }
        except Exception as exc:  # 单个实验失败不影响后续实验
            print(f"[ERROR] {name} 运行失败: {exc}")
            traceback.print_exc()
            failures.append(name)
            status["runs"][name] = {"status": "failed", "error": str(exc), "elapsed": time.perf_counter() - run_start}
            torch.cuda.empty_cache()
        save_json(status, status_path)

    if args.ensemble:
        paths = [ROOT / p for p in ENSEMBLE_MEMBERS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"[ensemble] 缺少 checkpoint，跳过: {missing}")
        else:
            try:
                ensemble_checkpoints(paths, device=device, tta=False, out_name="ensemble")
                ensemble_checkpoints(paths, device=device, tta=True, out_name="ensemble")
            except Exception as exc:
                print(f"[ensemble] 失败: {exc}")
                traceback.print_exc()

    print("\n" + "=" * 100)
    print(f"全部结束，总用时 {format_seconds(time.perf_counter() - started)}")
    print(f"状态文件: {status_path}")
    if failures:
        print(f"失败实验: {failures}")
        sys.exit(1)
    print("=" * 100)


if __name__ == "__main__":
    main()
