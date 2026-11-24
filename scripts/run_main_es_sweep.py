# 让所有主实验都触发早停的批量实验（方案 A）。


from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.experiments import EXPERIMENTS  # noqa: E402
from src.train import run_experiment  # noqa: E402
from src.utils import DEFAULT_OUTPUT_DIR, format_seconds, get_device, load_json, save_json  # noqa: E402

MAIN = [
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="主实验早停触发批量实验（最大预算=原预算+patience+5）")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    names = args.only or MAIN
    device = torch.device(args.device) if args.device else get_device()
    status_path = DEFAULT_OUTPUT_DIR / "run_main_es_status.json"
    status = load_json(status_path) if status_path.exists() else {"runs": {}}
    status.setdefault("runs", {})

    print(f"共 {len(names)} 个主实验，最大预算 = 原预算 + patience + 5；日程长度保持原预算")
    t0 = time.perf_counter()
    failures = []
    for idx, name in enumerate(names, 1):
        cfg = copy.deepcopy(EXPERIMENTS[name])
        orig_epochs = int(cfg["train"].get("epochs", 60))
        patience = int(cfg["train"].get("early_stop_patience", 20) or 20)
        new_name = f"{name}_es_main"
        cfg["name"] = new_name
        # 给 BN 统计漂移留足空间：最大预算 = 2×原预算 + patience
        cfg["train"]["epochs"] = orig_epochs * 2 + patience
        cfg["train"]["scheduler_epochs"] = orig_epochs
        # 保持原 patience 与监控方式；这样最佳 epoch 后仍有窗口可触发
        print(f"\n[{idx}/{len(names)}] {name} -> {new_name} | 原 {orig_epochs} 轮，最大 {cfg['train']['epochs']} 轮，patience {patience}")
        start = time.perf_counter()
        try:
            result = run_experiment(cfg, device=device, force=args.force)
            status["runs"][new_name] = {
                "status": result.get("status"),
                "stop_reason": result.get("stop_reason"),
                "epochs_run": len(result.get("history", [])),
                "best_epoch": result.get("best_epoch"),
                "test_acc": result.get("test_acc"),
                "orig_test_acc": None,
                "elapsed": time.perf_counter() - start,
            }
            orig_path = DEFAULT_OUTPUT_DIR / "results" / name / "results.json"
            if orig_path.exists():
                status["runs"][new_name]["orig_test_acc"] = load_json(orig_path).get("test_acc")
            print(
                f"[完成] {new_name}: {result.get('stop_reason')} | 实际 {len(result.get('history', []))} 轮 | "
                f"best {result.get('best_epoch')} | test {result.get('test_acc'):.2f}%"
            )
        except Exception as exc:
            print(f"[ERROR] {new_name}: {exc}")
            traceback.print_exc()
            failures.append(new_name)
            status["runs"][new_name] = {"status": "failed", "error": str(exc)}
        save_json(status, status_path)
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print(f"全部结束，总用时 {format_seconds(time.perf_counter() - t0)}")
    if failures:
        print("失败:", failures)
    print(f"状态文件: {status_path}")


if __name__ == "__main__":
    main()
