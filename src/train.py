# 单个实验的训练入口：训练、验证、测试、结果与图片落盘。


from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MixupCutmix, SoftLabelDataset, build_dataloaders
from .engine import (
    ModelEMA,
    build_optimizer,
    build_scheduler,
    evaluate,
    save_checkpoint,
    train_one_epoch,
)
from .losses import build_criterion
from .metrics import summarize
from .models import build_model
from .utils import (
    CIFAR10_CLASSES,
    DEFAULT_OUTPUT_DIR,
    count_parameters,
    deep_update,
    ensure_dir,
    format_seconds,
    get_device,
    load_json,
    save_json,
    set_seed,
    worker_init_fn,
)
from .visualize import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_per_class_metrics,
    plot_training_curves,
)


class Tee:
    """把 stdout 同时写到控制台和日志文件。"""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _fmt_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    text = f"{prefix} loss {metrics['loss']:.4f} | top1 {metrics['acc']:.2f}%"
    if "acc_top5" in metrics:
        text += f" | top5 {metrics['acc_top5']:.2f}%"
    return text


def _state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def run_experiment(
    cfg: dict,
    *,
    device: Optional[torch.device] = None,
    quick: bool = False,
    force: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """运行一个完整实验并返回结果字典。"""
    cfg = copy.deepcopy(cfg)
    if "name" not in cfg:
        raise ValueError("配置中必须包含 name 字段")
    name = str(cfg["name"])
    if quick:
        cfg.setdefault("data", {})
        cfg["data"]["limit_train"] = int(cfg["data"].get("limit_train") or 2000)
        cfg["data"]["limit_val"] = int(cfg["data"].get("limit_val") or 1000)
        cfg["data"]["limit_test"] = int(cfg["data"].get("limit_test") or 1000)
        cfg["data"]["num_workers"] = 0
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = min(int(cfg["train"].get("epochs", 60)), 2)
        cfg["train"]["early_stop_patience"] = 100
        cfg["train"]["log_interval"] = 10

    output_dir = Path(output_dir)
    result_dir = ensure_dir(output_dir / "results" / name)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    fig_dir = ensure_dir(output_dir / "figures")
    log_dir = ensure_dir(output_dir / "logs")
    result_path = result_dir / "results.json"

    if result_path.exists() and not force:
        existing = load_json(result_path)
        if existing.get("status") == "complete":
            print(f"[skip] {name} 已完成（{result_path}），使用 --force 重新训练。")
            return existing

    save_json(cfg, result_dir / "config.json")

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg["train"].get("deterministic", False)))
    device = device or get_device()
    train_cfg = cfg["train"]
    data_cfg = cfg.get("data", {})
    epochs = int(train_cfg.get("epochs", 60))

    log_path = log_dir / f"{name}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, log_file)
    start_time = time.perf_counter()
    try:
        print("=" * 100)
        print(f"实验: {name}")
        print(f"设备: {device} | 随机种子: {cfg.get('seed', 42)} | epochs: {epochs}")
        print(f"模型配置: {json.dumps(cfg.get('model', {}), ensure_ascii=False)}")
        print(f"数据配置: {json.dumps(data_cfg, ensure_ascii=False)}")
        print(f"训练配置: {json.dumps(train_cfg, ensure_ascii=False)}")

        train_loader, val_loader, test_loader, meta = build_dataloaders(cfg)
        print(
            f"数据: train={meta['num_train']} / val={meta['num_val']} / test={meta['num_test']} | "
            f"batch_size={meta['batch_size']} | 类别={meta['classes']}"
        )

        # 离线知识蒸馏：用教师概率替换训练集标签，避免每个 batch 额外前向教师
        if train_cfg.get("distill_probs"):
            probs = np.load(str(train_cfg["distill_probs"]), allow_pickle=False)
            train_loader = DataLoader(
                SoftLabelDataset(train_loader.dataset, probs),
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
                drop_last=True,
                persistent_workers=train_loader.num_workers > 0,
                worker_init_fn=worker_init_fn,
            )
            print(f"离线 KD: {train_cfg['distill_probs']} | alpha={train_cfg.get('distill_alpha', 0.7)} | T={train_cfg.get('distill_temp', 4.0)}")

        model = build_model(cfg.get("model", {})).to(device)
        num_params = count_parameters(model)
        print(f"模型: {cfg.get('model', {}).get('name')} | 参数量: {num_params / 1e6:.3f}M")

        # 知识蒸馏教师：只前向、不更新；用于 KD 损失
        teacher = None
        teacher_path = train_cfg.get("distill_teacher")
        if teacher_path:
            teacher_cfg = {
                "name": str(train_cfg.get("distill_teacher_model", "resnet18_cifar")),
                "pretrained": False,
            }
            teacher = build_model(teacher_cfg).to(device)
            teacher_ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
            state = teacher_ckpt.get("model", teacher_ckpt)
            teacher.load_state_dict(state)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            print(f"KD 教师: {teacher_path} | alpha={train_cfg.get('distill_alpha', 0.3)} | T={train_cfg.get('distill_temp', 4.0)}")

        criterion = build_criterion(train_cfg).to(device)
        optimizer = build_optimizer(model, train_cfg)
        steps_per_epoch = max(1, len(train_loader))
        scheduler, step_per_batch = build_scheduler(optimizer, train_cfg, steps_per_epoch)
        ema = ModelEMA(model, decay=float(train_cfg["ema_decay"])) if float(train_cfg.get("ema_decay", 0)) > 0 else None
        mixup = MixupCutmix(
            mode=str(train_cfg.get("mixup", "none")),
            alpha=float(train_cfg.get("mixup_alpha", 0.2)),
            prob=float(train_cfg.get("mixup_prob", 0.5)),
        )

        use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        print(f"AMP: {use_amp} | mixup: {mixup.mode} | EMA decay: {train_cfg.get('ema_decay', 0)}")
        print("=" * 100)

        history = []
        best_acc, best_epoch = 0.0, 0
        best_state = None
        best_ema_acc, best_ema_epoch, best_ema_state = 0.0, 0, None
        patience = int(train_cfg.get("early_stop_patience", 0) or 0)
        min_delta = float(train_cfg.get("early_stop_min_delta", 0.0) or 0.0)
        bad_epochs = 0
        stop_reason = "max_epochs"

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                mixup=mixup if mixup.enabled else None,
                ema=ema,
                grad_clip=float(train_cfg.get("grad_clip", 0.0) or 0.0),
                scheduler=scheduler,
                step_per_batch=step_per_batch,
                log_interval=int(train_cfg.get("log_interval", 100)),
                teacher=teacher,
                distill_alpha=float(train_cfg.get("distill_alpha", 0.3)),
                distill_temp=float(train_cfg.get("distill_temp", 4.0)),
            )
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            if not step_per_batch and scheduler is not None:
                scheduler.step()

            ema_acc = None
            if ema is not None:
                ema_metrics, _, _ = evaluate(ema.ema, val_loader, criterion, device)
                ema_acc = ema_metrics["acc"]
                if ema_acc > best_ema_acc + 1e-6:
                    best_ema_acc, best_ema_epoch = ema_acc, epoch
                    best_ema_state = _state_to_cpu(ema.ema.state_dict())

            if val_metrics["acc"] > best_acc + 1e-6:
                best_acc, best_epoch = val_metrics["acc"], epoch
                best_state = _state_to_cpu(model.state_dict())
                save_checkpoint(
                    ckpt_dir / f"{name}_best.pt",
                    model,
                    cfg,
                    epoch,
                    best_acc,
                    optimizer=optimizer,
                    ema=ema,
                )

            monitor_mode = str(train_cfg.get("early_stop_monitor", "val_ema")).lower()
            if monitor_mode == "val":
                monitor = val_metrics["acc"]
            else:
                monitor = max(val_metrics["acc"], ema_acc or 0.0)
            record = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_acc": float(train_metrics["acc"]),
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["acc"]),
                "val_acc_top5": float(val_metrics["acc_top5"]),
                "ema_val_acc": float(ema_acc) if ema_acc is not None else None,
                "lr": float(train_metrics["lr"]),
                "time_sec": time.perf_counter() - epoch_start,
                "elapsed_sec": time.perf_counter() - start_time,
            }
            history.append(record)

            if monitor_mode == "val":
                previous_best = max([h["val_acc"] for h in history[:-1]] + [0.0])
            else:
                previous_best = max(
                    [h["val_acc"] for h in history[:-1]]
                    + [h.get("ema_val_acc") or 0.0 for h in history[:-1]]
                    + [0.0]
                )
            improved = monitor > previous_best + min_delta
            if improved:
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch:3d}/{epochs} | {_fmt_metrics('train', train_metrics)} | "
                f"{_fmt_metrics('val', val_metrics)}"
                + (f" | EMA {ema_acc:.2f}%" if ema_acc is not None else "")
                + f" | lr {train_metrics['lr']:.5f} | {record['time_sec']:.1f}s"
            )

            if patience > 0 and bad_epochs >= patience:
                stop_reason = f"early_stop(no improvement for {patience} epochs)"
                print(f"[early stop] 验证集指标连续 {patience} 个 epoch 未提升，提前结束训练。")
                break

        if best_state is None:  # 理论上不会发生
            best_state = _state_to_cpu(model.state_dict())

        # 记录“训练结束时的最后一轮权重”在验证/测试集上的表现，
        # 用于对比“固定轮数取最后一轮”与“早停取最佳权重”的差异。
        last_val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        last_test_metrics, _, _ = evaluate(model, test_loader, criterion, device)

        used_ema = best_ema_state is not None and best_ema_acc > best_acc
        chosen_state = best_ema_state if used_ema else best_state
        chosen_val_acc = best_ema_acc if used_ema else best_acc
        chosen_epoch = best_ema_epoch if used_ema else best_epoch

        model.load_state_dict(chosen_state)
        model.to(device)
        val_final, _, _ = evaluate(model, val_loader, criterion, device)
        test_metrics, test_probs, test_labels = evaluate(model, test_loader, criterion, device, collect=True)

        test_tta_acc = None
        if bool(train_cfg.get("test_tta", False)):
            tta_metrics, _, _ = evaluate(model, test_loader, criterion, device, tta=True)
            test_tta_acc = float(tta_metrics["acc"])
            print(f"TTA 测试准确率: {test_tta_acc:.2f}%")

        summary = summarize(test_probs, test_labels, meta["classes"])
        cm = summary["confusion_matrix"]
        report = summary["per_class_report"]
        per_class_acc = summary["per_class_acc"]
        top_confusions = summary["top_confusions"]

        elapsed = time.perf_counter() - start_time
        results = {
            "name": name,
            "status": "complete",
            "model": cfg.get("model", {}),
            "num_params": num_params,
            "best_epoch": int(chosen_epoch),
            "best_val_acc": float(chosen_val_acc),
            "last_epoch": int(history[-1]["epoch"]) if history else 0,
            "last_val_acc": float(last_val_metrics["acc"]),
            "last_test_acc": float(last_test_metrics["acc"]),
            "used_ema": bool(used_ema),
            "test_acc": float(test_metrics["acc"]),
            "test_acc_top5": float(test_metrics["acc_top5"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc_tta": test_tta_acc,
            "test_macro_f1": summary["macro_f1"],
            "per_class_acc": per_class_acc,
            "per_class_report": report,
            "confusion_matrix": cm.tolist(),
            "top_confusions": top_confusions,
            "history": history,
            "num_train": meta["num_train"],
            "num_val": meta["num_val"],
            "num_test": meta["num_test"],
            "class_names": meta["classes"],
            "elapsed_sec": elapsed,
            "stop_reason": stop_reason,
            "config": cfg,
        }

        final_ckpt = {
            "model": chosen_state,
            "cfg": cfg,
            "epoch": int(chosen_epoch),
            "best_acc": float(chosen_val_acc),
            "used_ema": bool(used_ema),
        }
        torch.save(final_ckpt, ckpt_dir / f"{name}_best.pt")
        save_json(cfg, result_dir / "config.json")
        save_json(history, result_dir / "history.json")
        save_json(results, result_path)
        np.save(result_dir / "test_probs.npy", test_probs.astype(np.float32))
        np.save(result_dir / "test_labels.npy", test_labels.astype(np.int64))

        plot_training_curves(history, fig_dir / f"{name}_curves.png", title=name)
        plot_confusion_matrix(cm, meta["classes"], fig_dir / f"{name}_confusion.png", title=f"{name} (test acc {test_metrics['acc']:.2f}%)")
        plot_per_class_metrics(report, meta["classes"], fig_dir / f"{name}_per_class_metrics.png", title=f"{name}: per-class metrics")
        plot_per_class_accuracy(per_class_acc, meta["classes"], fig_dir / f"{name}_per_class_acc.png", overall=test_metrics["acc"])

        print("=" * 100)
        print(
            f"[完成] {name} | best_val {chosen_val_acc:.2f}% (epoch {chosen_epoch}, "
            f"{'EMA' if used_ema else 'raw'}) | test top1 {test_metrics['acc']:.2f}% | "
            f"top5 {test_metrics['acc_top5']:.2f}% | 用时 {format_seconds(elapsed)}"
        )
        print(f"结果: {result_path}")
        print(f"权重: {ckpt_dir / f'{name}_best.pt'}")
        print("=" * 100)
        return results
    finally:
        sys.stdout = old_stdout
        log_file.close()


def _apply_cli_overrides(cfg: dict, args) -> dict:
    cfg = copy.deepcopy(cfg)
    if args.name:
        cfg["name"] = args.name
    cfg.setdefault("train", {})
    cfg.setdefault("data", {})
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    return cfg


def main(argv=None) -> None:
    from .experiments import EXPERIMENTS

    parser = argparse.ArgumentParser(description="CIFAR-10 图像分类训练")
    parser.add_argument("--exp", type=str, default=None, help="experiments.py 中的实验名")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--quick", action="store_true", help="1~2 个 epoch 的快速冒烟测试")
    parser.add_argument("--force", action="store_true", help="即使已有结果也重新训练")
    parser.add_argument("--list", action="store_true", help="列出所有预置实验")
    args = parser.parse_args(argv)

    if args.list:
        for key, cfg in EXPERIMENTS.items():
            print(f"{key:32s} {cfg.get('description', '')}")
        return

    if args.exp is None and args.config is None:
        parser.error("需要指定 --exp 或 --config（或用 --list 查看预置实验）")

    if args.exp is not None:
        if args.exp not in EXPERIMENTS:
            parser.error(f"未知实验 {args.exp}，可用: {', '.join(EXPERIMENTS)}")
        cfg = copy.deepcopy(EXPERIMENTS[args.exp])
    else:
        cfg = load_json(args.config)

    cfg = _apply_cli_overrides(cfg, args)
    device = torch.device(args.device) if args.device else get_device()
    run_experiment(cfg, device=device, quick=args.quick, force=args.force)


if __name__ == "__main__":
    main()
