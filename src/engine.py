# 训练与评估核心逻辑：单轮训练、验证、EMA、优化器/学习率调度器。


from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import AverageMeter


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class ModelEMA:
    """指数滑动平均（Exponential Moving Average）权重。

    在训练后期对权重做平均，通常能稳定提升 0.1%~0.4% 的测试精度。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # 训练初期使用较小的 decay，让 EMA 快速跟上
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay).add_(model_v.detach(), alpha=1.0 - decay)
            else:
                ema_v.copy_(model_v)


class _StepLimitedScheduler:
    """包装按 batch step 的调度器，超过 max_steps 后不再 step。

    OneCycleLR 在超过 total_steps 后会抛 ValueError；当最大训练轮数大于
    学习率日程长度时（用于早停实验），需要这个保护。
    """

    def __init__(self, base_scheduler, max_steps: int) -> None:
        self.base_scheduler = base_scheduler
        self.max_steps = int(max_steps)
        self.steps = 0

    def step(self, *args, **kwargs):
        if self.steps < self.max_steps:
            self.base_scheduler.step(*args, **kwargs)
            self.steps += 1

    def state_dict(self):
        return self.base_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.base_scheduler.load_state_dict(state_dict)


# --------------------------------------------------------------------------- #
# 优化器与学习率调度
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "sgd")).lower()
    lr = float(cfg.get("lr", 0.1))
    wd = float(cfg.get("weight_decay", 5e-4))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=wd,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    if name in ("adamw", "adam"):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"未知优化器: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int
) -> Tuple[Optional[torch.optim.lr_scheduler.LRScheduler], bool]:
    """返回 ``(scheduler, step_per_batch)``。"""
    name = str(cfg.get("scheduler", "cosine")).lower()
    epochs = int(cfg.get("epochs", 60))
    # 允许“最大训练轮数”和“学习率日程长度”不同：
    # 例如 max_epochs=200，但 cosine 仍按 120 轮收敛，用于观察早停能否节省训练。
    schedule_epochs = int(cfg.get("scheduler_epochs", epochs))
    warmup = int(cfg.get("warmup_epochs", 0))
    if name in ("none", "constant"):
        return None, False
    if name == "cosine":
        eta_min = float(cfg.get("eta_min", 0.0))

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, schedule_epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), False
    if name == "step":
        milestones = cfg.get("step_milestones", [0.5, 0.75])
        if all(isinstance(m, float) and m <= 1.0 for m in milestones):
            milestones = [int(round(schedule_epochs * m)) for m in milestones]
        return (
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[int(m) for m in milestones], gamma=float(cfg.get("step_gamma", 0.1))
            ),
            False,
        )
    if name == "onecycle":
        pct_start = float(cfg.get("onecycle_pct_start", max(warmup / max(schedule_epochs, 1), 0.1)))
        total_steps = max(1, schedule_epochs * steps_per_epoch)
        base = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("lr", 0.1)),
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=float(cfg.get("onecycle_div", 25.0)),
            final_div_factor=float(cfg.get("onecycle_final_div", 1e4)),
        )
        return _StepLimitedScheduler(base, total_steps), True
    raise ValueError(f"未知学习率调度器: {name}")


# --------------------------------------------------------------------------- #
# 训练 / 评估
# --------------------------------------------------------------------------- #
def _accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
            res.append(correct_k * 100.0 / max(batch_size, 1))
        return res


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: Optional[torch.amp.GradScaler] = None,
    mixup=None,
    ema: Optional[ModelEMA] = None,
    grad_clip: float = 0.0,
    scheduler=None,
    step_per_batch: bool = False,
    log_interval: int = 100,
) -> Dict[str, float]:
    """训练一个 epoch，返回平均 loss / acc / lr。"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    use_amp = scaler is not None and scaler.is_enabled()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lam, y_a, y_b = 1.0, targets, targets
        if mixup is not None and mixup.enabled:
            images, y_a, y_b, lam = mixup(images, targets)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if lam < 1.0:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y_a)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if step_per_batch and scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_a = _accuracy(logits.detach(), y_a, topk=(1,))[0]
        if lam < 1.0:
            acc_b = _accuracy(logits.detach(), y_b, topk=(1,))[0]
            acc = lam * acc_a + (1.0 - lam) * acc_b
        else:
            acc = acc_a
        acc_meter.update(acc, batch_size)

        if log_interval and (step + 1) % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"    step {step + 1:4d}/{len(loader)} | loss {loss_meter.val:.4f} | "
                f"acc {acc_meter.val:5.2f}% | lr {lr_now:.5f}",
                flush=True,
            )

    return {
        "loss": loss_meter.sum / max(loss_meter.count, 1),
        "acc": acc_meter.sum / max(acc_meter.count, 1),
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    *,
    collect: bool = False,
    tta: bool = False,
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """评估模型，返回 ``(metrics, probs, labels)``。

    ``tta=True`` 时对原图与水平翻转图取平均概率（Test-Time Augmentation）。
    """
    model.eval()
    loss_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_probs, all_labels = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            # 原图与水平翻转图的预测概率取平均
            probs = 0.5 * (
                F.softmax(logits.float(), dim=1)
                + F.softmax(model(torch.flip(images, dims=[3])).float(), dim=1)
            )
            scores = probs
        else:
            scores = logits
            if criterion is not None:
                loss = criterion(logits, targets)
                loss_meter.update(loss.item(), targets.size(0))
        acc1, acc5 = _accuracy(scores, targets, topk=(1, 5))
        top1.update(acc1, targets.size(0))
        top5.update(acc5, targets.size(0))
        if collect:
            all_probs.append((scores.float() if tta else F.softmax(logits.float(), dim=1)).cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    metrics = {"loss": loss_meter.sum / max(loss_meter.count, 1), "acc": top1.sum / max(top1.count, 1), "acc_top5": top5.sum / max(top5.count, 1)}
    if collect:
        return metrics, np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)
    return metrics, None, None


def save_checkpoint(
    path,
    model: nn.Module,
    cfg: dict,
    epoch: int,
    best_acc: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[ModelEMA] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "cfg": cfg,
        "epoch": int(epoch),
        "best_acc": float(best_acc),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.ema.state_dict()
    torch.save(payload, path)


def load_model_state(model: nn.Module, checkpoint_path, device: torch.device) -> dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    return payload
