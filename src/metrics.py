# 统一的评估指标计算：Top-1/Top-5、每类指标、混淆矩阵、Top 混淆对。


from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk == labels[:, None]).any(axis=1)
    return float(hits.mean() * 100.0)


def top_confusions(cm: np.ndarray, classes: Sequence[str], k: int = 10) -> list[dict]:
    pairs = []
    for i in range(cm.shape[0]):
        total = max(int(cm[i].sum()), 1)
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": classes[i],
                        "pred": classes[j],
                        "count": int(cm[i, j]),
                        "ratio_of_true_class": float(cm[i, j] / total),
                    }
                )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:k]


def summarize(probs: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> dict:
    """根据预测概率计算完整指标集合。"""
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(np.int64)
    preds = probs.argmax(axis=1)
    num_classes = len(classes)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    report = classification_report(
        labels,
        preds,
        labels=list(range(num_classes)),
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    per_class_acc = [float(report[c]["recall"] * 100.0) for c in classes]
    return {
        "acc": float((preds == labels).mean() * 100.0),
        "acc_top5": topk_accuracy(probs, labels, k=5),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0) * 100.0),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0) * 100.0),
        "per_class_acc": per_class_acc,
        "per_class_report": {c: report[c] for c in classes},
        "confusion_matrix": cm,
        "top_confusions": top_confusions(cm, classes, k=10),
        "num_samples": int(len(labels)),
    }
