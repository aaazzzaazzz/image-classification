# Grad-CAM 可解释性分析与特征图可视化。


from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .utils import configure_matplotlib, ensure_dir

configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402


class GradCAM:
    """对指定卷积层做 Grad-CAM。

    用法::

        cam = GradCAM(model, target_layer)
        heatmaps = cam.generate(images, class_indices)  # (N, H, W)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

    def _register(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output
            output.register_hook(self._save_gradient)

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.activations = None
        self.gradients = None

    def generate(self, images: torch.Tensor, class_indices: torch.Tensor | None = None) -> np.ndarray:
        """返回 ``(N, H, W)``、归一化到 [0,1] 的 CAM。"""
        was_training = self.model.training
        self.model.eval()
        self._register()
        try:
            images = images.clone().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(images)
                if class_indices is None:
                    class_indices = logits.argmax(dim=1)
                class_indices = class_indices.to(logits.device).long()
                score = logits.gather(1, class_indices.view(-1, 1)).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()
            if self.activations is None or self.gradients is None:
                raise RuntimeError("未捕获到目标层激活/梯度，请检查 target_layer 是否为模型子模块")
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cams = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)
            cams = cams.squeeze(1).detach().cpu().numpy()
        finally:
            self._remove()
            if was_training:
                self.model.train()

        # 逐样本归一化
        spatial_shape = cams.shape[1:]
        flat = cams.reshape(len(cams), -1)
        cmin = flat.min(axis=1, keepdims=True)
        cmax = flat.max(axis=1, keepdims=True)
        flat = (flat - cmin) / np.maximum(cmax - cmin, 1e-8)
        return flat.reshape((flat.shape[0],) + spatial_shape)


def overlay_cam(image_hwc: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把 CAM 热力图叠加到原图上。"""
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * image_hwc + alpha * heat, 0, 1)


def plot_gradcam_grid(
    images_uint8: np.ndarray,
    cams: np.ndarray,
    titles: Sequence[str],
    out_path,
    cols: int = 6,
    alpha: float = 0.35,
    suptitle: str = "Grad-CAM",
) -> None:
    """images_uint8: (N,3,32,32) uint8；cams: (N,32,32)。"""
    n = len(images_uint8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.55, rows * 1.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, ax in enumerate(axes[:n]):
        img = np.transpose(images_uint8[i], (1, 2, 0)).astype(np.float32) / 255.0
        ax.imshow(overlay_cam(img, cams[i], alpha=alpha))
        ax.set_title(titles[i], fontsize=7)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    path = Path(out_path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def extract_feature_maps(model: torch.nn.Module, image: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """返回指定层对单张图片的特征图 ``(1, C, H, W)``。"""
    captured = {}

    def hook(_module, _inputs, output):
        captured["feat"] = output.detach()

    handle = layer.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(image.unsqueeze(0))
    finally:
        handle.remove()
        if was_training:
            model.train()
    if "feat" not in captured:
        raise RuntimeError("未捕获到特征图")
    return captured["feat"]
