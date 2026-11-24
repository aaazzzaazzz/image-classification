# 组件级冒烟测试：数据划分、模型前向、损失函数、Mixup、Grad-CAM。


from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data import MixupCutmix, load_cifar10, stratified_split
from src.gradcam import GradCAM
from src.losses import FocalLoss, LabelSmoothingCrossEntropy
from src.models import build_model, get_gradcam_target


class TestData(unittest.TestCase):
    def test_load_and_split(self):
        x, y, x_test, y_test, classes = load_cifar10(".")
        self.assertEqual(x.shape, (50000, 3, 32, 32))
        self.assertEqual(x_test.shape, (10000, 3, 32, 32))
        self.assertEqual(len(classes), 10)
        train_idx, val_idx = stratified_split(y, val_size=5000, seed=42)
        self.assertEqual(len(train_idx), 45000)
        self.assertEqual(len(val_idx), 5000)
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        counts = np.bincount(y[val_idx], minlength=10)
        self.assertTrue(np.all(counts == 500))

    def test_mixup(self):
        x = torch.rand(8, 3, 32, 32)
        y = torch.arange(8) % 10
        mixer = MixupCutmix(mode="mixup", alpha=0.2, prob=1.0)
        x2, ya, yb, lam = mixer(x, y)
        self.assertEqual(x2.shape, x.shape)
        self.assertEqual(ya.shape, y.shape)
        self.assertGreaterEqual(lam, 0.5)
        cutmix = MixupCutmix(mode="cutmix", alpha=1.0, prob=1.0)
        x3, _, _, lam3 = cutmix(x, y)
        self.assertEqual(x3.shape, x.shape)
        self.assertTrue(0.0 <= lam3 <= 1.0)


class TestModels(unittest.TestCase):
    def test_forward_shapes(self):
        x = torch.randn(2, 3, 32, 32)
        configs = [
            {"name": "simple_cnn"},
            {"name": "resnet20"},
            {"name": "resnet20", "attention": "se"},
            {"name": "resnet20", "attention": "cbam"},
            {"name": "resnet18_cifar", "pretrained": False},
            {"name": "mobilenet_v3_small", "pretrained": False},
        ]
        for cfg in configs:
            model = build_model(cfg).eval()
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape, (2, 10), msg=str(cfg))

    def test_gradcam(self):
        model = build_model({"name": "resnet20"}).eval()
        x = torch.randn(2, 3, 32, 32)
        layer = get_gradcam_target(model, "resnet20")
        cams = GradCAM(model, layer).generate(x, torch.tensor([1, 2]))
        self.assertEqual(cams.shape, (2, 32, 32))
        self.assertGreaterEqual(float(cams.min()), 0.0)
        self.assertLessEqual(float(cams.max()), 1.0)


class TestLosses(unittest.TestCase):
    def test_losses_finite(self):
        logits = torch.randn(4, 10, requires_grad=True)
        target = torch.tensor([0, 1, 2, 3])
        for loss_fn in (LabelSmoothingCrossEntropy(0.1), FocalLoss(2.0, 0.25)):
            loss = loss_fn(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
