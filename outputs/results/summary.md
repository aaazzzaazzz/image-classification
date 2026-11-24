# CIFAR-10 实验结果汇总

## 全部实验（按测试准确率排序）

| 实验 | 模型 | 注意力 | 预训练 | 参数量(M) | Epochs | 验证准确率(%) | 测试准确率(%) | Top5(%) | Macro-F1(%) | TTA(%) | 用时(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble_greedy_tta | ensemble | - | False | - | - | - | 96.48 | 99.94 | 96.48 | - | - |
| ensemble_greedy | ensemble | - | False | - | - | - | 96.19 | 99.91 | 96.19 | - | - |
| resnet18_pretrained | resnet18_cifar | - | True | 11.18 | 70.00 | 96.38 | 95.65 | 99.96 | 95.65 | - | 1055.85 |
| ensemble_tta | ensemble | - | False | 11.73 | - | - | 95.60 | 99.92 | 95.60 | - | - |
| resnet18_pretrained_final | resnet18_cifar | - | True | 11.18 | 75.00 | 96.34 | 95.37 | 99.81 | 95.37 | 96.00 | 650.34 |
| ensemble | ensemble | - | False | 11.73 | - | - | 95.21 | 99.94 | 95.20 | - | - |
| resnet18_scratch | resnet18_cifar | - | False | 11.18 | 140.00 | 94.62 | 94.35 | 99.87 | 94.35 | - | 1591.66 |
| resnet20_final | resnet20 | - | False | 0.27 | 280.00 | 93.72 | 93.07 | 99.79 | 93.06 | 93.48 | 6841.71 |
| se_resnet20_final | resnet20 | se | False | 0.28 | 280.00 | 94.02 | 92.64 | 99.74 | 92.64 | 93.33 | 4963.09 |
| simple_cnn | simple_cnn | - | False | 1.15 | 140.00 | 93.02 | 92.51 | 99.82 | 92.51 | - | 2069.45 |
| resnet20_basic | resnet20 | - | False | 0.27 | 140.00 | 92.18 | 91.98 | 99.77 | 91.97 | - | 1672.26 |
| resnet20_aug_randaugment | resnet20 | - | False | 0.27 | 140.00 | 92.70 | 91.91 | 99.82 | 91.90 | - | 2746.00 |
| resnet20_aug_cutout | resnet20 | - | False | 0.27 | 140.00 | 92.46 | 91.83 | 99.78 | 91.82 | - | 850.42 |
| resnet20_se | resnet20 | se | False | 0.28 | 140.00 | 92.12 | 91.82 | 99.74 | 91.82 | - | 3029.29 |
| resnet20_label_smoothing | resnet20 | - | False | 0.27 | 140.00 | 92.80 | 91.71 | 99.68 | 91.72 | - | 744.79 |
| resnet20_mixup | resnet20 | - | False | 0.27 | 140.00 | 92.80 | 91.68 | 99.76 | 91.68 | - | 1712.76 |
| resnet20_onecycle | resnet20 | - | False | 0.27 | 140.00 | 92.74 | 91.58 | 99.84 | 91.57 | - | 1570.69 |
| resnet20_step | resnet20 | - | False | 0.27 | 140.00 | 91.76 | 91.52 | 99.86 | 91.50 | - | 1321.06 |
| resnet20_cbam | resnet20 | cbam | False | 0.28 | 140.00 | 92.46 | 91.36 | 99.71 | 91.34 | - | 3897.84 |
| resnet20_focal | resnet20 | - | False | 0.27 | 140.00 | 91.88 | 91.20 | 99.75 | 91.18 | - | 830.42 |
| mobilenet_v3_pretrained | mobilenet_v3_small | - | True | 1.53 | 80.00 | 88.34 | 88.12 | 99.74 | 88.11 | - | 645.94 |
| resnet20_aug_none | resnet20 | - | False | 0.27 | 140.00 | 87.80 | 87.27 | 99.37 | 87.24 | - | 991.81 |

## 模型对比

| 实验 | 验证准确率 | 测试准确率 | Top5 | Macro-F1 | 参数量(M) | 相对基线 |
|---|---|---|---|---|---|---|
| simple_cnn | 93.02 | 92.51 | 99.82 | 92.51 | 1.149 | +0.53 |
| resnet20_basic | 92.18 | 91.98 | 99.77 | 91.97 | 0.272 | baseline |
| resnet18_scratch | 94.62 | 94.35 | 99.87 | 94.35 | 11.182 | +2.37 |
| resnet18_pretrained | 96.38 | 95.65 | 99.96 | 95.65 | 11.182 | +3.67 |
| mobilenet_v3_pretrained | 88.34 | 88.12 | 99.74 | 88.11 | 1.528 | -3.86 |

## 数据增强

| 实验 | 验证准确率 | 测试准确率 | Top5 | Macro-F1 | 参数量(M) | 相对基线 |
|---|---|---|---|---|---|---|
| resnet20_aug_none | 87.80 | 87.27 | 99.37 | 87.24 | 0.272 | -4.71 |
| resnet20_basic | 92.18 | 91.98 | 99.77 | 91.97 | 0.272 | baseline |
| resnet20_aug_cutout | 92.46 | 91.83 | 99.78 | 91.82 | 0.272 | -0.15 |
| resnet20_aug_randaugment | 92.70 | 91.91 | 99.82 | 91.90 | 0.272 | -0.07 |
| resnet20_mixup | 92.80 | 91.68 | 99.76 | 91.68 | 0.272 | -0.30 |

## 损失函数

| 实验 | 验证准确率 | 测试准确率 | Top5 | Macro-F1 | 参数量(M) | 相对基线 |
|---|---|---|---|---|---|---|
| resnet20_basic | 92.18 | 91.98 | 99.77 | 91.97 | 0.272 | baseline |
| resnet20_label_smoothing | 92.80 | 91.71 | 99.68 | 91.72 | 0.272 | -0.27 |
| resnet20_focal | 91.88 | 91.20 | 99.75 | 91.18 | 0.272 | -0.78 |

## 注意力机制

| 实验 | 验证准确率 | 测试准确率 | Top5 | Macro-F1 | 参数量(M) | 相对基线 |
|---|---|---|---|---|---|---|
| resnet20_basic | 92.18 | 91.98 | 99.77 | 91.97 | 0.272 | baseline |
| resnet20_se | 92.12 | 91.82 | 99.74 | 91.82 | 0.276 | -0.16 |
| resnet20_cbam | 92.46 | 91.36 | 99.71 | 91.34 | 0.276 | -0.62 |

## 学习率调度

| 实验 | 验证准确率 | 测试准确率 | Top5 | Macro-F1 | 参数量(M) | 相对基线 |
|---|---|---|---|---|---|---|
| resnet20_basic | 92.18 | 91.98 | 99.77 | 91.97 | 0.272 | baseline |
| resnet20_step | 91.76 | 91.52 | 99.86 | 91.50 | 0.272 | -0.46 |
| resnet20_onecycle | 92.74 | 91.58 | 99.84 | 91.57 | 0.272 | -0.40 |

## 综合模型

| 实验 | 验证准确率 | 测试准确率 | Top5 | Macro-F1 | 参数量(M) | 相对基线 |
|---|---|---|---|---|---|---|
| resnet20_basic | 92.18 | 91.98 | 99.77 | 91.97 | 0.272 | baseline |
| resnet20_final | 93.72 | 93.07 | 99.79 | 93.06 | 0.272 | +1.09 |
| se_resnet20_final | 94.02 | 92.64 | 99.74 | 92.64 | 0.276 | +0.66 |
| resnet18_pretrained_final | 96.34 | 95.37 | 99.81 | 95.37 | 11.182 | +3.39 |
| ensemble | nan | 95.21 | 99.94 | 95.20 | 11.730 | +3.23 |
| ensemble_greedy | nan | 96.19 | 99.91 | 96.19 | 0.000 | +4.21 |
| ensemble_greedy_tta | nan | 96.48 | 99.94 | 96.48 | 0.000 | +4.50 |

## 生成的图表

- `outputs/figures/fig_accuracy_vs_params.png`
- `outputs/figures/fig_attention_comparison.png`
- `outputs/figures/fig_aug_comparison.png`
- `outputs/figures/fig_augmentation_examples.png`
- `outputs/figures/fig_best_confusion_matrix.png`
- `outputs/figures/fig_best_per_class_accuracy.png`
- `outputs/figures/fig_best_per_class_metrics.png`
- `outputs/figures/fig_confidence_histogram.png`
- `outputs/figures/fig_dataset_samples.png`
- `outputs/figures/fig_feature_maps_last.png`
- `outputs/figures/fig_feature_maps_stem.png`
- `outputs/figures/fig_final_comparison.png`
- `outputs/figures/fig_gradcam_correct.png`
- `outputs/figures/fig_gradcam_errors.png`
- `outputs/figures/fig_loss_comparison.png`
- `outputs/figures/fig_lr_schedules.png`
- `outputs/figures/fig_misclassified.png`
- `outputs/figures/fig_model_comparison.png`
- `outputs/figures/fig_per_class_heatmap.png`
- `outputs/figures/fig_scheduler_comparison.png`
- `outputs/figures/fig_top_confusions.png`
- `outputs/figures/fig_val_curves_comparison.png`
- `outputs/figures/fig_val_vs_test.png`

误差分析 / Grad-CAM 使用的最佳单模型: `resnet18_pretrained`