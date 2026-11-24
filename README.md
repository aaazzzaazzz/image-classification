# 当代人工智能作业三：CIFAR-10 图像分类

本项目完整实现了一个 CIFAR-10 图像分类实验框架。
自定义 CNN、ResNet、SE/CBAM 注意力、预训练模型微调、数据增强消融、标签平滑 / Focal Loss、
多种学习率调度与早停、EMA、TTA、多模型集成、混淆矩阵与误分类分析、Grad-CAM 可解释性可视化。

- 数据集：CIFAR-10（本地 `cifar-10-batches-py`，50,000 训练 / 10,000 测试，10 类，32×32）
- 框架：PyTorch 2.7 + torchvision 0.22，单卡 NVIDIA RTX 4060 Laptop（8 GB）
- 所有实验固定随机种子 42；训练集按类别分层划分为 45,000 训练 / 5,000 验证，测试集只用于最终评估

---

## 1. 环境安装

```bash
# Python 3.10+（本项目在 Python 3.12.7 上验证）
pip install -r requirements.txt
```

`requirements.txt` 包含：torch、torchvision、numpy、pandas、matplotlib、seaborn、scikit-learn。
首次运行预训练模型时会自动从 PyTorch 官方源下载 ImageNet 权重，并缓存到项目内的 `.cache/torch/`。

GPU 可选；CPU 也能运行（把 `--device cpu` 或在配置中关闭 `amp`），但完整实验会非常慢。

## 2. 数据准备

目录结构应为：

```
cifar-10-python/
├── cifar-10-batches-py/
│   ├── data_batch_1 ... data_batch_5
│   ├── test_batch
│   └── batches.meta
├── src/
├── scripts/
└── ...
```

代码直接读取 pickle 文件（`src/data.py`），不依赖 `torchvision.datasets.CIFAR10` 的联网下载。

## 3. 目录结构

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py              # 数据读取、分层划分、数据增强（crop/flip/cutout/RandAugment/AutoAugment）、Mixup/Cutmix
│   ├── models/
│   │   ├── simple_cnn.py    # 自定义 6 层 VGG 风格 CNN（基础要求）
│   │   ├── resnet_cifar.py  # CIFAR 版 ResNet-20/32/44/56...
│   │   ├── attention.py     # SE / CBAM / ECA 注意力
│   │   └── pretrained.py    # ImageNet 预训练 ResNet18/34、MobileNetV3、VGG16-BN、ViT-B/16（32×32 适配）
│   ├── losses.py            # 交叉熵 / 标签平滑 / Focal Loss
│   ├── engine.py            # 训练循环、验证、EMA、优化器、学习率调度器
│   ├── train.py             # 单实验入口（训练→验证→测试→落盘）
│   ├── evaluate.py          # 单模型评估、TTA、多模型概率集成
│   ├── gradcam.py           # Grad-CAM 与特征图提取
│   ├── metrics.py           # 混淆矩阵、每类 P/R/F1、Top-5、Macro-F1
│   ├── visualize.py         # 全部可视化
│   └── experiments.py       # 18 个实验配置
├── scripts/
│   ├── run_experiments.py   # 批量训练（断点续跑 / 分组 / 自动集成）
│   ├── ensemble_search.py   # 验证集贪心加权集成
│   ├── make_report_assets.py# 汇总表格 + 报告图表（含 Grad-CAM / 误差分析）
│   └── build_report_docx.py # Markdown 报告转 Word
├── tests/test_components.py # 组件冒烟测试
├── outputs/                 # 训练结果（自动生成）
│   ├── results/<实验名>/    # results.json, history.json, config.json, test_probs.npy
│   ├── checkpoints/         # 最佳权重
│   ├── figures/             # 曲线、混淆矩阵、Grad-CAM、对比图
│   ├── logs/                # 每个实验的完整训练日志
│   └── evaluation/          # 单独评估 / 等权集成 / 加权集成结果
└── report/                  # 实验报告（Markdown / Word）
```

## 4. 快速开始

```bash
# 1) 冒烟测试（2 epoch、2000 张训练样本，约 10 秒）
python -m src.train --exp resnet20_basic --quick --force

# 2) 训练单个实验
python -m src.train --exp resnet20_final

# 3) 运行全部 18 个实验（自动跳过已完成项，最后训练并集成三个最终模型）
python scripts/run_experiments.py --group all --ensemble

# 4) 只跑某一组（baseline / models / aug / loss / attention / scheduler / pretrained / final / kd）
python scripts/run_experiments.py --group attention

# 5) 重新训练指定实验
python scripts/run_experiments.py --only resnet20_final --force

# 6) 查看所有预置实验
python -m src.train --list

# 7) 单模型评估 / TTA / 三模型等权集成
python -m src.evaluate --checkpoint outputs/checkpoints/resnet18_pretrained_final_best.pt --tta
python -m src.evaluate --ensemble \
  outputs/checkpoints/resnet20_final_best.pt \
  outputs/checkpoints/se_resnet20_final_best.pt \
  outputs/checkpoints/resnet18_pretrained_final_best.pt \
  --name ensemble --tta

# 8) 验证集贪心加权集成（最终 96.48% 的结果）
python scripts/ensemble_search.py

# 9) 知识蒸馏：先预计算教师概率，再训练 KD 学生
python scripts/precompute_teacher_probs.py
python scripts/run_experiments.py --group kd --force
```

Windows 下如果 DataLoader 多进程报错，可加 `--num-workers 0`（例如
`python -m src.train --exp resnet20_basic --num-workers 0`）。

## 5. 实验设计

### 5.1 基础要求

| 要求 | 实现 |
|---|---|
| 设计 CNN 模型 | `SimpleCNN`（6 个卷积层 + BN + Dropout）；`ResNet-20/18` |
| 自行划分训练/验证集 | 按类别分层划分 45,000 / 5,000，随机种子固定；测试集 10,000 仅最终评估 |
| 数据增强与正则化 | RandomCrop+Flip、Cutout、RandAugment、AutoAugment、Mixup/Cutmix；BN、Dropout、权重衰减 |
| 可视化训练过程 | 每个实验输出 loss/accuracy 曲线（含 EMA 曲线、最佳 epoch 标注） |
| 混淆矩阵与每类/总准确率 | `outputs/figures/*_confusion.png`、`*_per_class_*.png`；`results.json` 保存每类 P/R/F1 |
| 误分类样本分析 | `fig_misclassified.png`、`fig_top_confusions.png`、`fig_confidence_histogram.png` |

### 5.2 加分项

| 加分方向 | 实验 |
|---|---|
| 预训练模型微调 | `resnet18_pretrained`、`mobilenet_v3_pretrained`、`resnet18_pretrained_final` |
| 对比不同架构 | `simple_cnn`、`resnet20_basic`、`resnet18_scratch`、两个预训练模型 |
| 注意力机制 | `resnet20_se`（SE）、`resnet20_cbam`（CBAM）、`se_resnet20_final` |
| 学习率调度 / 早停 | `resnet20_step`、`resnet20_onecycle`、cosine + warmup；所有实验均带早停 |
| 数据增强策略对比 | `resnet20_aug_none`、`resnet20_basic`、`resnet20_aug_cutout`、`resnet20_aug_randaugment`、`resnet20_mixup` |
| 改进损失函数 | `resnet20_label_smoothing`、`resnet20_focal`、`resnet20_kd`（知识蒸馏） |
| 可解释性分析 | `fig_gradcam_correct.png`、`fig_gradcam_errors.png`、`fig_feature_maps_*.png` |
| 额外创新 | EMA 权重平均、TTA、三模型概率集成、Grad-CAM 误差分析 |

### 5.3 实验清单（18 个主实验 + KD）

| 实验名 | 说明 | 关键设置（原epoch数，改动后数量在后面） |
|---|---|---|
| `simple_cnn` | 自定义 CNN 基线 | 60 epoch，crop+flip |
| `resnet20_basic` | ResNet-20 基线（消融对照组） | 60 epoch，CE，cosine |
| `resnet18_scratch` | ResNet-18 从零训练 | 60 epoch，crop+flip |
| `resnet18_pretrained` | 预训练 ResNet-18 微调 | 25 epoch，lr 0.01 |
| `mobilenet_v3_pretrained` | 预训练 MobileNetV3-Small 微调 | 30 epoch，lr 0.01 |
| `resnet20_aug_none` | 无增强 | 60 epoch |
| `resnet20_aug_cutout` | Cutout(16) | 60 epoch |
| `resnet20_aug_randaugment` | RandAugment(n=2,m=9) | 60 epoch |
| `resnet20_mixup` | Mixup(α=0.2) | 60 epoch |
| `resnet20_label_smoothing` | 标签平滑 ε=0.1 | 60 epoch |
| `resnet20_focal` | Focal Loss(γ=2) | 60 epoch |
| `resnet20_se` | SE 通道注意力 | 60 epoch |
| `resnet20_cbam` | CBAM 通道+空间注意力 | 60 epoch |
| `resnet20_step` | MultiStep LR | 60 epoch |
| `resnet20_onecycle` | OneCycle LR | 60 epoch |
| `resnet20_final` | RandAugment + 标签平滑 + EMA | 120 epoch |
| `se_resnet20_final` | SE + RandAugment + 标签平滑 + EMA | 120 epoch |
| `resnet18_pretrained_final` | 预训练 ResNet-18 + Cutout + 标签平滑 + EMA | 30 epoch |
| `resnet20_kd` | ResNet-20 + 知识蒸馏（教师 ResNet-18） | 120 epoch，T=4，alpha=0.7 |
  
**表中 epoch 为学习率日程长度；18 个主实验的最大训练预算统一设为 `2×日程 + patience`，因此 18 个主实验全部触发早停；`resnet20_kd` 是额外的损失函数实验。**

除被研究因素外，同一组消融实验的超参数完全一致（SGD 0.1、momentum 0.9、weight decay 5e-4、
batch size 128、cosine + 1 epoch warmup、AMP）。

## 6. 结果

| 方法 | 测试准确率(%) | 说明 |
|---|---|---|
| ResNet-20 基线 | 91.98 | 60 轮日程，RandomCrop+Flip，早停 81 轮触发 |
| SimpleCNN | 92.51 | 1.15 M 参数 |
| ResNet-18 从零训练 | 94.35 | 60 轮日程，早停 83 轮触发 |
| ResNet-18 预训练微调 | 95.65 | 25 轮日程，早停 54 轮触发 |
| ResNet-18 预训练 final（Cutout+LS+EMA） | 95.37 | TTA 96.00 |
| ResNet-20 + KD（教师 ResNet-18） | 93.27 | TTA 93.97，120 轮日程 |
| 等权集成（3 个 final 模型） | 95.21 | 被较弱成员拖累 |
| **验证集加权集成** | **96.19** | 贪心选择权重 |
| **验证集加权集成 + TTA** | **96.48** | 最终最佳结果，比基线 +4.50 |

主要消融结论：

- 无数据增强时测试准确率仅 87.27%，RandomCrop+Flip 带来 +4.71，是最大的单项正则收益；
- Cutout、RandAugment、Mixup、标签平滑、SE/CBAM 与 MultiStep/OneCycle 均未超过基础基线，说明增强/正则强度需与训练日程匹配；
- 120 轮日程的 ResNet-20 final 为 93.07%（TTA 93.48%），SE 版本 92.64%（TTA 93.33%）；
- 知识蒸馏 `resnet20_kd` 达到 93.27%（TTA 93.97%），比 ResNet-20 final 高 0.20/0.49，是损失函数改进中最有效的方向；
- 预训练 ResNet-18 是最有效的单项手段；验证集贪心加权集成（SE final + 预训练 final + 预训练）达到 96.19%，TTA 后 96.48%；
- 18 个主实验的最大预算放大到 2×+patience 后，早停 18/18 全部触发，平均减少约 40% 训练轮数，测试平均变化 −0.03。

主要图表：

| 图 | 文件 |
|---|---|
| 训练曲线 | `outputs/figures/<实验名>_curves.png` |
| 混淆矩阵 | `outputs/figures/<实验名>_confusion.png` |
| 每类指标 | `outputs/figures/<实验名>_per_class_metrics.png` |
| 架构对比 | `outputs/figures/fig_model_comparison.png` |
| 增强消融 | `outputs/figures/fig_aug_comparison.png` |
| 损失函数对比 | `outputs/figures/fig_loss_comparison.png` |
| 注意力对比 | `outputs/figures/fig_attention_comparison.png` |
| 调度器对比 | `outputs/figures/fig_scheduler_comparison.png` |
| 最终模型 + 集成 | `outputs/figures/fig_final_comparison.png` |
| 误分类样本 | `outputs/figures/fig_misclassified.png` |
| 混淆对 | `outputs/figures/fig_top_confusions.png` |
| 置信度分布 | `outputs/figures/fig_confidence_histogram.png` |
| Grad-CAM（正确样本） | `outputs/figures/fig_gradcam_correct.png` |
| Grad-CAM（错误样本） | `outputs/figures/fig_gradcam_errors.png` |
| 特征图 | `outputs/figures/fig_feature_maps_*.png` |

## 7. 复现说明

- 所有实验随机种子固定为 42（`src/utils.py: set_seed`），数据划分、增强、初始化、shuffle 全部受控。
- 每个实验的完整配置会保存到 `outputs/results/<实验名>/config.json`，逐 epoch 指标保存到
  `history.json`，最终结果（含混淆矩阵、每类指标）保存到 `results.json`。
- 开启 AMP（自动混合精度）以在 8 GB 显存上训练；关闭 AMP 不影响结果，只是更慢。
- 由于 CUDA/cuDNN 的非确定性算子，重复运行可能有 ±0.1% 的波动；如需严格确定性，
  在配置中设置 `train.deterministic: true`。
- 集成评估使用同一测试集顺序，加载三个最佳权重后对 softmax 概率取平均。

## 8. 代码规范

- 所有模块含中文 docstring 与关键行注释；类型标注覆盖主要接口。
- 训练、评估、可视化、集成、Grad-CAM 解耦，可单独调用。
- 每个实验自动跳过已完成项，支持 `--force` 重跑；单个实验失败不阻断批量任务。
- 结果统一落盘为 JSON/CSV/PNG。
