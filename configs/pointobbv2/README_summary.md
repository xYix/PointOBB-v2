# CPM 系列模型总结

## 概述

本项目实现了 CPM (Center Point Matching) 目标检测模型的多个变体，从基础版本到集成 VPD 变分推断的高级版本。

## 模型变体

### 1. CPMHead - 基础版本
**文件**: `mmrotate/models/dense_heads/cpm_head.py`
**配置**: `configs/pointobbv2/train_cpm_dotav10.py`

**特点**:
- 仅使用分类损失
- 基于点到 GT 中心距离的标签分配
- 回归和 centerness 损失权重为 0

**损失函数**:
```python
loss_cls = cls_weight * FocalLoss
loss_bbox = 0
loss_centerness = 0
```

---

### 2. CPMHboxHead - 水平框版本
**文件**: `mmrotate/models/dense_heads/cpm_hbox_head.py`
**配置**: `configs/pointobbv2/train_cpm_hbox_dotav10.py`

**特点**:
- 输出水平边界框（Hbox）而非旋转框（Rbox）
- 使用标准 FCOS 检测器
- 保留 CPM 标签分配策略

**适用场景**: 目标方向不重要的场景

---

### 3. CPMRegHead - 回归分支版本
**文件**: `mmrotate/models/dense_heads/cpm_reg_head.py`
**配置**: `configs/pointobbv2/train_cpm_reg_dotav10.py`

**特点**:
- 添加完整的回归分支
- 使用 RotatedIoU Loss
- 分类和回归联合优化

**损失函数**:
```python
loss_cls = cls_weight * FocalLoss
loss_bbox = reg_weight * RotatedIoULoss
loss_centerness = CrossEntropyLoss
```

---

### 4. CPMVPDHead - VPD 变分推断版本
**文件**: `mmrotate/models/dense_heads/cpm_vpd_head.py`
**配置**:
- 完整监督: `configs/pointobbv2/train_cpm_vpd_dotav10.py`
- 点监督: `configs/pointobbv2/train_cpm_vpd_point_dotav10.py`

**特点**:
- 预测分布参数（均值 + log_std）
- 高斯重参数化采样
- 支持两种监督模式

#### 4.1 完整监督模式 (use_point_supervised=False)

**损失函数**:
```python
loss_cls = cls_weight * FocalLoss
loss_js = JSLoss(pred_dist, bbox_target)  # JS 散度
loss_bbox = 0  # 保持与 CPM 一致
loss_centerness = 0
```

**VPD 机制**:
1. 预测 10 通道：5 均值 + 5 log_std
2. 训练时采样：`bbox = mean + exp(log_std) * noise`
3. JS 散度损失：最大化预测分布与 GT 分布的相似度

#### 4.2 点监督模式 (use_point_supervised=True)

**损失函数**:
```python
loss_cls = cls_weight * FocalLoss
loss_vpd = PointSupervisedVPDLoss(pred_dist, points, gt_centers)
  ├─ center_weight * CenterRegressionLoss
  ├─ uncertainty_weight * UncertaintyRegularization
  └─ kl_weight * KLDivergenceLoss
```

**点监督 VPD 机制**:
1. **中心点回归**: 预测 bbox 中心接近 GT 中心
2. **不确定性正则化**: 鼓励低方差（高置信度）
3. **KL 散度正则化**: 约束到先验分布

**优势**: 只需要 GT 中心点，无需完整 bbox 标注

---

## 对比表格

| 模型 | 输出 | 监督信号 | 主要损失 | 变分推断 | 适用场景 |
|------|------|---------|---------|---------|---------|
| CPMHead | 5 通道 | 完整 bbox | 分类 | ✗ | 基础检测 |
| CPMHboxHead | 4 通道 | 完整 bbox | 分类 | ✗ | 水平框检测 |
| CPMRegHead | 5 通道 | 完整 bbox | 分类+回归 | ✗ | 精确定位 |
| CPMVPDHead (Full) | 10 通道 | 完整 bbox | 分类+JS | ✓ | 不确定性建模 |
| CPMVPDHead (Point) | 10 通道 | 仅中心点 | 分类+Center+KL | ✓ | 弱监督检测 |

## 文件结构

```
mmrotate/
├── models/
│   ├── dense_heads/
│   │   ├── cpm_head.py              # 基础 CPM
│   │   ├── cpm_hbox_head.py         # 水平框版本
│   │   ├── cpm_reg_head.py          # 回归分支版本
│   │   └── cpm_vpd_head.py          # VPD 变分推断版本
│   └── losses/
│       ├── js_loss.py               # JS 散度损失
│       └── point_supervised_vpd_loss.py  # 点监督 VPD 损失
└── configs/pointobbv2/
    ├── train_cpm_dotav10.py         # 基础 CPM
    ├── train_cpm_hbox_dotav10.py    # 水平框
    ├── train_cpm_reg_dotav10.py     # 回归分支
    ├── train_cpm_vpd_dotav10.py     # VPD 完整监督
    ├── train_cpm_vpd_point_dotav10.py  # VPD 点监督
    ├── README_hbox.md               # 水平框说明
    ├── README_reg.md                # 回归分支说明
    ├── README_vpd.md                # VPD 完整监督说明
    └── README_point_supervised.md   # 点监督说明
```

## 使用建议

### 选择模型

1. **基础检测任务** → CPMHead
   - 简单快速
   - 只需要分类准确

2. **不关心旋转角度** → CPMHboxHead
   - 输出水平框
   - 计算更快

3. **需要精确定位** → CPMRegHead
   - 完整回归分支
   - 更准确的边界框

4. **需要不确定性估计** → CPMVPDHead (Full)
   - 变分推断
   - 可以评估预测置信度

5. **只有中心点标注** → CPMVPDHead (Point)
   - 弱监督学习
   - 仍能建模不确定性

### 训练命令

```bash
# 基础 CPM
python tools/train.py configs/pointobbv2/train_cpm_dotav10.py

# 水平框
python tools/train.py configs/pointobbv2/train_cpm_hbox_dotav10.py

# 回归分支
python tools/train.py configs/pointobbv2/train_cpm_reg_dotav10.py

# VPD 完整监督
python tools/train.py configs/pointobbv2/train_cpm_vpd_dotav10.py

# VPD 点监督
python tools/train.py configs/pointobbv2/train_cpm_vpd_point_dotav10.py
```

## 核心创新

### 1. CPM 标签分配策略
- 基于点到 GT 中心的距离
- 正样本：`dist < min(dist_to_nearest_gt/2, thresh1)`
- 负样本：`dist > alpha * dist_to_nearest_gt`
- 额外负样本：同类别实例中点附近

### 2. VPD 变分推断
- 预测分布而非点估计
- 高斯重参数化：`x = μ + σ * ε`
- JS 散度损失：最大化分布相似度

### 3. 点监督 VPD
- 无需完整 bbox 标注
- 中心点回归 + 不确定性正则化 + KL 散度
- 在弱监督场景下仍能训练变分推断模型

## 参数调整

### 通用参数
```python
cls_weight = 20      # 分类损失权重
thresh1 = 8          # 正样本距离阈值
alpha = 1.5          # 负样本距离系数
```

### VPD 特定参数
```python
# 完整监督
js_weight = 1.0      # JS 散度权重

# 点监督
center_weight = 1.0       # 中心点回归权重
uncertainty_weight = 0.1  # 不确定性正则化权重
kl_weight = 0.01         # KL 散度权重
```

## 理论基础

### VPD 框架
参考论文和 `material/network.py` 实现：
1. 变分推断：学习 bbox 的概率分布
2. 重参数化技巧：使梯度可以反向传播
3. JS 散度：测量分布相似度

### 点监督学习
参考 `material/point_supervised_vpd_loss.md`：
1. 中心点约束：利用有限的监督信号
2. 不确定性建模：在弱监督下仍保持鲁棒性
3. 先验正则化：引入归纳偏置

## 总结

本项目提供了从基础到高级的完整 CPM 模型系列：
- **CPMHead**: 简单高效的基础版本
- **CPMHboxHead**: 适用于水平框场景
- **CPMRegHead**: 添加回归分支提升精度
- **CPMVPDHead**: 集成 VPD 变分推断，支持完整监督和点监督两种模式

所有模型都经过验证可以正常加载和训练。
