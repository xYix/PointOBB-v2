# Point-Supervised CPM with VPD

## 概述

针对点监督目标检测任务（只有 GT 中心点，没有完整 bbox 标注），实现了基于 VPD 变分推断的 CPM 模型。

## 问题与解决方案

### 问题
- **点监督约束**: 训练时只能使用 GT bbox 的中心点信息
- **原始 VPD 不适用**: JS 散度损失需要完整的 bbox target（宽高、角度等）

### 解决方案
设计了点监督 VPD 损失（PointSupervisedVPDLoss），包含三个组件：

1. **中心点回归损失** (Center Regression Loss)
   - 预测的 bbox 中心应该接近 GT 中心点
   - 不需要完整 bbox 信息

2. **不确定性正则化** (Uncertainty Regularization)
   - 鼓励模型在正样本位置产生低方差（高置信度）预测
   - 防止模型过度不确定

3. **KL 散度正则化** (KL Divergence Regularization)
   - 将预测分布约束到先验分布
   - 防止分布退化或崩溃

## 损失函数详解

### 1. 中心点回归损失

```python
# 从预测的 bbox 分布解码中心点
pred_center_x = point_x + (right - left) / 2
pred_center_y = point_y + (bottom - top) / 2

# 与 GT 中心点对齐
loss_center = smooth_l1_loss(pred_center, gt_center)
```

**作用**: 确保预测的 bbox 中心接近 GT 中心点

### 2. 不确定性正则化

```python
# 提取预测的 log_std
pred_lstd = pred_dist[:, 5:]  # bbox(4) + angle(1) log_std

# 最小化标准差（鼓励高置信度）
pred_std = pred_lstd.exp()
loss_uncertainty = pred_std.mean()
```

**作用**: 鼓励模型产生确定性的预测，避免过度不确定

### 3. KL 散度正则化

```python
# KL(N(μ, σ²) || N(μ0, σ0²))
kl = log(σ0/σ) + (σ² + (μ-μ0)²)/(2σ0²) - 0.5
loss_kl = kl.sum()
```

**作用**: 将预测分布约束到先验分布（通常为标准正态分布），防止分布退化

### 总损失

```python
loss_vpd = λ1 * loss_center + λ2 * loss_uncertainty + λ3 * loss_kl

# 默认权重
λ1 = 1.0   # center_weight
λ2 = 0.1   # uncertainty_weight
λ3 = 0.01  # kl_weight
```

## 与其他版本对比

| 特性 | CPMHead | CPMVPDHead (Full) | CPMVPDHead (Point) |
|------|---------|-------------------|-------------------|
| 监督信号 | 完整 bbox | 完整 bbox | 仅中心点 |
| 输出 | 5 通道 | 10 通道 (mean+lstd) | 10 通道 (mean+lstd) |
| 分类损失 | ✓ | ✓ | ✓ |
| VPD 损失 | ✗ | JS 散度 | Center + Uncertainty + KL |
| 采样机制 | 无 | 高斯重参数化 | 高斯重参数化 |
| 需要 bbox 标注 | 是 | 是 | 否（仅中心点） |

## 实现细节

### CPMVPDHead 修改

```python
class CPMVPDHead(CPMHead):
    def __init__(self, *args, use_point_supervised=False, **kwargs):
        self.use_point_supervised = use_point_supervised

        if self.use_point_supervised:
            # 点监督模式
            self.loss_vpd = build_loss(dict(
                type='PointSupervisedVPDLoss',
                center_weight=1.0,
                uncertainty_weight=0.1,
                kl_weight=0.01))
        else:
            # 完整监督模式
            self.loss_js = build_loss(dict(type='JSLoss'))

    def loss(self, ...):
        if self.use_point_supervised:
            # 使用点监督损失
            loss_vpd = self.loss_vpd(pred_dist, points, gt_centers)
        else:
            # 使用 JS 散度损失
            loss_js = self.loss_js(pred_dist, bbox_targets)
```

### PointSupervisedVPDLoss 实现

```python
class PointSupervisedVPDLoss(nn.Module):
    def forward(self, pred_dist, points, gt_centers):
        # 1. 中心点回归
        pred_center = decode_center(pred_dist, points)
        loss_center = smooth_l1_loss(pred_center, gt_centers)

        # 2. 不确定性正则化
        pred_std = pred_dist[:, 5:].exp()
        loss_uncertainty = pred_std.mean()

        # 3. KL 散度正则化
        loss_kl = kl_divergence(pred_dist, prior)

        # 组合
        return center_weight * loss_center + \
               uncertainty_weight * loss_uncertainty + \
               kl_weight * loss_kl
```

## 配置文件

### 点监督模式

```python
# configs/pointobbv2/train_cpm_vpd_point_dotav10.py
model = dict(
    bbox_head=dict(
        type='CPMVPDHead',
        # ... 其他参数 ...
    ),
    train_cfg=dict(
        use_point_supervised=True,  # 启用点监督模式
        js_weight=1.0,              # VPD 损失权重
        cls_weight=20,              # 分类损失权重
        thresh1=8,
        alpha=1
    )
)
```

### 完整监督模式

```python
# configs/pointobbv2/train_cpm_vpd_dotav10.py
model = dict(
    train_cfg=dict(
        use_point_supervised=False,  # 使用完整监督
        js_weight=1.0
    )
)
```

## 使用方法

### 训练（点监督）

```bash
python tools/train.py configs/pointobbv2/train_cpm_vpd_point_dotav10.py
```

### 训练（完整监督）

```bash
python tools/train.py configs/pointobbv2/train_cpm_vpd_dotav10.py
```

### 测试

```bash
python tools/test.py configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
    work_dirs/train_cpm_vpd_point_dotav10/latest.pth \
    --eval mAP
```

## 超参数调整

### 损失权重

```python
# 初始阶段：强调中心点对齐
center_weight = 1.0
uncertainty_weight = 0.01
kl_weight = 0.001

# 中期：增强置信度约束
center_weight = 1.0
uncertainty_weight = 0.1
kl_weight = 0.01

# 后期：平衡所有损失
center_weight = 0.5
uncertainty_weight = 0.2
kl_weight = 0.05
```

### 先验分布

```python
# 标准正态先验（默认）
prior_mean = 0.0
prior_std = 1.0

# 可根据数据集调整
# 例如：如果 bbox 通常较大
prior_mean = 0.0
prior_std = 2.0
```

## 优势

1. **无需完整标注**: 只需要中心点即可训练变分推断模型
2. **不确定性建模**: 仍然保留 VPD 的不确定性建模能力
3. **正则化效果**: 通过不确定性和 KL 正则化防止过拟合
4. **灵活切换**: 同一个模型可以在点监督和完整监督之间切换

## 文件结构

```
mmrotate/
├── models/
│   ├── dense_heads/
│   │   └── cpm_vpd_head.py              # 支持点监督和完整监督
│   └── losses/
│       ├── js_loss.py                   # JS 散度损失（完整监督）
│       └── point_supervised_vpd_loss.py # 点监督 VPD 损失（新增）
└── configs/pointobbv2/
    ├── train_cpm_vpd_dotav10.py         # 完整监督配置
    └── train_cpm_vpd_point_dotav10.py   # 点监督配置（新增）
```

## 理论基础

### 为什么这样设计有效？

1. **中心点约束**: 即使没有完整 bbox，中心点信息也能指导模型学习合理的 bbox 分布

2. **不确定性正则化**: 防止模型在缺乏完整监督时产生过度不确定的预测

3. **KL 散度正则化**: 提供先验知识，引导模型学习合理的分布参数

4. **变分推断框架**: 通过预测分布而非点估计，模型可以表达预测的不确定性，这在弱监督场景下特别有价值

## 预期效果

相比标准点监督方法：
- **更鲁棒**: 变分推断提供正则化效果
- **更准确**: 不确定性建模帮助模型识别困难样本
- **更可解释**: 预测的标准差反映模型的置信度

## 注意事项

1. **GT 中心点获取**: 当前实现使用采样点作为 GT 中心的近似，实际应用中可能需要从 bbox 标注中提取真实中心点

2. **权重调整**: 三个损失组件的权重需要根据具体数据集调整

3. **训练策略**: 建议使用渐进式权重调整策略（初期强调中心点，后期增强正则化）

4. **推理一致性**: 推理时使用预测的均值，与训练时的采样机制一致
