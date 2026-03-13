# CPM with VPD Variational Inference (CPMVPDHead)

## 概述

参考 VPD (Variational Pedestrian Detection) 框架的完整实现，为 CPM 添加了变分推断机制。该实现严格遵循 VPD 的核心思想：预测均值和标准差，使用高斯重参数化采样，并通过 JS 散度损失优化分布。

## VPD 核心机制

根据 `material/network.py` 的实现，VPD 的关键特性：

1. **预测分布参数**
   ```python
   # 预测 8 个通道：4 个均值 + 4 个 log_std
   all_pred_mean = all_pred_dist[:, :4]
   all_pred_lstd = all_pred_dist[:, 4:]
   ```

2. **高斯重参数化采样**
   ```python
   # 训练时从分布中采样
   all_pred_reg = all_pred_mean + all_pred_lstd.exp() * torch.randn_like(all_pred_mean)
   ```

3. **JS 散度损失**
   ```python
   # 最大化预测分布与 GT 分布的相似度
   loss_jsd = js_loss(all_pred_dist[fg_mask], bbox_target[fg_mask], config.js_weight)
   ```

4. **三个损失函数**
   - `loss_cls`: Focal Loss（分类）
   - `loss_reg`: Smooth L1 Loss（回归）
   - `loss_jsd`: JS Divergence Loss（分布相似度）

## CPMVPDHead 实现

### 主要特性

1. **变分推断输出**
   - 预测 10 个通道：5 个均值（bbox 4 + angle 1）+ 5 个 log_std
   - 训练时使用高斯重参数化采样
   - 推理时使用均值

2. **CPM 标签分配**
   - 保留 CPM 的距离基准标签分配策略
   - 正样本：距离 < min(dist_to_nearest_gt/2, thresh1)
   - 负样本：距离 > alpha * dist_to_nearest_gt

3. **损失函数**
   ```python
   loss_cls = cls_weight * FocalLoss(classification)
   loss_js = JSLoss(distribution, target)  # VPD 核心
   # loss_bbox 和 loss_centerness 保持为 0（与原始 CPM 一致）
   ```

### 网络结构

```
Input Features
    ↓
Classification Branch → cls_score
    ↓
Regression Branch → bbox_dist (10 channels)
    ├─ bbox_mean (4 channels)
    ├─ bbox_lstd (4 channels)
    ├─ angle_mean (1 channel)
    └─ angle_lstd (1 channel)
    ↓
Training: Gaussian Reparameterization
    bbox_sample = bbox_mean + exp(bbox_lstd) * noise
    ↓
Losses:
    - Focal Loss (classification)
    - JS Divergence Loss (distribution similarity)
```

### 关键代码

#### 1. 前向传播

```python
def forward_single(self, x, scale, stride):
    # 预测分布：10 通道（5 均值 + 5 log_std）
    bbox_dist = self.conv_reg(reg_feat)

    bbox_mean = bbox_dist[:, :5]  # bbox(4) + angle(1) 均值
    bbox_lstd = bbox_dist[:, 5:]  # bbox(4) + angle(1) log_std

    # 缩放均值
    bbox_mean = scale(bbox_mean).float()

    # 分离 bbox 和 angle
    bbox_pred = torch.cat([bbox_mean[:, :4], bbox_lstd[:, :4]], dim=1)  # 8 通道
    angle_pred = torch.cat([bbox_mean[:, 4:5], bbox_lstd[:, 4:5]], dim=1)  # 2 通道

    return cls_score, bbox_pred, angle_pred, centerness
```

#### 2. 损失计算

```python
def loss(self, ...):
    # 分离均值和 log_std
    pos_bbox_mean = pos_bbox_dist[:, :4]
    pos_bbox_lstd = pos_bbox_dist[:, 4:]
    pos_angle_mean = pos_angle_dist[:, 0:1]
    pos_angle_lstd = pos_angle_dist[:, 1:2]

    # VPD: 高斯重参数化采样
    pos_bbox_reg = pos_bbox_mean + pos_bbox_lstd.exp() * torch.randn_like(pos_bbox_mean)
    pos_angle_reg = pos_angle_mean + pos_angle_lstd.exp() * torch.randn_like(pos_angle_mean)

    # 解码并计算回归损失（用于梯度，但权重为 0）
    pos_bbox_pred = torch.cat([pos_bbox_reg, pos_angle_reg], dim=-1)
    pos_decoded_bbox_preds = bbox_coder.decode(pos_points, pos_bbox_pred)

    # JS 散度损失（VPD 核心）
    pos_dist_full = torch.cat([pos_bbox_dist, pos_angle_dist], dim=-1)  # 10 通道
    pos_target_full = torch.cat([pos_bbox_targets, pos_angle_targets], dim=-1)  # 5 通道
    loss_js = self.loss_js(pos_dist_full, pos_target_full, weight=pos_centerness_targets)

    return dict(
        loss_cls=cls_weight * loss_cls,
        loss_bbox=0. * loss_bbox,  # 保持为 0
        loss_centerness=0. * loss_centerness,  # 保持为 0
        loss_js=loss_js)  # VPD 损失
```

#### 3. JS 散度损失

```python
class JSLoss(nn.Module):
    def forward(self, pred_dist, target, weight=None):
        # pred_dist: (N, 10) - 5 均值 + 5 log_std
        # target: (N, 5) - GT 值

        pred_mean = pred_dist[:, :5]
        pred_lstd = pred_dist[:, 5:]
        pred_std = pred_lstd.exp()
        diff = pred_mean - target

        # JS 散度近似
        js_div = 0.5 * (
            pred_lstd +  # log(σ)
            (diff.pow(2) / (pred_std.pow(2) + 1e-6)) +  # (μ-target)²/σ²
            pred_std.pow(2) -  # σ²
            1.0  # 常数项
        )

        loss = js_div.sum(dim=-1)
        return loss_weight * loss.mean()
```

## 与其他版本的对比

| 特性 | CPMHead | CPMRegHead | CPMVPDHead |
|------|---------|------------|------------|
| 分类损失 | ✓ | ✓ | ✓ |
| 回归损失 | ✗ | ✓ | ✗ (权重为 0) |
| JS 散度损失 | ✗ | ✗ | ✓ (VPD 核心) |
| 输出通道 | 5 (bbox+angle) | 5 (bbox+angle) | 10 (mean+lstd) |
| 采样机制 | 无 | 无 | 高斯重参数化 |
| 标签分配 | CPM | CPM | CPM |
| 推理方式 | 直接预测 | 直接预测 | 使用均值 |

## 配置参数

```python
bbox_head=dict(
    type='CPMVPDHead',
    # ... 其他参数 ...
)

train_cfg=dict(
    cls_weight=20,        # 分类损失权重
    thresh1=8,            # 正样本距离阈值
    alpha=1,              # 负样本距离系数
    js_weight=1.0         # JS 散度损失权重（VPD）
)
```

## 使用方法

### 训练

```bash
python tools/train.py configs/pointobbv2/train_cpm_vpd_dotav10.py
```

### 测试

```bash
python tools/test.py configs/pointobbv2/train_cpm_vpd_dotav10.py \
    work_dirs/train_cpm_vpd_dotav10/latest.pth \
    --eval mAP
```

## VPD 的优势

1. **不确定性建模**: 通过预测分布而非点估计，模型可以表达预测的不确定性
2. **更鲁棒的训练**: 高斯重参数化提供了正则化效果，防止过拟合
3. **分布匹配**: JS 散度损失鼓励预测分布接近真实分布，提高泛化能力
4. **理论基础**: 基于变分推断的理论框架，有坚实的数学基础

## 文件结构

```
mmrotate/
├── models/
│   ├── dense_heads/
│   │   ├── cpm_head.py          # 原始 CPM
│   │   ├── cpm_reg_head.py      # CPM + 回归分支
│   │   └── cpm_vpd_head.py      # CPM + VPD 变分推断（新增）
│   └── losses/
│       └── js_loss.py           # JS 散度损失（新增）
└── configs/pointobbv2/
    ├── train_cpm_dotav10.py     # 原始 CPM
    ├── train_cpm_reg_dotav10.py # CPM + 回归
    └── train_cpm_vpd_dotav10.py # CPM + VPD（新增）
```

## 注意事项

1. **采样的随机性**: 训练时使用随机采样，每次前向传播结果略有不同
2. **推理使用均值**: 测试时直接使用预测的均值，不进行采样
3. **JS 权重调整**: `js_weight` 可以根据验证集表现调整，建议范围 [0.5, 2.0]
4. **与原始 CPM 的一致性**: 保持 `loss_bbox` 和 `loss_centerness` 为 0，只使用分类损失和 JS 损失
