# Point-Supervised VPD Loss Design

## 问题分析

在点监督目标检测中：
- **可用信息**: GT 中心点位置 (x, y) 和类别标签
- **不可用信息**: GT bbox 的宽高、角度等完整信息
- **挑战**: 原始 VPD 的 JS loss 需要完整的 bbox target，无法直接使用

## 解决方案

### 方案 1: 基于不确定性的自监督损失

不需要 GT bbox，而是通过最小化预测分布的不确定性来训练：

```python
# 鼓励模型在正样本位置产生低方差（高置信度）的预测
loss_uncertainty = pred_lstd.exp().mean()  # 最小化标准差
```

**优点**: 完全不需要 bbox 信息
**缺点**: 可能导致过度自信的预测

### 方案 2: KL 散度正则化

使用 KL 散度将预测分布约束到一个先验分布：

```python
# 假设先验分布为标准正态分布 N(0, 1)
# KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
loss_kl = 0.5 * (pred_mean.pow(2) + pred_std.pow(2) - pred_lstd - 1).sum(dim=-1)
```

**优点**: 提供正则化，防止分布退化
**缺点**: 先验分布的选择需要调整

### 方案 3: 对比学习损失（推荐）

利用同类别实例之间的相对关系：

```python
# 同类别的实例应该有相似的 bbox 分布
# 不同类别的实例应该有不同的 bbox 分布
loss_contrastive = contrastive_loss(pred_dist, labels)
```

### 方案 4: 伪标签 + 分布匹配（最实用）

结合 CPM 的思想，使用伪 bbox 标签：

1. 根据点到 GT 中心的距离生成伪 bbox
2. 使用伪 bbox 计算分布损失
3. 使用置信度加权

```python
# 生成伪 bbox target
pseudo_bbox = generate_pseudo_bbox(points, gt_centers, distances)

# 计算加权损失
confidence = compute_confidence(distances, thresh1)
loss_dist = distribution_loss(pred_dist, pseudo_bbox, weight=confidence)
```

### 方案 5: 负对数似然损失（NLL）- 最简单有效

直接最大化在采样点处观察到 GT 中心点的似然：

```python
# 预测分布在 GT 中心点处的负对数似然
# 鼓励预测的 bbox 中心接近 GT 中心
loss_nll = -log_likelihood(pred_dist, gt_centers)
```

## 推荐实现：混合损失

结合多个损失函数，平衡不同目标：

```python
# 1. 分类损失（已有）
loss_cls = focal_loss(pred_cls, labels)

# 2. 中心点回归损失
# 预测的 bbox 中心应该接近 GT 中心点
pred_center = decode_center(points, pred_mean)
loss_center = smooth_l1_loss(pred_center, gt_centers)

# 3. 不确定性正则化
# 鼓励模型在正样本处产生低方差预测
loss_uncertainty = pred_std.mean()

# 4. KL 散度正则化
# 防止分布退化
loss_kl = kl_divergence(pred_dist, prior_dist)

# 总损失
loss = loss_cls + λ1 * loss_center + λ2 * loss_uncertainty + λ3 * loss_kl
```

## 详细实现

### 中心点回归损失

```python
def center_regression_loss(pred_dist, points, gt_centers):
    """
    Args:
        pred_dist: (N, 10) - 预测分布 [bbox_mean(4), bbox_lstd(4), angle_mean(1), angle_lstd(1)]
        points: (N, 2) - 采样点位置
        gt_centers: (N, 2) - GT 中心点位置
    """
    # 提取 bbox 均值预测
    pred_bbox_mean = pred_dist[:, :4]  # [left, top, right, bottom]

    # 解码得到预测的 bbox 中心
    # center_x = point_x + (right - left) / 2
    # center_y = point_y + (bottom - top) / 2
    pred_center_x = points[:, 0] + (pred_bbox_mean[:, 2] - pred_bbox_mean[:, 0]) / 2
    pred_center_y = points[:, 1] + (pred_bbox_mean[:, 3] - pred_bbox_mean[:, 1]) / 2
    pred_center = torch.stack([pred_center_x, pred_center_y], dim=-1)

    # 计算中心点距离损失
    loss = F.smooth_l1_loss(pred_center, gt_centers, reduction='none').sum(dim=-1)

    return loss
```

### 不确定性正则化损失

```python
def uncertainty_regularization_loss(pred_dist):
    """
    鼓励模型产生低方差（高置信度）的预测

    Args:
        pred_dist: (N, 10) - 预测分布
    """
    # 提取 log_std
    pred_lstd = pred_dist[:, 4:]  # [bbox_lstd(4), angle_lstd(1)]

    # 最小化标准差（鼓励高置信度）
    pred_std = pred_lstd.exp()
    loss = pred_std.mean(dim=-1)

    return loss
```

### KL 散度正则化

```python
def kl_divergence_loss(pred_dist, prior_mean=0.0, prior_std=1.0):
    """
    KL(q||p) where q is predicted distribution, p is prior

    Args:
        pred_dist: (N, 10) - 预测分布
        prior_mean: float - 先验均值
        prior_std: float - 先验标准差
    """
    pred_mean = pred_dist[:, :5]  # [bbox(4), angle(1)]
    pred_lstd = pred_dist[:, 5:]
    pred_std = pred_lstd.exp()

    # KL(N(μ, σ²) || N(μ0, σ0²))
    kl = torch.log(prior_std / pred_std) + \
         (pred_std.pow(2) + (pred_mean - prior_mean).pow(2)) / (2 * prior_std ** 2) - 0.5

    loss = kl.sum(dim=-1)

    return loss
```

### 完整损失函数

```python
def point_supervised_vpd_loss(pred_cls, pred_dist, points, gt_centers, gt_labels):
    """
    点监督 VPD 损失

    Args:
        pred_cls: (N, C) - 分类预测
        pred_dist: (N, 10) - 分布预测
        points: (N, 2) - 采样点
        gt_centers: (N, 2) - GT 中心点
        gt_labels: (N,) - GT 标签
    """
    # 1. 分类损失
    loss_cls = focal_loss(pred_cls, gt_labels)

    # 2. 中心点回归损失
    loss_center = center_regression_loss(pred_dist, points, gt_centers)

    # 3. 不确定性正则化
    loss_uncertainty = uncertainty_regularization_loss(pred_dist)

    # 4. KL 散度正则化
    loss_kl = kl_divergence_loss(pred_dist)

    # 加权组合
    loss = loss_cls + \
           1.0 * loss_center.mean() + \
           0.1 * loss_uncertainty.mean() + \
           0.01 * loss_kl.mean()

    return {
        'loss_cls': loss_cls,
        'loss_center': loss_center.mean(),
        'loss_uncertainty': loss_uncertainty.mean(),
        'loss_kl': loss_kl.mean()
    }
```

## 权重调整建议

```python
# 初始训练阶段（前 1/3 epoch）
λ_center = 1.0        # 强调中心点对齐
λ_uncertainty = 0.01  # 弱正则化
λ_kl = 0.001          # 弱正则化

# 中期训练（中间 1/3 epoch）
λ_center = 1.0
λ_uncertainty = 0.1   # 增强置信度约束
λ_kl = 0.01

# 后期训练（最后 1/3 epoch）
λ_center = 0.5        # 减弱中心点约束
λ_uncertainty = 0.2   # 进一步增强置信度
λ_kl = 0.05           # 增强分布正则化
```

## 总结

对于点监督目标检测，推荐使用：

1. **主损失**: 中心点回归损失（确保预测 bbox 中心接近 GT 中心）
2. **辅助损失 1**: 不确定性正则化（鼓励高置信度预测）
3. **辅助损失 2**: KL 散度正则化（防止分布退化）

这种组合不需要完整的 bbox 标注，只需要中心点信息即可训练变分推断模型。
