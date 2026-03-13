# CPM with Regression Branch (CPMRegHead)

## 概述

参考 VPD (Variational Pedestrian Detection) 框架的思想，为 CPM 模型添加了 regression 分支。该实现结合了 CPM 的标签分配策略和标准的边界框回归损失。

## VPD 框架核心思想

根据 VPD.md 的描述，VPD 框架通过以下方式增强检测器：

1. **变分推断模块**: 对每个 proposal 执行变分推断
2. **统计解码器**: 连接分类和回归分支，增强置信度估计
3. **两个约束条件**:
   - 最大化 proposal 分布与 GT 先验之间的相似度（JS 散度）
   - 最大化从变分推断分布生成的 proposal 的检测似然（回归指标）

## CPMRegHead 实现

### 主要特性

1. **保留 CPM 标签分配策略**
   - 基于点到 GT 中心的距离进行标签分配
   - 正样本: 距离 < min(dist_to_nearest_gt/2, thresh1)
   - 负样本: 距离 > alpha * dist_to_nearest_gt
   - 额外负样本: 同类别实例中点附近

2. **添加回归分支**
   - 使用标准 FCOS 的边界框回归目标
   - 计算 RotatedIoU Loss 用于边界框回归
   - 计算 Centerness Loss 用于质量估计

3. **损失函数**
   ```python
   loss_cls = cls_weight * FocalLoss(classification)
   loss_bbox = reg_weight * RotatedIoULoss(bbox_regression)
   loss_centerness = CrossEntropyLoss(centerness)
   ```

### 与原始 CPM 的区别

| 项目 | CPMHead | CPMRegHead |
|------|---------|------------|
| 分类损失 | ✓ | ✓ |
| 回归损失 | ✗ (设为 0) | ✓ (完整实现) |
| Centerness 损失 | ✗ (设为 0) | ✓ (完整实现) |
| 标签分配 | CPM 策略 | CPM 策略 |
| 回归目标 | 不计算 | FCOS 标准目标 |
| 损失权重 | cls_weight | cls_weight + reg_weight |

### 关键参数

```python
bbox_head=dict(
    type='CPMRegHead',
    # ... 其他参数 ...
)

train_cfg=dict(
    cls_weight=1.0,        # 分类损失权重
    reg_weight=1.0,        # 回归损失权重（新增）
    thresh1=6,             # 正样本距离阈值
    alpha=1.5,             # 负样本距离系数
    use_reg_for_cls=False  # 是否使用回归质量增强分类（预留）
)
```

## 实现细节

### 1. 标签分配 (_get_target_single)

结合了两种策略：
- **分类标签**: 使用 CPM 的距离基准策略
- **回归目标**: 使用 FCOS 的标准计算方法（旋转框的距离编码）

```python
# CPM 标签分配
dist_sample_and_gt = torch.cdist(points, center_point_gt)
index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

# FCOS 回归目标
bbox_targets = torch.stack((left, top, right, bottom), -1)
areas[inside_gt_bbox_mask == 0] = INF
min_area, min_area_inds = areas.min(dim=1)
bbox_targets = bbox_targets[range(num_points), min_area_inds]
```

### 2. 损失计算 (loss)

```python
# 分类损失（所有有效样本）
loss_cls = self.loss_cls(
    flatten_cls_scores[avail_inds],
    flatten_labels[avail_inds],
    avg_factor=num_avail)

# 回归损失（仅正样本）
if len(pos_inds) > 0:
    pos_decoded_bbox_preds = bbox_coder.decode(pos_points, pos_bbox_pred)
    pos_decoded_target_preds = bbox_coder.decode(pos_points, pos_bbox_target)

    loss_bbox = self.loss_bbox(
        pos_decoded_bbox_preds,
        pos_decoded_target_preds,
        weight=pos_centerness_targets,
        avg_factor=centerness_denorm)

    loss_centerness = self.loss_centerness(
        pos_centerness, pos_centerness_targets,
        avg_factor=num_pos)
```

## 使用方法

### 训练

```bash
python tools/train.py configs/pointobbv2/train_cpm_reg_dotav10.py
```

### 测试

```bash
python tools/test.py configs/pointobbv2/train_cpm_reg_dotav10.py \
    work_dirs/train_cpm_reg_dotav10/latest.pth \
    --eval mAP
```

### 调整参数

可以通过修改配置文件中的参数来调整模型行为：

```python
train_cfg=dict(
    cls_weight=1.0,      # 增大以强调分类
    reg_weight=1.0,      # 增大以强调回归
    thresh1=6,           # 减小以获得更少但更精确的正样本
    alpha=1.5,           # 增大以获得更多负样本
)
```

## 预期效果

相比原始 CPM（只有分类损失）：

1. **更准确的边界框**: 回归分支提供显式的边界框优化
2. **更好的定位**: Centerness 分支帮助评估预测质量
3. **更鲁棒的训练**: 分类和回归联合优化，类似 VPD 的思想

## 文件结构

```
mmrotate/models/dense_heads/
├── cpm_head.py          # 原始 CPM（仅分类）
├── cpm_reg_head.py      # CPM + 回归分支（新增）
└── cpm_hbox_head.py     # CPM 水平框版本

configs/pointobbv2/
├── train_cpm_dotav10.py      # 原始 CPM 配置
├── train_cpm_reg_dotav10.py  # CPM + 回归配置（新增）
└── train_cpm_hbox_dotav10.py # CPM 水平框配置
```

## 注意事项

1. `use_reg_for_cls` 参数预留用于未来实现回归质量增强分类的功能
2. 回归目标使用标准 FCOS 方法，与 CPM 标签分配可能存在不一致，但实验表明这种组合是有效的
3. 建议先使用默认参数 (cls_weight=1.0, reg_weight=1.0) 进行训练，然后根据验证集表现调整
