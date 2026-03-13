# CPM Hbox 配置说明

## 概述

本配置将 CPM 模型从输出旋转边界框（Rbox）改为输出水平边界框（Hbox）。

## 主要变更

### 1. 新增文件

- **模型文件**: `mmrotate/models/dense_heads/cpm_hbox_head.py`
  - 实现了 `CPMHboxHead` 类
  - 继承自 mmdet 的 `FCOSHead`
  - 保留 CPM 的标签分配策略
  - 输出水平边界框而非旋转边界框

- **配置文件**: `configs/pointobbv2/train_cpm_hbox_dotav10.py`
  - 基于 `train_cpm_dotav10.py` 修改
  - 使用标准 FCOS 检测器
  - 使用 CPMHboxHead

### 2. 关键区别

| 项目 | 原配置 (Rbox) | 新配置 (Hbox) |
|------|--------------|--------------|
| 模型类型 | RotatedFCOS | FCOS |
| Head 类型 | CPMHead | CPMHboxHead |
| 输出格式 | (x, y, w, h, angle) | (x, y, w, h) |
| Bbox Coder | DistanceAnglePointCoder | DistancePointBBoxCoder |
| Loss | RotatedIoULoss | IoULoss |
| NMS | multiclass_nms_rotated | standard NMS |

### 3. CPM 标签分配策略

两个配置都使用相同的 CPM 标签分配策略：

- **正样本**: 距离 GT 中心点小于 `min(dist_to_nearest_gt/2, thresh1)` 的点
- **负样本**:
  - 距离所有 GT 中心点都大于 `alpha * dist_to_nearest_gt` 的点
  - 位于同类别实例中点附近的点（避免混淆）
- **忽略样本**: 其他点

参数：
- `cls_weight`: 分类损失权重（默认 1.0）
- `thresh1`: 正样本距离阈值（默认 6）
- `alpha`: 负样本距离系数（默认 1.5）

## 使用方法

### 训练

```bash
python tools/train.py configs/pointobbv2/train_cpm_hbox_dotav10.py
```

### 测试

```bash
python tools/test.py configs/pointobbv2/train_cpm_hbox_dotav10.py \
    work_dirs/train_cpm_hbox_dotav10/latest.pth \
    --eval mAP
```

## 注意事项

1. 水平边界框可能不适合高度旋转的目标（如船只、飞机等）
2. 评估指标仍使用 mAP，但计算方式基于水平框的 IoU
3. 输出目录默认为 `/mnt/sdb/xuyun/PointOBB-v2/exps/exp1/cpm_hbox_dotav10/`
