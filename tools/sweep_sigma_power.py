#!/usr/bin/env python3
"""Sweep sigma_power to find optimal value for pseudo-label quality.

Loads model once, generates pseudo-labels for N images per power value,
computes mIoU against GT.

Usage:
    python tools/sweep_sigma_power.py --device cuda:0 --n 300
    python tools/sweep_sigma_power.py --device cuda:1 --n 300 --powers 0.1 0.2 3.0
"""
import argparse, os, tempfile, shutil, warnings
import numpy as np
import cv2
from pathlib import Path

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/pointobbv2/gen_vpd_sigma_pca28.py')
    p.add_argument('--ckpt', default='work_dirs/frozen_vpd_v2/epoch_1.pth')
    p.add_argument('--powers', nargs='+', type=float,
                   default=[0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--gt-dir',
                   default='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles')
    p.add_argument('--img-prefix',
                   default='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images')
    return p.parse_args()


def load_obb(path):
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            boxes.append(np.array([float(x) for x in parts[:8]]).reshape(4, 2))
    return boxes


def box_iou(a, b):
    ra = cv2.minAreaRect(a.astype(np.float32))
    rb = cv2.minAreaRect(b.astype(np.float32))
    ret, region = cv2.rotatedRectangleIntersection(ra, rb)
    if ret == cv2.INTERSECT_NONE or region is None:
        return 0.0
    inter = cv2.contourArea(region)
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return inter / union if union > 1e-6 else 0.0


def main():
    args = parse_args()
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter
    import mmrotate  # noqa

    from mmrotate.models import build_detector

    cfg = Config.fromfile(args.cfg)
    # Disable random flip for deterministic evaluation
    for t in cfg.data.train.pipeline:
        if t['type'] == 'RRandomFlip':
            t['flip_ratio'] = [0.0, 0.0, 0.0]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    model = model.to(args.device).float().eval()

    # Use train pipeline (includes GT loading)
    train_pipeline = Compose(cfg.data.train.pipeline)

    # Pick N images
    all_imgs = sorted(os.listdir(args.img_prefix))
    np.random.seed(args.seed)
    idxs = np.random.choice(len(all_imgs), min(args.n, len(all_imgs)), replace=False)
    img_names = [all_imgs[i] for i in sorted(idxs)]

    # Pre-load GT
    gt_cache = {}
    for name in img_names:
        stem = Path(name).stem
        gt_cache[stem] = load_obb(os.path.join(args.gt_dir, stem + '.txt'))

    print(f'Sweeping sigma_power on {len(img_names)} images, device={args.device}')
    print(f'{"power":<8} {"mIoU":>8} {"F1@0.5":>8} {"matchIoU":>10} {"matched":>10}')
    print('-' * 50)

    for power in args.powers:
        model.bbox_head.sigma_power = power
        tmpdir = tempfile.mkdtemp()
        model.bbox_head.train_cfg.store_ann_dir = tmpdir + '/'

        all_best = []
        n_matched = 0
        matched_ious = []

        for img_name in img_names:
            stem = Path(img_name).stem
            img_path = os.path.join(args.img_prefix, img_name)
            ann_path = os.path.join(args.gt_dir, stem + '.txt')

            # Build data dict like DOTADataset
            data = dict(
                img_info=dict(filename=img_path),
                img_prefix=None,
                ann_info=dict(ann_file=ann_path),
                bbox_fields=[],
            )
            # Use the raw loading from file
            from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations
            loader = LoadImageFromFile()
            data = loader(data)

            # Load annotations manually
            gt_bboxes = []
            gt_labels = []
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    coords = [float(x) for x in parts[:8]]
                    pts = np.array(coords).reshape(4, 2)
                    # Convert to (cx, cy, w, h, angle) via minAreaRect
                    rect = cv2.minAreaRect(pts.astype(np.float32))
                    cx, cy = rect[0]
                    w, h = rect[1]
                    angle = rect[2]
                    # DOTA classes
                    cls_name = parts[8]
                    cls_map = {c: i for i, c in enumerate(
                        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                         'basketball-court', 'storage-tank', 'soccer-ball-field',
                         'roundabout', 'harbor', 'swimming-pool', 'helicopter'))}
                    if cls_name not in cls_map:
                        continue
                    gt_bboxes.append([cx, cy, w, h, angle * np.pi / 180])
                    gt_labels.append(cls_map[cls_name])

            if not gt_bboxes:
                continue

            data['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            data['gt_labels'] = np.array(gt_labels, dtype=np.int64)
            data['bbox_fields'] = ['gt_bboxes']

            # Apply remaining pipeline (Resize, Normalize, Pad, Format)
            from mmcv import Config as Cfg
            remaining = []
            for t_cfg in cfg.data.train.pipeline:
                if t_cfg['type'] in ('LoadImageFromFile', 'LoadAnnotations'):
                    continue
                remaining.append(t_cfg)
            pipe = Compose(remaining)
            data = pipe(data)

            # Collate and scatter
            data = collate([data], samples_per_gpu=1)
            dev = torch.device(args.device)
            data = scatter(data, [dev.index or 0])[0]

            with torch.no_grad():
                model.forward_train(**data)

            # Evaluate
            pred_path = os.path.join(tmpdir, stem + '.txt')
            preds = load_obb(pred_path)
            gts = gt_cache[stem]

            for g in gts:
                if preds:
                    best = max(box_iou(g, p) for p in preds)
                else:
                    best = 0.0
                all_best.append(best)
                if best >= 0.5:
                    n_matched += 1
                    matched_ious.append(best)

        miou = np.mean(all_best) if all_best else 0.0
        f1 = n_matched / len(all_best) if all_best else 0.0
        m_iou = np.mean(matched_ious) if matched_ious else 0.0

        print(f'{power:<8.2f} {miou:>8.4f} {f1:>8.4f} {m_iou:>10.4f} '
              f'{n_matched:>5}/{len(all_best):<5}')

        shutil.rmtree(tmpdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
