#!/usr/bin/env python3
"""Visualize sigma uncertainty maps — each panel saved as a separate file.

Per-image output (in {out-dir}/{stem}/):
  image.png       — Original image + GT center points (red +)
  sigma.png       — Combined uncertainty σ + GT center points (white +)
  sigma_dx.png    — σ_dx (horizontal uncertainty)
  sigma_dy.png    — σ_dy (vertical uncertainty)
  cls_prob.png    — Max classification probability (baseline signal)

Usage:
    python tools/vis_sigma.py
    python tools/vis_sigma.py --images P0001__1024__228___2472.png
    python tools/vis_sigma.py --n 10 --seed 42
"""
import argparse
import os
import warnings

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='work_dirs/frozen_vpd_v2/epoch_1.pth')
    p.add_argument('--cfg', default='configs/pointobbv2/generate_vpd_pseudo_label_dotav10.py')
    p.add_argument('--images', nargs='*', default=None)
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='visualization/sigma_maps')
    p.add_argument('--gt-dir', default='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def load_gt_centers(ann_path):
    centers = []
    if not os.path.exists(ann_path):
        return centers
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            pts = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            centers.append((cx, cy))
    return centers


def build_model(cfg_path, ckpt_path, device):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import mmrotate  # noqa: F401
    from mmrotate.models import build_detector

    cfg = Config.fromfile(cfg_path)
    for t in cfg.data.train.pipeline:
        if t['type'] == 'RRandomFlip':
            t['flip_ratio'] = [0.0, 0.0, 0.0]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model = model.to(device).float()
    model.eval()
    return model, cfg


def get_sigma_and_cls(model, img_path, device, level=0):
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter

    cfg = model.cfg
    pipeline = Compose(cfg.data.test.pipeline)
    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if 'cuda' in str(device):
        dev = torch.device(device)
        data = scatter(data, [dev.index or 0])[0]
        img = data['img'][0] if isinstance(data['img'], list) else data['img']
    else:
        # CPU: manually unpack DataContainer
        img_dc = data['img']
        if isinstance(img_dc, list):
            img_dc = img_dc[0]
        if hasattr(img_dc, 'data'):
            img_tensor = img_dc.data
            if isinstance(img_tensor, list):
                img_tensor = img_tensor[0]
            if img_tensor.dim() == 3:
                img = img_tensor.unsqueeze(0)
            else:
                img = img_tensor
        else:
            img = img_dc

    with torch.no_grad():
        x = model.extract_feat(img)
        outs = model.bbox_head(x)

    stride = model.bbox_head.strides[level]
    log_sigma = model.bbox_head._sigma_per_level[stride][0]
    sigma = log_sigma.exp().cpu().numpy()

    cls_scores = outs[0][level][0].sigmoid()
    max_cls = cls_scores.max(dim=0)[0].cpu().numpy()

    return sigma, max_cls, stride


def _save(img_bg, overlay, centers, cmap, title, path, dpi,
          vmin=None, vmax=None, cbar_label=None,
          center_color='w', center_size=4, bg_alpha=0.25, overlay_alpha=0.75):
    H, W = overlay.shape
    fig, ax = plt.subplots(1, 1, figsize=(W / dpi * 2.5, H / dpi * 2.5))
    ax.imshow(img_bg, alpha=bg_alpha)
    im = ax.imshow(overlay, cmap=cmap, alpha=overlay_alpha, vmin=vmin, vmax=vmax)
    for cx, cy in centers:
        ax.plot(cx, cy, '.', color=center_color, markersize=1.0)
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def _save_image(img_rgb, centers, title, path, dpi):
    H, W = img_rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(W / dpi * 2.5, H / dpi * 2.5))
    ax.imshow(img_rgb)
    for cx, cy in centers:
        ax.plot(cx, cy, '.', color='r', markersize=1.0)
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    img_prefix = '/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images'

    if args.images:
        img_names = args.images
    else:
        all_imgs = sorted(os.listdir(img_prefix))
        np.random.seed(args.seed)
        idxs = np.random.choice(len(all_imgs), min(args.n, len(all_imgs)),
                                replace=False)
        img_names = [all_imgs[i] for i in sorted(idxs)]

    print(f'Building model from {args.ckpt}...')
    model, cfg = build_model(args.cfg, args.ckpt, args.device)
    model.cfg = cfg

    for img_name in img_names:
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'  [SKIP] {img_name} not found')
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_img, W_img = img_bgr.shape[:2]

        sigma, max_cls, stride = get_sigma_and_cls(
            model, img_path, args.device, args.level)
        H_feat, W_feat = sigma.shape[1], sigma.shape[2]
        center_unc = np.sqrt(sigma[0] ** 2 + sigma[1] ** 2)

        gt_centers = load_gt_centers(os.path.join(args.gt_dir, stem + '.txt'))
        sx, sy = W_feat / W_img, H_feat / H_img
        centers_feat = [(cx * sx, cy * sy) for cx, cy in gt_centers]

        img_small = cv2.resize(img_rgb, (W_feat, H_feat),
                               interpolation=cv2.INTER_AREA)

        img_dir = os.path.join(args.out_dir, stem)
        os.makedirs(img_dir, exist_ok=True)
        dpi = args.dpi

        # 1. Image + GT center points
        _save_image(img_small, centers_feat, 'Image + point annotations',
                    os.path.join(img_dir, 'image.png'), dpi)

        # 2. Combined uncertainty sigma (high=blue, low=red)
        v0, v1 = np.percentile(center_unc, 2), np.percentile(center_unc, 98)
        _save(img_small, center_unc, centers_feat, 'RdYlBu',
              'Uncertainty  $\\sigma = \\sqrt{\\sigma_{dx}^2 + \\sigma_{dy}^2}$',
              os.path.join(img_dir, 'sigma.png'), dpi,
              vmin=v0, vmax=v1, cbar_label='$\\sigma$')

        # 3. Max classification probability (keep RdYlBu_r: high=red)
        _save(img_small, max_cls, centers_feat, 'RdYlBu_r',
              'Max classification probability (CPM baseline)',
              os.path.join(img_dir, 'cls_prob.png'), dpi,
              vmin=0, vmax=0.5, cbar_label='$P(class)$',
              center_color='w')

        # 4. sigma_dx (high=blue, low=red)
        v0, v1 = np.percentile(sigma[0], 2), np.percentile(sigma[0], 98)
        _save(img_small, sigma[0], centers_feat, 'RdYlBu',
              '$\\sigma_{dx}$ (horizontal uncertainty)',
              os.path.join(img_dir, 'sigma_dx.png'), dpi,
              vmin=v0, vmax=v1, cbar_label='$\\sigma_{dx}$')

        # 5. sigma_dy (high=blue, low=red)
        v0, v1 = np.percentile(sigma[1], 2), np.percentile(sigma[1], 98)
        _save(img_small, sigma[1], centers_feat, 'RdYlBu',
              '$\\sigma_{dy}$ (vertical uncertainty)',
              os.path.join(img_dir, 'sigma_dy.png'), dpi,
              vmin=v0, vmax=v1, cbar_label='$\\sigma_{dy}$')

        # 6. Save raw numpy data for downstream combination scripts
        np.savez_compressed(os.path.join(img_dir, 'raw_data.npz'),
                            img_small=img_small,
                            sigma=sigma,
                            center_unc=center_unc,
                            max_cls=max_cls,
                            gt_centers_feat=np.array(centers_feat))

        print(f'  {stem}: 5 files + raw_data.npz -> {img_dir}/')

    print(f'\nAll results saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
