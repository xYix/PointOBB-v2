#!/usr/bin/env python
"""Visualization tools for PointOBB-v2 VPD training.

Subcommands:
    loss    -- Plot training loss curves from log.json files
    detect  -- Run model inference on images and draw predicted OBBs + GT
    cpm     -- CPM classification score heatmap (per-class colored)
    varmap  -- VPD posterior variance / uncertainty heatmap

Examples:
    # Plot losses from a single run
    python tools/visualize.py loss work_dirs/vpd_dotav10/20260401_234142.log.json

    # Compare two runs
    python tools/visualize.py loss run1.log.json run2.log.json --keys loss_center loss_kl

    # Visualize detections
    python tools/visualize.py detect \
        configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
        work_dirs/vpd_dotav10/epoch_2.pth \
        --images P0000__1024__0___0.png --score-thr 0.3

    # CPM classification heatmap
    python tools/visualize.py cpm \
        configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
        work_dirs/vpd_dotav10/epoch_2.pth \
        --images P0000__1024__0___0.png --score-thr 0.05

    # VPD variance map
    python tools/visualize.py varmap \
        configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
        work_dirs/vpd_dotav10/epoch_2.pth \
        --images P0000__1024__0___0.png --level 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


# ─── DOTA palette ────────────────────────────────────────────────────────────
CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter',
)

PALETTE = [
    (0, 127, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 255),
]


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _build_model(config_path, checkpoint_path, device='cuda:0'):
    """Build model with both train/test cfg, load checkpoint."""
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import mmrotate  # noqa: F401
    from mmrotate.models import build_detector

    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cfg = cfg
    model.CLASSES = cfg.get('classes', CLASSES)
    model.to(device)
    model.eval()
    return model, cfg


def _resolve_images(cfg, images_arg, num_images, seed):
    """Resolve image list from args or random sample."""
    img_prefix = cfg.data.train.img_prefix
    if images_arg:
        return images_arg, img_prefix
    all_imgs = sorted(os.listdir(img_prefix))
    np.random.seed(seed)
    idxs = np.random.choice(len(all_imgs), min(num_images, len(all_imgs)),
                            replace=False)
    return [all_imgs[i] for i in sorted(idxs)], img_prefix


def _prepare_input(model, img_path, device):
    """Build model input dict from an image path (test pipeline)."""
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter

    cfg = model.cfg
    test_pipeline = cfg.data.test.pipeline
    pipeline = Compose(test_pipeline)
    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # scatter needs list[int] for target_gpus
    if 'cuda' in str(device):
        dev = torch.device(device)
        data = scatter(data, [dev.index or 0])[0]
    return data


def _forward_features(model, data, device):
    """Run backbone + neck + head forward, return per-level raw outputs.

    Returns:
        cls_scores: list[Tensor], each (1, C, H, W) — raw logits
        bbox_preds: list[Tensor], each (1, 8, H, W) for VPD / (1, 4, H, W) for CPM
        centernesses: list[Tensor], each (1, 1, H, W)
    """
    with torch.no_grad():
        img = data['img'][0] if isinstance(data['img'], list) else data['img']
        x = model.extract_feat(img)
        outs = model.bbox_head(x)
    # outs is (cls_scores, bbox_preds, centernesses) for VPD
    # or (cls_scores, bbox_preds, angle_preds, centernesses) for original FCOS
    return outs


def load_gt_annotations(ann_dir, img_name):
    """Load DOTA-format GT annotations for a given image."""
    stem = Path(img_name).stem
    ann_path = os.path.join(ann_dir, stem + '.txt')
    gt_boxes = []
    if not os.path.isfile(ann_path):
        return gt_boxes
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = list(map(float, parts[:8]))
            cls_name = parts[8]
            gt_boxes.append((coords, cls_name))
    return gt_boxes


def draw_rotated_box(img, corners, color, thickness=2):
    pts = np.array(corners, dtype=np.float32).reshape(4, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_obb_from_cxcywha(img, cx, cy, w, h, angle, color, thickness=2):
    rect = ((float(cx), float(cy)), (float(w), float(h)),
            float(angle) * 180.0 / np.pi)
    box_pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(img, [box_pts], isClosed=True, color=color,
                  thickness=thickness)


def _overlay_heatmap(base_img, heatmap_hw, colormap=cv2.COLORMAP_JET,
                     alpha=0.5, vmin=None, vmax=None):
    """Overlay a single-channel heatmap on base_img (BGR).

    Args:
        base_img: (H, W, 3) uint8 BGR
        heatmap_hw: (h, w) float numpy
    Returns:
        blended: (H, W, 3) uint8 BGR
    """
    H, W = base_img.shape[:2]
    hm = heatmap_hw.astype(np.float32)
    if vmin is None:
        vmin = float(np.percentile(hm, 2))
    if vmax is None:
        vmax = float(np.percentile(hm, 98))
    vmax = max(vmax, vmin + 1e-6)
    hm = np.clip((hm - vmin) / (vmax - vmin), 0, 1)
    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    hm_resized = cv2.resize(hm_color, (W, H), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(base_img, 1 - alpha, hm_resized, alpha, 0)
    return blended


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: loss
# ═══════════════════════════════════════════════════════════════════════════════

def parse_log_json(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get('mode') == 'train':
                records.append(d)
    return records


def cmd_loss(args):
    log_files = args.log_files
    keys = args.keys
    smooth = args.smooth

    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 4 * len(keys)),
                             squeeze=False, sharex=True)

    for log_path in log_files:
        records = parse_log_json(log_path)
        if not records:
            print(f'[WARN] No training records in {log_path}')
            continue

        label = Path(log_path).stem
        iters_per_epoch = max(r['iter'] for r in records)
        global_iters = [(r['epoch'] - 1) * iters_per_epoch + r['iter']
                        for r in records]

        for ax_row, key in zip(axes, keys):
            ax = ax_row[0]
            vals = [r.get(key) for r in records]
            if all(v is None for v in vals):
                continue
            vals = [v if v is not None else float('nan') for v in vals]

            if smooth > 0:
                smoothed, ema = [], vals[0]
                for v in vals:
                    if np.isnan(v):
                        smoothed.append(float('nan'))
                    else:
                        ema = smooth * ema + (1 - smooth) * v
                        smoothed.append(ema)
                ax.plot(global_iters, smoothed, label=label, linewidth=1.5)
                ax.plot(global_iters, vals, alpha=0.15, linewidth=0.5,
                        color=ax.get_lines()[-1].get_color())
            else:
                ax.plot(global_iters, vals, label=label, linewidth=1)

    for ax_row, key in zip(axes, keys):
        ax = ax_row[0]
        ax.set_ylabel(key)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel('iteration')
    fig.suptitle('Training Loss Curves', fontsize=14)
    fig.tight_layout()

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: detect
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_detect(args):
    from mmdet.apis import inference_detector

    model, cfg = _build_model(args.config, args.checkpoint, args.device)
    img_names, img_prefix = _resolve_images(cfg, args.images,
                                            args.num_images, args.seed)
    ann_dir = cfg.data.train.ann_file
    os.makedirs(args.output_dir, exist_ok=True)

    for img_name in img_names:
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'[WARN] Image not found: {img_path}')
            continue

        img = cv2.imread(img_path)
        img_det = img.copy()
        img_gt = img.copy()

        result = inference_detector(model, img_path)
        det_count = 0
        for cls_id, dets in enumerate(result):
            if len(dets) == 0:
                continue
            color = PALETTE[cls_id % len(PALETTE)]
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            for det in dets:
                score = det[5] if len(det) > 5 else det[4]
                if score < args.score_thr:
                    continue
                cx, cy, w, h, angle = det[:5]
                draw_obb_from_cxcywha(img_det, cx, cy, w, h, angle,
                                      color, thickness=2)
                tx, ty = int(cx - w / 3), int(cy - h / 3)
                cv2.putText(img_det, f'{cls_name} {score:.2f}', (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                det_count += 1

        gt_boxes = load_gt_annotations(ann_dir, img_name)
        for coords, cls_name in gt_boxes:
            cls_id = CLASSES.index(cls_name) if cls_name in CLASSES else 0
            draw_rotated_box(img_gt, coords, PALETTE[cls_id % len(PALETTE)], 2)

        cv2.putText(img_gt, 'GT', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img_det, f'Det (n={det_count}, thr={args.score_thr})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 127, 255), 2)

        canvas = np.concatenate([img_gt, img_det], axis=1)
        stem = Path(img_name).stem
        out_path = os.path.join(args.output_dir, f'{stem}_vis.jpg')
        cv2.imwrite(out_path, canvas)
        print(f'[{det_count:3d} dets] {out_path}')

    print(f'\nDone. Results saved to {args.output_dir}/')


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: cpm  —  Per-class classification probability heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_cpm(args):
    model, cfg = _build_model(args.config, args.checkpoint, args.device)
    img_names, img_prefix = _resolve_images(cfg, args.images,
                                            args.num_images, args.seed)
    ann_dir = cfg.data.train.ann_file
    levels = args.levels
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve which classes to show
    num_classes = len(CLASSES)
    if args.classes:
        show_cls = []
        for c in args.classes:
            if c.isdigit():
                show_cls.append(int(c))
            elif c in CLASSES:
                show_cls.append(CLASSES.index(c))
            else:
                print(f'[WARN] Unknown class: {c}')
        if not show_cls:
            show_cls = list(range(num_classes))
    else:
        show_cls = list(range(num_classes))

    for img_name in img_names:
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'[WARN] Image not found: {img_path}')
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_bgr.shape[:2]
        data = _prepare_input(model, img_path, args.device)
        outs = _forward_features(model, data, args.device)
        cls_scores = outs[0]  # list[Tensor(1, C, H, W)]

        stem = Path(img_name).stem
        num_levels = len(cls_scores)
        use_levels = levels if levels else list(range(num_levels))

        # Load GT for annotating which classes are present
        gt_boxes = load_gt_annotations(ann_dir, img_name)
        gt_cls_set = set()
        gt_centers = []  # (cx, cy, cls_id)
        for coords, cls_name in gt_boxes:
            cls_id = CLASSES.index(cls_name) if cls_name in CLASSES else -1
            gt_cls_set.add(cls_id)
            pts = np.array(coords).reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            gt_centers.append((cx, cy, cls_id))

        for lvl in use_levels:
            if lvl >= num_levels:
                continue
            cls_score = cls_scores[lvl][0]  # (C, H, W)
            probs = cls_score.sigmoid().cpu().numpy()  # (C, H, W)
            H, W = probs.shape[1:]
            stride = model.bbox_head.strides[lvl]

            n_show = len(show_cls)
            ncols = min(n_show + 1, 6)  # +1 for max-prob panel
            nrows = (n_show + 1 + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4 * ncols, 4 * nrows),
                                     squeeze=False)

            img_small = cv2.resize(img_rgb, (W, H),
                                   interpolation=cv2.INTER_AREA)

            # Panel 0: max class probability
            ax = axes.flat[0]
            max_prob = probs[show_cls].max(axis=0)  # (H, W)
            ax.imshow(img_small, alpha=0.3)
            im = ax.imshow(max_prob, cmap='hot', alpha=0.7, vmin=0,
                           vmax=max(0.3, float(max_prob.max())))
            ax.set_title('max prob', fontsize=9, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Draw GT centers
            for cx, cy, cid in gt_centers:
                fx, fy = cx / img_w * W, cy / img_h * H
                ax.plot(fx, fy, 'g+', markersize=6, markeredgewidth=1.5)

            # Per-class panels
            for idx, cls_id in enumerate(show_cls):
                ax = axes.flat[idx + 1]
                prob_hw = probs[cls_id]  # (H, W)
                ax.imshow(img_small, alpha=0.3)
                vmax = max(0.1, float(np.percentile(prob_hw, 99.5)))
                im = ax.imshow(prob_hw, cmap='hot', alpha=0.7,
                               vmin=0, vmax=vmax)
                title = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                # Mark title green if this class has GT in this image
                color = 'green' if cls_id in gt_cls_set else 'black'
                ax.set_title(title, fontsize=8, color=color, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Draw GT centers of this class
                for cx, cy, cid in gt_centers:
                    if cid == cls_id:
                        fx, fy = cx / img_w * W, cy / img_h * H
                        ax.plot(fx, fy, 'g+', markersize=6,
                                markeredgewidth=1.5)

            # Hide unused panels
            for idx in range(n_show + 1, nrows * ncols):
                axes.flat[idx].axis('off')

            fig.suptitle(f'{stem}  Level {lvl} (stride={stride}, {H}x{W})',
                         fontsize=12)
            fig.tight_layout()
            out_path = os.path.join(args.output_dir,
                                    f'{stem}_cpm_L{lvl}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        print(f'[cpm] {stem}: {len(use_levels)} levels, '
              f'{len(show_cls)} classes')

    print(f'\nDone. Results saved to {args.output_dir}/')


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: varmap  —  VPD variance / uncertainty heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_varmap(args):
    model, cfg = _build_model(args.config, args.checkpoint, args.device)
    img_names, img_prefix = _resolve_images(cfg, args.images,
                                            args.num_images, args.seed)
    ann_dir = cfg.data.train.ann_file
    levels = args.levels
    os.makedirs(args.output_dir, exist_ok=True)

    # Verify this is a VPD head (8-channel bbox_pred)
    is_vpd = hasattr(model.bbox_head, 'loss_vpd')
    if not is_vpd:
        print('[ERROR] varmap requires a CPMVPDHead (8-ch bbox_pred).')
        sys.exit(1)

    for img_name in img_names:
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'[WARN] Image not found: {img_path}')
            continue

        img_bgr = cv2.imread(img_path)
        img_h, img_w = img_bgr.shape[:2]
        data = _prepare_input(model, img_path, args.device)
        outs = _forward_features(model, data, args.device)

        cls_scores = outs[0]   # list[(1, C, H, W)]
        bbox_preds = outs[1]   # list[(1, 8, H, W)]
        num_levels = len(bbox_preds)
        use_levels = levels if levels else list(range(num_levels))

        stem = Path(img_name).stem

        # Channel semantics for VPD:
        #   bbox_pred[:, 0:4] = mu  (delta_x, delta_y, log_w, log_h)
        #   bbox_pred[:, 4:8] = log_sigma (log_sx, log_sy, log_sw, log_sh)
        channel_names = ['sigma_dx', 'sigma_dy', 'sigma_w', 'sigma_h']
        mu_names = ['mu_dx', 'mu_dy', 'mu_logw', 'mu_logh']

        for lvl in use_levels:
            if lvl >= num_levels:
                continue
            bp = bbox_preds[lvl][0]     # (8, H, W)
            cs = cls_scores[lvl][0]     # (C, H, W)
            H, W = bp.shape[1:]
            stride = model.bbox_head.strides[lvl]

            log_sigma = bp[4:8].cpu().numpy()   # (4, H, W)
            sigma = np.exp(log_sigma)           # (4, H, W)
            mu = bp[0:4].cpu().numpy()          # (4, H, W)
            max_prob = cs.sigmoid().max(dim=0)[0].cpu().numpy()  # (H, W)

            # --- Figure: 2 rows x 4 cols ---
            # Row 1: mu channels (delta_x, delta_y, log_w, log_h)
            # Row 2: sigma channels (uncertainty)
            # + 1 col for cls_score + 1 col for total uncertainty
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            fig.suptitle(f'{stem}  Level {lvl} (stride={stride}, {H}x{W})',
                         fontsize=14)

            img_small = cv2.resize(img_bgr, (W, H),
                                   interpolation=cv2.INTER_AREA)
            img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

            # Row 0: mu channels
            for ch in range(4):
                ax = axes[0, ch]
                ax.imshow(img_small_rgb, alpha=0.3)
                im = ax.imshow(mu[ch], cmap='RdBu_r', alpha=0.7,
                               vmin=np.percentile(mu[ch], 2),
                               vmax=np.percentile(mu[ch], 98))
                ax.set_title(mu_names[ch], fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row 0, col 4: cls_score heatmap
            ax = axes[0, 4]
            ax.imshow(img_small_rgb, alpha=0.3)
            im = ax.imshow(max_prob, cmap='hot', alpha=0.7, vmin=0, vmax=1)
            ax.set_title('max cls_prob', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row 1: sigma channels
            for ch in range(4):
                ax = axes[1, ch]
                ax.imshow(img_small_rgb, alpha=0.3)
                im = ax.imshow(sigma[ch], cmap='magma', alpha=0.7,
                               vmin=np.percentile(sigma[ch], 2),
                               vmax=np.percentile(sigma[ch], 98))
                ax.set_title(channel_names[ch], fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row 1, col 4: total uncertainty = mean(sigma across 4 dims)
            ax = axes[1, 4]
            total_unc = sigma.mean(axis=0)  # (H, W)
            ax.imshow(img_small_rgb, alpha=0.3)
            im = ax.imshow(total_unc, cmap='magma', alpha=0.7,
                           vmin=np.percentile(total_unc, 2),
                           vmax=np.percentile(total_unc, 98))
            ax.set_title('total uncertainty', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            out_path = os.path.join(args.output_dir,
                                    f'{stem}_varmap_L{lvl}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            # --- Also save a single clean overlay for quick viewing ---
            # center uncertainty = sqrt(sigma_dx^2 + sigma_dy^2)
            center_unc = np.sqrt(sigma[0] ** 2 + sigma[1] ** 2)
            overlay = _overlay_heatmap(img_bgr, center_unc,
                                       colormap=cv2.COLORMAP_MAGMA, alpha=0.5)
            cv2.putText(overlay,
                        f'Center uncertainty  L{lvl} (stride={stride})',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            # Draw GT center points
            gt_boxes = load_gt_annotations(ann_dir, img_name)
            for coords, cls_name in gt_boxes:
                pts = np.array(coords).reshape(4, 2)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)

            overlay_path = os.path.join(args.output_dir,
                                        f'{stem}_center_unc_L{lvl}.jpg')
            cv2.imwrite(overlay_path, overlay)

        print(f'[varmap] {stem}: {len(use_levels)} levels')

    print(f'\nDone. Results saved to {args.output_dir}/')


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def _add_common_model_args(parser):
    """Add config/checkpoint/images/device args shared by model-based cmds."""
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint (.pth) path')
    parser.add_argument('--images', nargs='*', default=None,
                        help='Specific image filenames')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of random images (if --images not given)')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--output-dir', '-o', default='vis_output',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


def main():
    parser = argparse.ArgumentParser(
        description='PointOBB-v2 VPD Visualization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest='command')

    # --- loss ---
    p_loss = sub.add_parser('loss', help='Plot training loss curves')
    p_loss.add_argument('log_files', nargs='+', help='log.json file(s)')
    p_loss.add_argument('--keys', nargs='+',
                        default=['loss_center', 'loss_kl', 'loss', 'grad_norm'],
                        help='Loss keys to plot')
    p_loss.add_argument('--smooth', type=float, default=0.9,
                        help='EMA smoothing factor (0=none, 0.9=default)')
    p_loss.add_argument('--output', '-o', default='loss_curves.png',
                        help='Output image path')

    # --- detect ---
    p_det = sub.add_parser('detect', help='Visualize model detections')
    _add_common_model_args(p_det)
    p_det.add_argument('--score-thr', type=float, default=0.3,
                       help='Score threshold for detections')

    # --- cpm ---
    p_cpm = sub.add_parser('cpm',
                           help='Per-class classification probability heatmap')
    _add_common_model_args(p_cpm)
    p_cpm.add_argument('--levels', type=int, nargs='*', default=None,
                       help='FPN levels to visualize (default: all)')
    p_cpm.add_argument('--classes', nargs='*', default=None,
                       help='Classes to show (name or index, default: all)')

    # --- varmap ---
    p_var = sub.add_parser('varmap',
                           help='VPD posterior variance / uncertainty map')
    _add_common_model_args(p_var)
    p_var.add_argument('--levels', type=int, nargs='*', default=None,
                       help='FPN levels to visualize (default: all)')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmds = {
        'loss': cmd_loss,
        'detect': cmd_detect,
        'cpm': cmd_cpm,
        'varmap': cmd_varmap,
    }
    cmds[args.command](args)


if __name__ == '__main__':
    main()
