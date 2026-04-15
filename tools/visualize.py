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
    python tools/visualize.py loss run1.log.json run2.log.json --keys loss_mu_dense loss_sigma_dense

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
        bbox_preds: list[Tensor], each (1, 4, H, W) — [mu_dx, mu_dy, log_sigma_dx, log_sigma_dy]
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
#  Subcommand: cpm  —  CPM classification score heatmap
#  Style: matches cpm_head.draw_image — PIL-based, image|mask side-by-side
# ═══════════════════════════════════════════════════════════════════════════════

CPM_PALETTE = [
    (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
    (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
    (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
    (0, 0, 255), (147, 116, 116), (0, 0, 255),
]


def _get_cls_mask(max_probs, max_indices, thr, H, W):
    """Build class-colored mask: pixel = class color if prob > thr, else transparent (black with alpha=0)."""
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.float32)
    above = max_probs > thr
    for i in range(H):
        for j in range(W):
            if above[i, j]:
                mask[i, j] = CPM_PALETTE[int(max_indices[i, j]) % len(CPM_PALETTE)]
                alpha[i, j] = min(float(max_probs[i, j]), 1.0)
    return mask, alpha


def cmd_cpm(args):
    from PIL import Image

    model, cfg = _build_model(args.config, args.checkpoint, args.device)
    img_names, img_prefix = _resolve_images(cfg, args.images,
                                            args.num_images, args.seed)
    ann_dir = cfg.data.train.ann_file
    levels = args.levels
    thresholds = args.thresholds
    os.makedirs(args.output_dir, exist_ok=True)

    for img_name in img_names:
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'[WARN] Image not found: {img_path}')
            continue

        data = _prepare_input(model, img_path, args.device)
        outs = _forward_features(model, data, args.device)
        cls_scores = outs[0]  # list[Tensor(1, C, H, W)]

        stem = Path(img_name).stem
        num_levels = len(cls_scores)
        use_levels = levels if levels else list(range(num_levels))

        for lvl in use_levels:
            if lvl >= num_levels:
                continue
            score_probs = cls_scores[lvl][0].sigmoid()  # (C, H, W)
            H, W = score_probs.shape[1:]
            stride = model.bbox_head.strides[lvl]

            max_probs, max_indices = score_probs.max(dim=0)  # (H, W)
            max_probs = max_probs.cpu().numpy()
            max_indices = max_indices.cpu().numpy()

            img_pil = Image.open(img_path).convert('RGB')
            img_small = img_pil.resize((W, H))

            out_dir = os.path.join(args.output_dir,
                                   f'{stem}_L{lvl}_s{stride}')
            os.makedirs(out_dir, exist_ok=True)

            for thr in thresholds:
                mask, alpha = _get_cls_mask(max_probs, max_indices, thr, H, W)
                # Blend: overlay class mask on image, alpha modulated by prob
                img_np = np.array(img_small).astype(np.float32)
                mask_f = mask.astype(np.float32)
                a = (alpha * 0.6)[..., None]  # overlay strength
                blended = img_np * (1.0 - a) + mask_f * a
                blended = blended.astype(np.uint8)
                Image.fromarray(blended).save(
                    os.path.join(out_dir, f'{thr}.jpg'))

            # --- Per-class probability heatmaps ---
            probs_np = score_probs.cpu().numpy()  # (C, H, W)
            img_small_np = np.array(img_small)    # (H, W, 3) RGB

            for cls_id in range(probs_np.shape[0]):
                prob_hw = probs_np[cls_id]    # (H, W)
                pmax = max(prob_hw.max(), 1e-6)
                # Normalize to [0, 255] and apply JET colormap
                norm = (np.clip(prob_hw / pmax, 0, 1) * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # BGR
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                # Blend: 40% image + 60% heatmap
                blended = (img_small_np * 0.4 + heatmap_rgb * 0.6).astype(np.uint8)
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                out_img = Image.fromarray(blended)
                out_img.save(os.path.join(out_dir, f'cls_{cls_name}.jpg'))

        print(f'[cpm] {stem}: {len(use_levels)} levels x '
              f'{len(thresholds)} thrs + {probs_np.shape[0]} class maps')

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

    # Verify this is a VPD head (has conv_sigma)
    is_vpd = hasattr(model.bbox_head, 'conv_sigma')
    if not is_vpd:
        print('[ERROR] varmap requires a CPMVPDHead (conv_sigma).')
        sys.exit(1)

    for img_name in img_names:
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'[WARN] Image not found: {img_path}')
            continue

        img_bgr = cv2.imread(img_path)
        data = _prepare_input(model, img_path, args.device)
        outs = _forward_features(model, data, args.device)

        cls_scores = outs[0]   # list[(1, C, H, W)]
        bbox_preds = outs[1]   # list[(1, 4, H, W)]
        num_levels = len(bbox_preds)
        use_levels = levels if levels else list(range(num_levels))

        stem = Path(img_name).stem

        # Channel semantics (4-ch, center-only, no w/h):
        #   bbox_pred[:, 0:2] = mu  (delta_x, delta_y)
        #   bbox_pred[:, 2:4] = log_sigma (log_sx, log_sy)

        for lvl in use_levels:
            if lvl >= num_levels:
                continue
            bp = bbox_preds[lvl][0]     # (4, H, W)
            cs = cls_scores[lvl][0]     # (C, H, W)
            H, W = bp.shape[1:]
            stride = model.bbox_head.strides[lvl]

            mu = bp[0:2].cpu().numpy()              # (2, H, W)
            log_sigma = bp[2:4].cpu().numpy()       # (2, H, W)
            sigma = np.exp(log_sigma)               # (2, H, W)
            max_prob = cs.sigmoid().max(dim=0)[0].cpu().numpy()  # (H, W)

            mu_names = ['mu_dx', 'mu_dy']
            sigma_names = ['sigma_dx', 'sigma_dy']

            # --- Figure: 2 rows x 3 cols ---
            # Row 0: mu_dx, mu_dy, max_cls_prob
            # Row 1: sigma_dx, sigma_dy, center_uncertainty
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{stem}  Level {lvl} (stride={stride}, {H}x{W})',
                         fontsize=14)

            img_small = cv2.resize(img_bgr, (W, H),
                                   interpolation=cv2.INTER_AREA)
            img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            # Row 0: mu channels
            for ch in range(2):
                ax = axes[0, ch]
                ax.imshow(img_small_rgb, alpha=0.3)
                im = ax.imshow(mu[ch], cmap='RdBu_r', alpha=0.7,
                               vmin=np.percentile(mu[ch], 2),
                               vmax=np.percentile(mu[ch], 98))
                ax.set_title(mu_names[ch], fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row 0, col 2: cls_score heatmap
            ax = axes[0, 2]
            ax.imshow(img_small_rgb, alpha=0.3)
            im = ax.imshow(max_prob, cmap='hot', alpha=0.7, vmin=0, vmax=1)
            ax.set_title('max cls_prob', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row 1: sigma channels
            for ch in range(2):
                ax = axes[1, ch]
                ax.imshow(img_small_rgb, alpha=0.3)
                # im = ax.imshow(sigma[ch], cmap='magma', alpha=0.7,
                im = ax.imshow(sigma[ch], cmap='RdBu_r', alpha=0.7,
                               vmin=np.percentile(sigma[ch], 2),
                               vmax=np.percentile(sigma[ch], 98))
                ax.set_title(sigma_names[ch], fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row 1, col 2: center uncertainty = sqrt(sigma_dx^2 + sigma_dy^2)
            ax = axes[1, 2]
            center_unc = np.sqrt(sigma[0] ** 2 + sigma[1] ** 2)
            ax.imshow(img_small_rgb, alpha=0.3)
            # im = ax.imshow(center_unc, cmap='magma', alpha=0.7,
            im = ax.imshow(center_unc, cmap='RdBu_r', alpha=0.7,
                           vmin=np.percentile(center_unc, 2),
                           vmax=np.percentile(center_unc, 98))
            ax.set_title('center uncertainty', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            out_path = os.path.join(args.output_dir,
                                    f'{stem}_varmap_L{lvl}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            # --- Clean center uncertainty heatmap (no background) ---
            fig_clean, ax_clean = plt.subplots(1, 1, figsize=(8, 8))
            im_clean = ax_clean.imshow(center_unc, cmap='jet',
                                       vmin=np.percentile(center_unc, 2),
                                       vmax=max(np.percentile(center_unc, 98),
                                                np.percentile(center_unc, 2) + 1e-6))
            ax_clean.axis('off')
            plt.colorbar(im_clean, ax=ax_clean, fraction=0.046, pad=0.04)
            fig_clean.tight_layout()
            clean_path = os.path.join(args.output_dir,
                                      f'{stem}_center_unc_clean_L{lvl}.png')
            fig_clean.savefig(clean_path, dpi=150, bbox_inches='tight')
            plt.close(fig_clean)

            # --- Also save a single clean overlay for quick viewing ---
            # overlay = _overlay_heatmap(img_bgr, center_unc,
            #                            colormap=cv2.COLORMAP_MAGMA, alpha=0.5)
            overlay = _overlay_heatmap(img_bgr, center_unc,
                                       colormap=cv2.COLORMAP_JET, alpha=0.5)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: gtvis  —  GT annotation visualization (OBB + center point)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_gtvis(args):
    """Draw GT oriented bounding boxes and center points on images."""
    from mmcv import Config

    cfg = Config.fromfile(args.config)
    img_prefix = cfg.data.train.img_prefix
    ann_dir = cfg.data.train.ann_file

    if args.images:
        img_names = args.images
    else:
        import random
        random.seed(args.seed)
        all_imgs = sorted(os.listdir(img_prefix))
        img_names = random.sample(all_imgs,
                                  min(args.num_images, len(all_imgs)))

    os.makedirs(args.output_dir, exist_ok=True)

    for img_name in img_names:
        img_path = os.path.join(img_prefix, img_name)
        if not os.path.isfile(img_path):
            print(f'[WARN] Image not found: {img_path}')
            continue

        img = cv2.imread(img_path)
        gt_boxes = load_gt_annotations(ann_dir, img_name)

        cls_to_color = {}
        for coords, cls_name in gt_boxes:
            if cls_name not in cls_to_color:
                idx = CLASSES.index(cls_name) if cls_name in CLASSES else len(cls_to_color)
                cls_to_color[cls_name] = PALETTE[idx % len(PALETTE)]

            color = cls_to_color[cls_name]
            pts = np.array(coords, dtype=np.float32).reshape(4, 2)

            # Draw OBB
            pts_int = pts.astype(np.int32)
            cv2.polylines(img, [pts_int], isClosed=True,
                          color=color, thickness=2)

            # Draw center point
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.circle(img, (cx, cy), 4, color, -1)

        # Legend
        y0 = 20
        for cls_name, color in cls_to_color.items():
            cv2.putText(img, cls_name, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y0 += 18

        stem = Path(img_name).stem
        out_path = os.path.join(args.output_dir, f'{stem}_gt.jpg')
        cv2.imwrite(out_path, img)

    print(f'[gtvis] {len(img_names)} images → {args.output_dir}/')


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
                        default=['loss_mu_dense', 'loss_sigma_dense',
                                 'loss_cls', 'loss', 'grad_norm'],
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
                           help='CPM classification score heatmap')
    _add_common_model_args(p_cpm)
    p_cpm.add_argument('--levels', type=int, nargs='*', default=None,
                       help='FPN levels to visualize (default: all)')
    p_cpm.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
                       help='Score thresholds for class mask')

    # --- varmap ---
    p_var = sub.add_parser('varmap',
                           help='VPD posterior variance / uncertainty map')
    _add_common_model_args(p_var)
    p_var.add_argument('--levels', type=int, nargs='*', default=None,
                       help='FPN levels to visualize (default: all)')

    # --- gtvis ---
    p_gt = sub.add_parser('gtvis',
                          help='GT annotation visualization (OBB + center)')
    p_gt.add_argument('config', help='Config file path (for data paths)')
    p_gt.add_argument('--images', nargs='*', default=None,
                      help='Specific image filenames')
    p_gt.add_argument('--num-images', type=int, default=10,
                      help='Number of random images (if --images not given)')
    p_gt.add_argument('--output-dir', '-o', default='vis_output',
                      help='Output directory')
    p_gt.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmds = {
        'loss': cmd_loss,
        'detect': cmd_detect,
        'cpm': cmd_cpm,
        'varmap': cmd_varmap,
        'gtvis': cmd_gtvis,
    }
    cmds[args.command](args)


if __name__ == '__main__':
    main()
