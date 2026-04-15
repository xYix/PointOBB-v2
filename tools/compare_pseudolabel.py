#!/usr/bin/env python
"""Compare pseudo-labels with ground-truth annotations.

Draws GT (green) and pseudo-labels (red) side-by-side or overlaid on images.
Also computes per-image and aggregate matching statistics.

Usage:
    # Visual comparison for specific images
    python tools/compare_pseudolabel.py \
        /mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles \
        /mnt/tmp/datasets/DOTAv10pseudolabel \
        /mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images \
        --images P0001__1024__228___2472.png P0002__1024__0___0.png \
        -o vis_output/pseudo_compare

    # Aggregate stats over all images
    python tools/compare_pseudolabel.py \
        /mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles \
        /mnt/tmp/datasets/DOTAv10pseudolabel \
        /mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images \
        --stats-only
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter',
)

GT_COLOR = (0, 255, 0)       # green
PSEUDO_COLOR = (0, 0, 255)   # red (BGR)
CENTER_RADIUS = 4


def load_annotations(ann_path):
    """Load DOTA-format annotations. Returns list of (coords_8, class_name)."""
    boxes = []
    if not os.path.isfile(ann_path):
        return boxes
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = list(map(float, parts[:8]))
            cls_name = parts[8]
            boxes.append((coords, cls_name))
    return boxes


def box_center(coords):
    pts = np.array(coords).reshape(4, 2)
    return pts.mean(axis=0)


def box_iou_obb(coords_a, coords_b):
    """Approximate OBB IoU via rotated rectangle intersection (cv2)."""
    pts_a = np.array(coords_a, dtype=np.float32).reshape(4, 2)
    pts_b = np.array(coords_b, dtype=np.float32).reshape(4, 2)
    rect_a = cv2.minAreaRect(pts_a)
    rect_b = cv2.minAreaRect(pts_b)
    ret, region = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if ret == cv2.INTERSECT_NONE or region is None:
        return 0.0
    inter = cv2.contourArea(region)
    area_a = cv2.contourArea(pts_a)
    area_b = cv2.contourArea(pts_b)
    union = area_a + area_b - inter
    if union < 1e-6:
        return 0.0
    return inter / union


def match_boxes(gt_boxes, pred_boxes, iou_thr=0.5):
    """Match pred to GT by IoU. Returns (matched_pairs, unmatched_gt, unmatched_pred)."""
    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, (gc, _) in enumerate(gt_boxes):
        for j, (pc, _) in enumerate(pred_boxes):
            iou_matrix[i, j] = box_iou_obb(gc, pc)

    matched = []
    used_gt = set()
    used_pred = set()

    # Greedy matching: highest IoU first
    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]
        if max_iou < iou_thr:
            break
        gi, pi = max_idx
        matched.append((gi, pi, max_iou))
        used_gt.add(gi)
        used_pred.add(pi)
        iou_matrix[gi, :] = -1
        iou_matrix[:, pi] = -1

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return matched, unmatched_gt, unmatched_pred


def draw_comparison(img, gt_boxes, pred_boxes, matched, unmatched_gt, unmatched_pred):
    """Draw GT (green) and pseudo (red) boxes with match lines."""
    vis = img.copy()

    # Draw all GT boxes in green
    for coords, cls_name in gt_boxes:
        pts = np.array(coords, dtype=np.float32).reshape(4, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, GT_COLOR, 2)
        cx, cy = box_center(coords).astype(int)
        cv2.circle(vis, (cx, cy), CENTER_RADIUS, GT_COLOR, -1)

    # Draw all pseudo boxes in red
    for coords, cls_name in pred_boxes:
        pts = np.array(coords, dtype=np.float32).reshape(4, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, PSEUDO_COLOR, 2)
        cx, cy = box_center(coords).astype(int)
        cv2.circle(vis, (cx, cy), CENTER_RADIUS, PSEUDO_COLOR, -1)

    # Draw match lines (yellow)
    for gi, pi, iou in matched:
        gc = box_center(gt_boxes[gi][0]).astype(int)
        pc = box_center(pred_boxes[pi][0]).astype(int)
        cv2.line(vis, tuple(gc), tuple(pc), (0, 255, 255), 1)

    # Mark unmatched GT with X
    for gi in unmatched_gt:
        cx, cy = box_center(gt_boxes[gi][0]).astype(int)
        cv2.drawMarker(vis, (cx, cy), GT_COLOR, cv2.MARKER_TILTED_CROSS, 12, 2)

    # Mark unmatched pseudo with X
    for pi in unmatched_pred:
        cx, cy = box_center(pred_boxes[pi][0]).astype(int)
        cv2.drawMarker(vis, (cx, cy), PSEUDO_COLOR, cv2.MARKER_TILTED_CROSS, 12, 2)

    return vis


def compute_stats(gt_boxes, pred_boxes, iou_thr=0.5):
    """Compute matching statistics for one image."""
    matched, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_thr)

    # Center distance for matched pairs
    center_dists = []
    ious = []
    for gi, pi, iou in matched:
        gc = box_center(gt_boxes[gi][0])
        pc = box_center(pred_boxes[pi][0])
        center_dists.append(np.linalg.norm(gc - pc))
        ious.append(iou)

    # mIoU: for every GT box, find its best IoU with any pred (no threshold)
    gt_best_ious = []
    if gt_boxes and pred_boxes:
        for gc, _ in gt_boxes:
            best = max(box_iou_obb(gc, pc) for pc, _ in pred_boxes)
            gt_best_ious.append(best)
    elif gt_boxes:
        gt_best_ious = [0.0] * len(gt_boxes)

    return dict(
        n_gt=len(gt_boxes),
        n_pred=len(pred_boxes),
        n_matched=len(matched),
        n_missed=len(unmatched_gt),
        n_false_pos=len(unmatched_pred),
        mean_iou=np.mean(ious) if ious else 0.0,
        mean_center_dist=np.mean(center_dists) if center_dists else 0.0,
        median_center_dist=np.median(center_dists) if center_dists else 0.0,
        gt_best_ious=gt_best_ious,
    )


def compute_per_class_metrics(all_gt, all_pred, iou_thr=0.5):
    """Compute per-class TP/FP/FN across all images.

    Args:
        all_gt: list of list of (coords, cls_name) per image.
        all_pred: list of list of (coords, cls_name) per image.

    Returns:
        dict: {class_name: {tp, fp, fn, precision, recall, f1}}
    """
    # Collect per-class GT and pred across all images
    class_tp = {c: 0 for c in CLASSES}
    class_fp = {c: 0 for c in CLASSES}
    class_fn = {c: 0 for c in CLASSES}

    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        # Group by class
        gt_by_cls = {}
        for coords, cls_name in gt_boxes:
            gt_by_cls.setdefault(cls_name, []).append(coords)
        pred_by_cls = {}
        for coords, cls_name in pred_boxes:
            pred_by_cls.setdefault(cls_name, []).append(coords)

        all_cls = set(list(gt_by_cls.keys()) + list(pred_by_cls.keys()))
        for cls_name in all_cls:
            gts = gt_by_cls.get(cls_name, [])
            preds = pred_by_cls.get(cls_name, [])

            if not gts:
                class_fp.setdefault(cls_name, 0)
                class_fp[cls_name] += len(preds)
                continue
            if not preds:
                class_fn.setdefault(cls_name, 0)
                class_fn[cls_name] += len(gts)
                continue

            # IoU matching within this class
            iou_mat = np.zeros((len(gts), len(preds)))
            for i, gc in enumerate(gts):
                for j, pc in enumerate(preds):
                    iou_mat[i, j] = box_iou_obb(gc, pc)

            matched_g = set()
            matched_p = set()
            # Greedy match
            while True:
                if iou_mat.size == 0:
                    break
                idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                if iou_mat[idx] < iou_thr:
                    break
                matched_g.add(idx[0])
                matched_p.add(idx[1])
                iou_mat[idx[0], :] = -1
                iou_mat[:, idx[1]] = -1

            tp = len(matched_g)
            class_tp.setdefault(cls_name, 0)
            class_fp.setdefault(cls_name, 0)
            class_fn.setdefault(cls_name, 0)
            class_tp[cls_name] += tp
            class_fp[cls_name] += len(preds) - tp
            class_fn[cls_name] += len(gts) - tp

    results = {}
    for cls_name in CLASSES:
        tp = class_tp.get(cls_name, 0)
        fp = class_fp.get(cls_name, 0)
        fn = class_fn.get(cls_name, 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        results[cls_name] = dict(tp=tp, fp=fp, fn=fn,
                                 precision=prec, recall=rec, f1=f1)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare pseudo-labels with GT annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('gt_dir', help='GT annotation directory')
    parser.add_argument('pseudo_dir', help='Pseudo-label annotation directory')
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('--images', nargs='*', default=None,
                        help='Specific image filenames (default: all)')
    parser.add_argument('--num-images', type=int, default=20,
                        help='Max images for visualization (if --images not given)')
    parser.add_argument('--iou-thr', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--output-dir', '-o', default='vis_output/pseudo_compare',
                        help='Output directory for visualizations')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print aggregate stats, no images')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Resolve image list
    if args.images:
        img_names = args.images
    else:
        # Use all images that have both GT and pseudo annotations
        gt_stems = {Path(f).stem for f in os.listdir(args.gt_dir) if f.endswith('.txt')}
        pseudo_stems = {Path(f).stem for f in os.listdir(args.pseudo_dir) if f.endswith('.txt')}
        common = sorted(gt_stems & pseudo_stems)
        if not args.stats_only:
            import random
            random.seed(args.seed)
            common = random.sample(common, min(args.num_images, len(common)))
        img_names = [s + '.png' for s in common]

    if not args.stats_only:
        os.makedirs(args.output_dir, exist_ok=True)

    # Aggregate stats
    all_stats = []
    all_gt_boxes = []
    all_pred_boxes = []

    for img_name in tqdm(img_names, desc='Comparing', unit='img'):
        stem = Path(img_name).stem
        gt_boxes = load_annotations(os.path.join(args.gt_dir, stem + '.txt'))
        pred_boxes = load_annotations(os.path.join(args.pseudo_dir, stem + '.txt'))

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

        stats = compute_stats(gt_boxes, pred_boxes, args.iou_thr)
        all_stats.append(stats)

        if not args.stats_only:
            img_path = os.path.join(args.img_dir, img_name)
            if not os.path.isfile(img_path):
                print(f'[WARN] Image not found: {img_path}')
                continue

            img = cv2.imread(img_path)
            matched, unmatched_gt, unmatched_pred = match_boxes(
                gt_boxes, pred_boxes, args.iou_thr)
            vis = draw_comparison(img, gt_boxes, pred_boxes,
                                  matched, unmatched_gt, unmatched_pred)

            # Add stats text
            text = (f"GT:{stats['n_gt']} Pred:{stats['n_pred']} "
                    f"Match:{stats['n_matched']} Miss:{stats['n_missed']} "
                    f"FP:{stats['n_false_pos']} "
                    f"mIoU:{stats['mean_iou']:.2f} "
                    f"mDist:{stats['mean_center_dist']:.1f}px")
            cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            out_path = os.path.join(args.output_dir, f'{stem}_compare.jpg')
            cv2.imwrite(out_path, vis)

    # Print aggregate statistics
    if all_stats:
        total_gt = sum(s['n_gt'] for s in all_stats)
        total_pred = sum(s['n_pred'] for s in all_stats)
        total_matched = sum(s['n_matched'] for s in all_stats)
        total_missed = sum(s['n_missed'] for s in all_stats)
        total_fp = sum(s['n_false_pos'] for s in all_stats)

        recall = total_matched / max(total_gt, 1)
        precision = total_matched / max(total_pred, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        matched_stats = [s for s in all_stats if s['n_matched'] > 0]
        mean_iou = np.mean([s['mean_iou'] for s in matched_stats]) if matched_stats else 0
        mean_dist = np.mean([s['mean_center_dist'] for s in matched_stats]) if matched_stats else 0

        # mIoU over all GT boxes (best-match, no threshold)
        all_gt_best = []
        for s in all_stats:
            all_gt_best.extend(s['gt_best_ious'])
        miou_all = np.mean(all_gt_best) if all_gt_best else 0.0

        print(f'\n{"="*60}')
        print(f' Pseudo-label Quality (IoU threshold = {args.iou_thr})')
        print(f'{"="*60}')
        print(f' Images evaluated:  {len(all_stats)}')
        print(f' Total GT boxes:    {total_gt}')
        print(f' Total predictions: {total_pred}')
        print(f' Matched:           {total_matched}')
        print(f' Missed (FN):       {total_missed}')
        print(f' False positives:   {total_fp}')
        print(f'{"─"*60}')
        print(f' Recall:            {recall:.4f}')
        print(f' Precision:         {precision:.4f}')
        print(f' F1:                {f1:.4f}')
        print(f' Mean IoU (matched):{mean_iou:.4f}')
        print(f' mIoU (all GT):     {miou_all:.4f}')
        print(f' Mean center dist:  {mean_dist:.1f} px')

        # Per-class metrics
        per_cls = compute_per_class_metrics(all_gt_boxes, all_pred_boxes, args.iou_thr)
        print(f'\n{"─"*60}')
        print(f' Per-class metrics @ IoU={args.iou_thr}')
        print(f'{"─"*60}')
        print(f' {"Class":<22s} {"GT":>5s} {"Pred":>5s} {"TP":>5s} '
              f'{"Prec":>6s} {"Rec":>6s} {"F1":>6s}')
        print(f' {"─"*57}')

        total_cls_tp = 0
        total_cls_gt = 0
        total_cls_pred = 0
        active_classes = 0

        for cls_name in CLASSES:
            m = per_cls[cls_name]
            n_gt_cls = m['tp'] + m['fn']
            n_pred_cls = m['tp'] + m['fp']
            print(f' {cls_name:<22s} {n_gt_cls:>5d} {n_pred_cls:>5d} {m["tp"]:>5d} '
                  f'{m["precision"]:>6.3f} {m["recall"]:>6.3f} {m["f1"]:>6.3f}')
            total_cls_tp += m['tp']
            total_cls_gt += n_gt_cls
            total_cls_pred += n_pred_cls
            if n_gt_cls > 0:
                active_classes += 1

        # mAP50-style metric: mean of per-class recall (= per-class AP when no score)
        mean_recall = np.mean([per_cls[c]['recall'] for c in CLASSES
                               if per_cls[c]['tp'] + per_cls[c]['fn'] > 0])
        mean_prec = np.mean([per_cls[c]['precision'] for c in CLASSES
                             if per_cls[c]['tp'] + per_cls[c]['fp'] > 0])
        mean_f1 = np.mean([per_cls[c]['f1'] for c in CLASSES
                           if per_cls[c]['tp'] + per_cls[c]['fn'] > 0])
        print(f' {"─"*57}')
        print(f' {"Mean":<22s} {"":>5s} {"":>5s} {"":>5s} '
              f'{mean_prec:>6.3f} {mean_recall:>6.3f} {mean_f1:>6.3f}')
        print(f'{"="*60}')

        if not args.stats_only:
            stats_path = os.path.join(args.output_dir, 'stats.txt')
            with open(stats_path, 'w') as f:
                f.write(f'iou_thr={args.iou_thr}\n')
                f.write(f'images={len(all_stats)}\n')
                f.write(f'gt={total_gt} pred={total_pred} matched={total_matched}\n')
                f.write(f'recall={recall:.4f} precision={precision:.4f} f1={f1:.4f}\n')
                f.write(f'mean_iou={mean_iou:.4f} miou_all={miou_all:.4f} mean_dist={mean_dist:.1f}px\n\n')
                f.write(f'Per-class @ IoU={args.iou_thr}:\n')
                f.write(f'{"Class":<22s} {"GT":>5s} {"Pred":>5s} {"TP":>5s} '
                        f'{"Prec":>6s} {"Rec":>6s} {"F1":>6s}\n')
                for cls_name in CLASSES:
                    m = per_cls[cls_name]
                    n_gt_cls = m['tp'] + m['fn']
                    n_pred_cls = m['tp'] + m['fp']
                    f.write(f'{cls_name:<22s} {n_gt_cls:>5d} {n_pred_cls:>5d} '
                            f'{m["tp"]:>5d} {m["precision"]:>6.3f} '
                            f'{m["recall"]:>6.3f} {m["f1"]:>6.3f}\n')
                f.write(f'Mean precision={mean_prec:.4f} recall={mean_recall:.4f} '
                        f'f1={mean_f1:.4f}\n')
            print(f'\nVisualizations saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
