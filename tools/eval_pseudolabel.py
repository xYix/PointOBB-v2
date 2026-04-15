#!/usr/bin/env python3
"""Evaluate pseudolabel quality: generate VPD pseudolabels (no random flip, GPU)
and compare against baseline and GT.

Usage:
    python tools/eval_pseudolabel.py                                          # defaults
    python tools/eval_pseudolabel.py --ckpt work_dirs/stride4/epoch_2.pth     # custom ckpt
    python tools/eval_pseudolabel.py --n 1000                                 # more images
    python tools/eval_pseudolabel.py --baseline-only                          # skip VPD gen
"""
import torch, warnings, os, glob, time, argparse
import numpy as np, cv2
warnings.filterwarnings('ignore')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='work_dirs/stride4/epoch_2.pth')
    p.add_argument('--cfg', default='configs/pointobbv2/generate_vpd_pseudo_label_dotav10.py')
    p.add_argument('--gt-dir', default='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles')
    p.add_argument('--baseline-dir', default='/mnt/tmp/datasets/DOTAv10pseudolabel')
    p.add_argument('--out-dir', default='/tmp/vpd_eval_output')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--baseline-only', action='store_true', help='Skip VPD generation, only evaluate existing dirs')
    p.add_argument('--extra-dirs', nargs='*', default=[], help='Extra pseudolabel dirs to evaluate (name:path)')
    return p.parse_args()


def load_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            boxes.append(([float(x) for x in parts[:8]], parts[8]))
    return boxes


def box_iou(a, b):
    pa = np.array(a, dtype=np.float32).reshape(4, 2)
    pb = np.array(b, dtype=np.float32).reshape(4, 2)
    ra, rb = cv2.minAreaRect(pa), cv2.minAreaRect(pb)
    ret, reg = cv2.rotatedRectangleIntersection(ra, rb)
    if ret == cv2.INTERSECT_NONE or reg is None:
        return 0.0
    inter = cv2.contourArea(reg)
    return inter / max(cv2.contourArea(pa) + cv2.contourArea(pb) - inter, 1e-6)


def evaluate(pdir, gt_dir, stems):
    tp, fn, fp, ious = 0, 0, 0, []
    all_gt_best_ious = []  # best IoU for every GT box (0 if unmatched)
    sizes = []
    for s in stems:
        gt = load_boxes(f'{gt_dir}/{s}.txt')
        pred = load_boxes(f'{pdir}/{s}.txt')
        for c, _ in pred:
            pts = np.array(c).reshape(4, 2)
            sizes.append((np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
        if not gt and not pred:
            continue
        if not pred:
            fn += len(gt)
            all_gt_best_ious.extend([0.0] * len(gt))
            continue
        if not gt:
            fp += len(pred); continue
        iou_mat = np.zeros((len(gt), len(pred)))
        for i, (gc, _) in enumerate(gt):
            for j, (pc, _) in enumerate(pred):
                iou_mat[i, j] = box_iou(gc, pc)
        # Record best IoU per GT (before greedy matching zeroes rows)
        gt_best = iou_mat.max(axis=1)
        all_gt_best_ious.extend(gt_best.tolist())
        # Greedy matching for TP/FP/FN
        ug, up = set(), set()
        while True:
            idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            if iou_mat[idx] < 0.5:
                break
            ug.add(idx[0]); up.add(idx[1]); ious.append(iou_mat[idx])
            iou_mat[idx[0], :] = -1; iou_mat[:, idx[1]] = -1
        tp += len(ug); fn += len(gt) - len(ug); fp += len(pred) - len(up)
    r = tp / max(tp + fn, 1); p = tp / max(tp + fp, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    miou_matched = np.mean(ious) if ious else 0
    miou_all = np.mean(all_gt_best_ious) if all_gt_best_ious else 0
    arr = np.array(sizes) if sizes else np.zeros((0, 2))
    return dict(r=r, p=p, f1=f1, miou=miou_matched, miou_all=miou_all,
                tp=tp, fn=fn, fp=fp,
                mean_w=arr[:, 0].mean() if len(arr) else 0,
                mean_h=arr[:, 1].mean() if len(arr) else 0)


def generate_vpd(cfg_path, ckpt, out_dir, n):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import mmrotate  # noqa: F401
    from mmrotate.models import build_detector
    from mmdet.datasets import build_dataset
    from mmcv.parallel import collate

    cfg = Config.fromfile(cfg_path)
    # Disable random flip for deterministic evaluation
    for t in cfg.data.train.pipeline:
        if t['type'] == 'RRandomFlip':
            t['flip_ratio'] = [0.0, 0.0, 0.0]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt, map_location='cpu', strict=False)
    model = model.cuda().float()
    model.train()

    print(f'  sigma_power={model.bbox_head.sigma_power}, mu_refine_radius={model.bbox_head.mu_refine_radius}')

    dataset = build_dataset(cfg.data.train)
    os.makedirs(out_dir, exist_ok=True)
    model.bbox_head.store_ann_dir = out_dir + '/'
    for f in glob.glob(f'{out_dir}/*.txt'):
        os.remove(f)

    t0 = time.time()
    for idx in range(min(n, len(dataset))):
        data = dataset[idx]
        batch = collate([data], samples_per_gpu=1)
        with torch.no_grad():
            feats = model.extract_feat(batch['img'].data[0].cuda().float())
            outs = model.bbox_head(feats)
            model.bbox_head.loss(
                *outs,
                [b.cuda() for b in batch['gt_bboxes'].data[0]],
                [l.cuda() for l in batch['gt_labels'].data[0]],
                batch['img_metas'].data[0])
        if (idx + 1) % 100 == 0:
            print(f'  Generated {idx+1}/{n} ({time.time()-t0:.0f}s)')
    print(f'  Done {min(n, len(dataset))} images in {time.time()-t0:.0f}s')


def main():
    args = parse_args()

    if not args.baseline_only:
        print(f'Generating VPD pseudolabels (ckpt={args.ckpt}, n={args.n})...')
        generate_vpd(args.cfg, args.ckpt, args.out_dir, args.n)

    # Collect stems from VPD output or GT
    if not args.baseline_only:
        stems = sorted([os.path.splitext(os.path.basename(f))[0]
                        for f in glob.glob(f'{args.out_dir}/*.txt')])
    else:
        stems = sorted([os.path.splitext(os.path.basename(f))[0]
                        for f in glob.glob(f'{args.gt_dir}/*.txt')])[:args.n]

    datasets = [('baseline', args.baseline_dir)]
    if not args.baseline_only:
        datasets.append(('vpd', args.out_dir))
    for extra in args.extra_dirs:
        name, path = extra.split(':', 1)
        datasets.append((name, path))

    print(f'\n{"="*88}')
    print(f'Results on {len(stems)} images  (IoU threshold = 0.5, no random flip)')
    print(f'  mIoU = matched only,  aIoU = all GT (unmatched counted as 0)')
    print(f'{"="*88}')
    print(f'{"Method":<22s} {"Recall":>7s} {"Prec":>7s} {"F1":>7s} {"mIoU":>7s} {"aIoU":>7s}'
          f' {"TP":>6s} {"FN":>6s} {"FP":>6s}  {"mean w×h":>12s}')
    print('-' * 88)
    for name, d in datasets:
        e = evaluate(d, args.gt_dir, stems)
        print(f'{name:<22s} {e["r"]:7.4f} {e["p"]:7.4f} {e["f1"]:7.4f} {e["miou"]:7.4f} {e["miou_all"]:7.4f}'
              f' {e["tp"]:6d} {e["fn"]:6d} {e["fp"]:6d}  {e["mean_w"]:5.1f}×{e["mean_h"]:<5.1f}')
    print(f'{"="*88}')


if __name__ == '__main__':
    main()
