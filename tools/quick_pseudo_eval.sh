#!/usr/bin/env bash
# ============================================================================
# Quick pseudo-label evaluation on a small subset (200 images by default).
# Generates pseudo-labels from a checkpoint, then compares with GT.
#
# Usage:
#   bash tools/quick_pseudo_eval.sh                                # defaults
#   bash tools/quick_pseudo_eval.sh work_dirs/stride4/epoch_6.pth  # custom ckpt
#   N=500 bash tools/quick_pseudo_eval.sh                          # more images
# ============================================================================
set -euo pipefail

CKPT="${1:-work_dirs/stride4/latest.pth}"
N="${N:-200}"
CFG="configs/pointobbv2/generate_vpd_pseudo_label_dotav10.py"
GT_DIR="/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles"
IMG_DIR="/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images"
BASE_DIR="/mnt/tmp/datasets/DOTAv10pseudolabel"
OUT_DIR="/tmp/quick_pseudo_eval"

echo "================================================"
echo " Quick Pseudo-label Evaluation"
echo " Checkpoint: $CKPT"
echo " N images:   $N"
echo "================================================"

python3 << PYEOF
import torch, warnings, os, glob, sys, time
import numpy as np, cv2
warnings.filterwarnings('ignore')
from mmcv import Config
from mmcv.runner import load_checkpoint
import mmrotate
from mmrotate.models import build_detector
from mmdet.datasets import build_dataset
from mmcv.parallel import collate

N = int("${N}")
ckpt = "${CKPT}"
gt_dir = "${GT_DIR}"
base_dir = "${BASE_DIR}"
out_dir = "${OUT_DIR}"

cfg = Config.fromfile("${CFG}")
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, ckpt, map_location='cpu', strict=False)
model = model.cuda().float()
model.train()

# Print what's enabled
has_mu = hasattr(model.bbox_head, 'conv_mu')
has_sigma = model.bbox_head.sigma_power > 0
print(f'  mu refinement: {has_mu}, sigma power: {model.bbox_head.sigma_power}')

dataset = build_dataset(cfg.data.train)
os.makedirs(out_dir, exist_ok=True)
model.bbox_head.store_ann_dir = out_dir + '/'
for f in glob.glob(f'{out_dir}/*.txt'): os.remove(f)

# Generate
t0 = time.time()
for idx in range(min(N, len(dataset))):
    data = dataset[idx]
    batch = collate([data], samples_per_gpu=1)
    with torch.no_grad():
        feats = model.extract_feat(batch['img'].data[0].cuda().float())
        outs = model.bbox_head(feats)
        model.bbox_head.loss(*outs,
            [b.cuda() for b in batch['gt_bboxes'].data[0]],
            [l.cuda() for l in batch['gt_labels'].data[0]],
            batch['img_metas'].data[0])
    if (idx+1) % 50 == 0:
        print(f'  Generated {idx+1}/{N} ({time.time()-t0:.0f}s)')
print(f'  Generated {min(N,len(dataset))} images in {time.time()-t0:.0f}s')

# Evaluate
def load_boxes(path):
    boxes = []
    if not os.path.exists(path): return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            boxes.append(([float(x) for x in parts[:8]], parts[8]))
    return boxes

def box_iou(a, b):
    pa = np.array(a, dtype=np.float32).reshape(4,2)
    pb = np.array(b, dtype=np.float32).reshape(4,2)
    ra, rb = cv2.minAreaRect(pa), cv2.minAreaRect(pb)
    ret, reg = cv2.rotatedRectangleIntersection(ra, rb)
    if ret == cv2.INTERSECT_NONE or reg is None: return 0.0
    inter = cv2.contourArea(reg)
    return inter / max(cv2.contourArea(pa) + cv2.contourArea(pb) - inter, 1e-6)

def evaluate(pdir, stems):
    tp, fn, fp, ious = 0, 0, 0, []
    sizes = []
    for s in stems:
        gt = load_boxes(f'{gt_dir}/{s}.txt')
        pred = load_boxes(f'{pdir}/{s}.txt')
        for c, _ in pred:
            pts = np.array(c).reshape(4,2)
            sizes.append((np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
        if not gt and not pred: continue
        if not pred: fn += len(gt); continue
        if not gt: fp += len(pred); continue
        iou_mat = np.zeros((len(gt), len(pred)))
        for i,(gc,_) in enumerate(gt):
            for j,(pc,_) in enumerate(pred):
                iou_mat[i,j] = box_iou(gc, pc)
        ug, up = set(), set()
        while True:
            idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            if iou_mat[idx] < 0.5: break
            ug.add(idx[0]); up.add(idx[1]); ious.append(iou_mat[idx])
            iou_mat[idx[0],:] = -1; iou_mat[:,idx[1]] = -1
        tp += len(ug); fn += len(gt)-len(ug); fp += len(pred)-len(up)
    r = tp/max(tp+fn,1); p = tp/max(tp+fp,1)
    f1 = 2*p*r/max(p+r,1e-6)
    miou = np.mean(ious) if ious else 0
    arr = np.array(sizes) if sizes else np.zeros((0,2))
    return dict(r=r, p=p, f1=f1, miou=miou, tp=tp, fn=fn, fp=fp,
                mean_w=arr[:,0].mean() if len(arr) else 0,
                mean_h=arr[:,1].mean() if len(arr) else 0,
                med_w=np.median(arr[:,0]) if len(arr) else 0,
                med_h=np.median(arr[:,1]) if len(arr) else 0)

stems = [os.path.splitext(os.path.basename(f))[0]
         for f in glob.glob(f'{out_dir}/*.txt')]

print(f'\n{"="*62}')
print(f' Results on {len(stems)} images  (IoU threshold = 0.5)')
print(f'{"="*62}')
print(f' {"Method":<16s} {"Recall":>7s} {"Prec":>7s} {"F1":>7s} {"mIoU":>7s}  {"mean w×h":>12s}  {"med w×h":>12s}')
print(f' {"-"*56}')
for name, d in [('baseline', base_dir), ('vpd', out_dir)]:
    e = evaluate(d, stems)
    print(f' {name:<16s} {e["r"]:7.4f} {e["p"]:7.4f} {e["f1"]:7.4f} {e["miou"]:7.4f}'
          f'  {e["mean_w"]:5.1f}×{e["mean_h"]:<5.1f}  {e["med_w"]:5.1f}×{e["med_h"]:<5.1f}')
print(f'{"="*62}')
PYEOF
