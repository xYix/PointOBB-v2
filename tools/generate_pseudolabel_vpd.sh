#!/usr/bin/env bash
# Generate VPD-enhanced pseudo-labels using trained VPD checkpoint.
# Multi-GPU version using distributed training.
#
# Usage:
#   bash tools/generate_pseudolabel_vpd.sh                                    # default
#   bash tools/generate_pseudolabel_vpd.sh work_dirs/stride4/epoch_3.pth      # custom ckpt
#   GPUS=1 bash tools/generate_pseudolabel_vpd.sh                             # single GPU

CKPT="${1:-work_dirs/stride4/epoch_2.pth}"
GPUS="${GPUS:-2}"

CFG="configs/pointobbv2/generate_vpd_pseudo_label_dotav10.py"
WORK_DIR="work_dirs/vpd_pseudo_label"
mkdir -p "$WORK_DIR"

# Inject load_from into config (avoid --resume-from which restores epoch counter)
python -c "
from mmcv import Config
cfg = Config.fromfile('${CFG}')
cfg.load_from = '${CKPT}'
cfg.resume_from = None
cfg.dump('${WORK_DIR}/_tmp_cfg.py')
"

bash tools/dist_train.sh \
    "${WORK_DIR}/_tmp_cfg.py" \
    "$GPUS" \
    --work-dir "$WORK_DIR"
