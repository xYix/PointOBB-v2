#!/usr/bin/env bash
# ============================================================================
# Merge baseline (6-epoch cls) + VPD (sigma tower) checkpoints.
# The baseline provides high-quality classification features,
# VPD provides the learned sigma (uncertainty) tower.
#
# Usage:
#   bash tools/merge_checkpoints.sh
#   bash tools/merge_checkpoints.sh baseline.pth vpd.pth output.pth
# ============================================================================

BASE_CKPT="${1:-checkpoints/train_cpm_epoch_6.pth}"
VPD_CKPT="${2:-work_dirs/stride4/epoch_2.pth}"
OUT_CKPT="${3:-work_dirs/stride4/merged_base6_vpd2.pth}"

python3 -c "
import torch

base = torch.load('${BASE_CKPT}', map_location='cpu')['state_dict']
vpd = torch.load('${VPD_CKPT}', map_location='cpu')['state_dict']

merged = {}
n_base, n_vpd = 0, 0

# Compatible keys: take from baseline (better cls branch, 6 epochs)
for k in base:
    if k in vpd and base[k].shape == vpd[k].shape:
        merged[k] = base[k]
        n_base += 1

# Sigma tower: take from VPD
for k in vpd:
    if 'sigma' in k:
        merged[k] = vpd[k]
        n_vpd += 1

# conv_reg: take from VPD (2-ch, not baseline's 4-ch)
for k in vpd:
    if 'conv_reg' in k:
        merged[k] = vpd[k]
        n_vpd += 1

torch.save({'state_dict': merged, 'meta': {'info': 'merged baseline+vpd'}}, '${OUT_CKPT}')
print(f'Merged: {n_base} keys from baseline, {n_vpd} keys from VPD -> ${OUT_CKPT}')
"
