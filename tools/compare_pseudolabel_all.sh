#!/bin/bash
# Evaluate pseudo-label quality for both baseline and VPD-sigma
GT=/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles
IMG=/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images

# echo "========== Baseline (DOTAv10pseudolabel) =========="
# DIR=/mnt/tmp/datasets/DOTAv10pseudolabel
# python tools/compare_pseudolabel.py $GT $DIR $IMG --stats-only | tee $DIR/compare_stats.txt

# echo ""
# echo "========== VPD Sigma (DOTAv10pseudolabel_vpd_sigma) =========="
DIR=/mnt/tmp/datasets/DOTAv10pseudolabel_vpd_tuned
python tools/compare_pseudolabel.py $GT $DIR $IMG --stats-only | tee $DIR/compare_stats.txt
