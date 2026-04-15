#!/bin/bash
# Sweep sigma_power on 500-image subset (fast: ~10min per value)
# Usage: bash tools/sweep_sigma_power.sh <gpu_id> <power1> <power2> ...

GPU=$1; shift
GT=/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles_sub500
IMG=/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images
BASE_CFG=configs/pointobbv2/gen_vpd_sigma_pca28.py

for POWER in "$@"; do
    OUTDIR="/mnt/tmp/datasets/_sweep_power_${POWER}_gpu${GPU}"
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

    # Create temp config in configs dir (so _base_ relative paths work)
    TMPCFG=configs/pointobbv2/_sweep_p${POWER}_g${GPU}.py
    sed "s|sigma_power=1.0|sigma_power=${POWER}|
         s|DOTAv10pseudolabel_vpd_sigma_pca28/|_sweep_power_${POWER}_gpu${GPU}/|
         s|trainval/annfiles/|trainval/annfiles_sub500/|" $BASE_CFG > $TMPCFG

    echo ">>> sigma_power=$POWER (GPU $GPU) ..."
    python tools/train.py $TMPCFG \
        --work-dir /tmp/_sweep_wd_${POWER}_${GPU} \
        --gpu-ids $GPU 2>/dev/null | tail -1

    N=$(ls "$OUTDIR"/*.txt 2>/dev/null | wc -l)
    echo -n "    $N files | "

    python tools/compare_pseudolabel.py $GT "$OUTDIR" $IMG \
        --stats-only 2>/dev/null | grep -E "mIoU \(all|F1:" | tr '\n' ' '
    echo ""

    rm -f $TMPCFG
    rm -rf /tmp/_sweep_wd_${POWER}_${GPU} "$OUTDIR"
done
