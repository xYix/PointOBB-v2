#!/usr/bin/env bash
# ============================================================================
# Compare pseudo-labels with GT annotations (visualization + stats)
#
# Usage:
#   bash tools/compare_pseudo_vis.sh                          # default images
#   bash tools/compare_pseudo_vis.sh --images P0001__1024__228___2472.png
#   bash tools/compare_pseudo_vis.sh --stats-only             # aggregate stats, no images
# ============================================================================
set -euo pipefail

GT_ANN_DIR="/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles"
PSEUDO_DIR="/mnt/tmp/datasets/DOTAv10pseudolabel_vpdv2"
IMG_DIR="/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images"
OUT_DIR="vis_output/pseudo_compare"

DEFAULT_IMAGES=(
    "P0001__1024__228___2472.png"
    "P0060__1024__824___824.png"
    "P1181__1024__1648___0.png"
    "P2759__1024__0___0.png"
    "P0002__1024__0___0.png"
    "P0010__1024__0___0.png"
    "P0020__1024__3296___0.png"
    "P0883__1024__314___387.png"
    "P0890__1024__650___2472.png"
    "P2476__1024__824___824.png"
)

IMAGES=()
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                IMAGES+=("$1"); shift
            done
            ;;
        --out)          OUT_DIR="$2"; shift 2 ;;
        --gt-dir)       GT_ANN_DIR="$2"; shift 2 ;;
        --pseudo-dir)   PSEUDO_DIR="$2"; shift 2 ;;
        --img-dir)      IMG_DIR="$2"; shift 2 ;;
        --stats-only)   EXTRA_ARGS="--stats-only"; shift ;;
        --iou-thr)      EXTRA_ARGS="${EXTRA_ARGS} --iou-thr $2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ${#IMAGES[@]} -eq 0 ]; then
    IMAGES=("${DEFAULT_IMAGES[@]}")
fi

IMG_ARGS="${IMAGES[*]}"

echo "================================================"
echo " Pseudo-label vs GT Comparison"
echo "================================================"
echo " GT:     ${GT_ANN_DIR}"
echo " Pseudo: ${PSEUDO_DIR}"
echo " Images: ${#IMAGES[@]}"
echo " Output: ${OUT_DIR}"
echo "================================================"
echo ""

python tools/compare_pseudolabel.py \
    "${GT_ANN_DIR}" "${PSEUDO_DIR}" "${IMG_DIR}" \
    --images ${IMG_ARGS} \
    -o "${OUT_DIR}" \
    ${EXTRA_ARGS}
