#!/usr/bin/env bash
# ============================================================================
# Batch visualization: compare PointOBB-v2 baseline vs VPD method
#
# Generates side-by-side results for paper figure selection:
#   - cpm:    CPM class probability heatmaps (both methods)
#   - varmap: Uncertainty + mu maps (VPD only)
#   - detect: Detection results with GT overlay (both methods)
#
# Usage:
#   bash tools/batch_compare_vis.sh                   # default images
#   bash tools/batch_compare_vis.sh --images P0001__1024__228___2472.png P0060__1024__824___824.png
#   bash tools/batch_compare_vis.sh --device cuda:0
# ============================================================================
set -euo pipefail

# ── Configs & checkpoints ────────────────────────────────────────────────────
CFG_OBBV2="configs/pointobbv2/train_cpm_dotav10.py"
CKPT_OBBV2="checkpoints/train_cpm_epoch_6.pth"

CFG_VPD="configs/pointobbv2/train_cpm_vpd_point_dotav10.py"
CKPT_VPD="work_dirs/stride4/epoch_1.pth"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEVICE="cuda:1"
LEVELS="0 1"
OUT_ROOT="vis_output/compare"

# A curated set of images covering diverse scenes.
# Override via: --images img1.png img2.png ...
DEFAULT_IMAGES=(
    "P0001__1024__228___2472.png"   # airport with planes
    "P0060__1024__824___824.png"    # residential area
    "P1181__1024__1648___0.png"     # mixed scene
    "P2759__1024__0___0.png"        # sparse scene
    "P0002__1024__0___0.png"        # dense buses
    "P0010__1024__0___0.png"        # multi class
    "P0020__1024__3296___0.png"
    "P0883__1024__314___387.png"
    "P0890__1024__650___2472.png"
    "P2476__1024__824___824.png"
)

# ── Parse CLI args ───────────────────────────────────────────────────────────
IMAGES=()
SKIP_OBBV2=false
SKIP_VPD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                IMAGES+=("$1"); shift
            done
            ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --levels)
            shift; LEVELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LEVELS="$LEVELS $1"; shift
            done
            ;;
        --out)          OUT_ROOT="$2"; shift 2 ;;
        --ckpt-obbv2)   CKPT_OBBV2="$2"; shift 2 ;;
        --ckpt-vpd)     CKPT_VPD="$2"; shift 2 ;;
        --skip-obbv2)   SKIP_OBBV2=true; shift ;;
        --skip-vpd)     SKIP_VPD=true; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ${#IMAGES[@]} -eq 0 ]; then
    IMAGES=("${DEFAULT_IMAGES[@]}")
fi

IMG_ARGS="${IMAGES[*]}"

echo "================================================"
echo " Batch Comparison: PointOBB-v2 vs VPD"
echo "================================================"
echo " Device:     $DEVICE"
echo " Levels:     $LEVELS"
echo " Images:     ${IMAGES[*]}"
echo " Output:     $OUT_ROOT/"
echo "================================================"
echo ""

# ── Helper ───────────────────────────────────────────────────────────────────
run_vis() {
    local subcmd="$1"
    local tag="$2"
    local cfg="$3"
    local ckpt="$4"
    local out_dir="${OUT_ROOT}/${subcmd}/${tag}"
    shift 4
    # remaining args are extra flags

    # --levels only supported by cpm and varmap, not detect/loss
    local level_args=""
    if [[ "$subcmd" == "cpm" || "$subcmd" == "varmap" ]]; then
        level_args="--levels ${LEVELS}"
    fi

    echo "[${subcmd}] ${tag} → ${out_dir}/"
    python tools/visualize.py "${subcmd}" \
        "${cfg}" "${ckpt}" \
        --images ${IMG_ARGS} \
        ${level_args} \
        --device "${DEVICE}" \
        -o "${out_dir}" \
        "$@"
}

# ── 0. Raw images + GT annotations ───────────────────────────────────────────
# Resolve image prefix from config
IMG_PREFIX=$(python -c "
from mmcv import Config
cfg = Config.fromfile('${CFG_OBBV2}')
print(cfg.data.train.img_prefix)
")

RAW_OUT="${OUT_ROOT}/raw"
mkdir -p "${RAW_OUT}"
echo "=== Raw images ==="
for img in ${IMG_ARGS}; do
    src="${IMG_PREFIX}/${img}"
    if [ -f "$src" ]; then
        cp "$src" "${RAW_OUT}/"
    else
        echo "[WARN] not found: $src"
    fi
done
echo "[raw] ${RAW_OUT}/ (${#IMAGES[@]} files)"
echo ""

echo "=== GT annotations (OBB + center points) ==="
GT_OUT="${OUT_ROOT}/gtvis"
echo "[gtvis] → ${GT_OUT}/"
python tools/visualize.py gtvis \
    "$CFG_OBBV2" \
    --images ${IMG_ARGS} \
    -o "${GT_OUT}"
echo ""

# ── 1. CPM heatmaps (both methods) ──────────────────────────────────────────
echo "=== CPM classification heatmaps ==="
if [ "$SKIP_OBBV2" = false ]; then
    run_vis cpm obbv2 "$CFG_OBBV2" "$CKPT_OBBV2"
fi
if [ "$SKIP_VPD" = false ]; then
    run_vis cpm vpd "$CFG_VPD" "$CKPT_VPD"
fi
echo ""

# ── 2. Detection results (both methods) ─────────────────────────────────────
echo "=== Detection results ==="
if [ "$SKIP_OBBV2" = false ]; then
    run_vis detect obbv2 "$CFG_OBBV2" "$CKPT_OBBV2" --score-thr 0.3
fi
if [ "$SKIP_VPD" = false ]; then
    run_vis detect vpd "$CFG_VPD" "$CKPT_VPD" --score-thr 0.3
fi
echo ""

# ── 3. Variance / uncertainty maps (VPD only) ───────────────────────────────
echo "=== Variance maps (VPD only) ==="
if [ "$SKIP_VPD" = false ]; then
    run_vis varmap vpd "$CFG_VPD" "$CKPT_VPD"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "================================================"
echo " Done. Results:"
echo ""
find "${OUT_ROOT}" -type d | sort | while read -r d; do
    count=$(find "$d" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "   ${d}/  (${count} files)"
    fi
done
echo ""
echo " Directory structure:"
echo "   ${OUT_ROOT}/"
echo "     raw/          — Original images"
echo "     gtvis/        — GT annotations (OBB + center points)"
echo "     cpm/obbv2/    — CPM heatmaps (baseline)"
echo "     cpm/vpd/      — CPM heatmaps (ours)"
echo "     detect/obbv2/ — Detection results (baseline)"
echo "     detect/vpd/   — Detection results (ours)"
echo "     varmap/vpd/       — Uncertainty & mu maps (ours)"
echo "================================================"
