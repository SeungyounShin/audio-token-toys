#!/bin/bash
set -e

# ── Audio Encoder Scaling Experiment ──
# 4 configs × {train + evaluate} on LibriSpeech train-clean-100

CONFIGS=("40ms_0pct" "40ms_50pct" "80ms_0pct" "80ms_50pct")
BASE_CFG="configs/base.yaml"

echo "============================================"
echo " Audio Encoder Scaling Experiment"
echo " Configs: ${CONFIGS[*]}"
echo "============================================"

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Training: ${config}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    accelerate launch \
        scripts/train.py \
        --base_config ${BASE_CFG} \
        --config configs/${config}.yaml

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Evaluating: ${config}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python scripts/evaluate.py \
        --base_config ${BASE_CFG} \
        --config configs/${config}.yaml \
        --checkpoint outputs/${config}/final.pt
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Generating scaling plots"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/plot_scaling.py

echo ""
echo "Done! Check outputs/ and figures/ for results."
