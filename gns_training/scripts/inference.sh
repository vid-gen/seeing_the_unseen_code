#!/bin/bash
# 全场景推理脚本: 对 7 个场景各随机选 5 个 trial 生成 GIF
# 使用 RANSAC 刚体投影，全时段 rollout
# (Derived from the original run_inference_all.sh; virtualenv and absolute paths removed.
#  Trials are now sampled from <Scenario>/valid.txt by default, see --split in inference_pyg.py.)
#
# Usage (from anywhere):
#   bash scripts/inference.sh <DATA_ROOT> [CHECKPOINT] [OUTPUT_BASE]
#   DATA_ROOT   : preprocessed data root containing <Scenario>/{valid.txt,<trial>/...}
#   CHECKPOINT  : default ../checkpoints/gns/without_Drape/best_model.pth (relative to gns_training/)
#   OUTPUT_BASE : default ./inference_output_all
#   SPLIT       : (env, optional) valid | train | all   (default valid)

set -e

DATA_ROOT="${1:-${DATA_ROOT:-}}"
if [ -z "${DATA_ROOT}" ]; then
    echo "Usage: bash scripts/inference.sh <DATA_ROOT> [CHECKPOINT] [OUTPUT_BASE]" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 路径配置
CHECKPOINT="${2:-${SCRIPT_DIR}/../checkpoints/gns/without_Drape/best_model.pth}"
OUTPUT_BASE="${3:-./inference_output_all}"
SCRIPT="${SCRIPT_DIR}/inference_pyg.py"
SPLIT="${SPLIT:-valid}"

# 场景列表
SCENARIOS=(Collide Contain Dominoes Drop Link Roll Support)

echo "============================================================"
echo "全场景推理开始"
echo "Checkpoint: $CHECKPOINT"
echo "输出目录: $OUTPUT_BASE"
echo "场景数: ${#SCENARIOS[@]}"
echo "每场景 trial 数: 5"
echo "Split: $SPLIT"
echo "RANSAC: ON"
echo "============================================================"

for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "============================================================"
    echo "开始推理场景: $scenario"
    echo "============================================================"

    # python puts the directory of inference_pyg.py on sys.path, so `models.GNS_pyg` / `utils_geom`
    # resolve regardless of the current working directory.
    python "$SCRIPT" \
        --checkpoint "$CHECKPOINT" \
        --data_root "$DATA_ROOT" \
        --scenario "$scenario" \
        --split "$SPLIT" \
        --n_trials 5 \
        --full_rollout \
        --start_timestep 45 \
        --training_fpt 3 \
        --neighbor_radius 0.08 \
        --ransac_on_pred 1 \
        --visualize \
        --gif_fps 4 \
        --seed 42 \
        --outf "${OUTPUT_BASE}/${scenario}"

    echo "场景 $scenario 推理完成"
done

echo ""
echo "============================================================"
echo "全部场景推理完成!"
echo "输出目录: $OUTPUT_BASE"
echo "============================================================"
