#!/bin/bash
# Usage: ./pipeline_preprocess.sh <scenario_name> [num_workers]
# Example: PHYSION_RAW_DIR=../data/physion_raw/dynamics_training ./pipeline_preprocess.sh Drop 32
#   PHYSION_RAW_DIR : directory holding <Scenario>_dynamics_training_HDF5s.tar.gz (or the extracted <Scenario>/);
#                     must equal `training_data_dir` in paths.yaml.
#   BINVOX_DIR      : (optional) directory containing the `binvox` executable, prepended to PATH.

set -e

# === binvox 环境配置 ===
if [ -n "${BINVOX_DIR:-}" ]; then
    export PATH="${BINVOX_DIR}:$PATH"
fi

# 启动 Xvfb 虚拟显示（如果未运行）
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting Xvfb..."
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 2
fi
export DISPLAY=:99

# 验证 binvox 可用
if ! command -v binvox &> /dev/null; then
    echo "ERROR: binvox not found in PATH"
    exit 1
fi
echo "binvox found: $(which binvox)"
# === binvox 配置结束 ===

SCENARIO=${1:-Drop}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "${PHYSION_RAW_DIR:-$SCRIPT_DIR/../data/physion_raw/dynamics_training}" && pwd)"

echo "=========================================="
echo "Processing Physion scenario: $SCENARIO"
echo "=========================================="

# 1. 解压
cd "$DATA_DIR"
if [ -f "${SCENARIO}_dynamics_training_HDF5s.tar.gz" ]; then
    if [ ! -d "$SCENARIO" ]; then
        echo "[1/4] Extracting ${SCENARIO} dataset..."
        python3 "$SCRIPT_DIR/extract_with_progress.py" "$DATA_DIR/${SCENARIO}_dynamics_training_HDF5s.tar.gz" "$DATA_DIR"
    else
        echo "[1/4] Directory already exists, skipping extraction..."
    fi
else
    echo "[1/4] Archive not found, checking if already extracted..."
    if [ ! -d "$SCENARIO" ]; then
        echo "ERROR: Neither archive nor directory found for $SCENARIO"
        exit 1
    fi
fi

# 2. 预处理 (使用并行版本)
cd "$SCRIPT_DIR"
NUM_WORKERS=${2:-$(nproc)}
echo "[2/4] Preprocessing ${SCENARIO} scenario with ${NUM_WORKERS} workers..."
python3 preprocessing_tdw_parallel.py --scenario "$SCENARIO" --mode train --num_workers "$NUM_WORKERS"

# 3. 生成 train/valid 划分
echo "[3/4] Creating train/valid split..."
python3 create_train_valid.py

# # 4. 删除原始数据释放空间
# echo "[4/4] Cleaning up..."
# rm -rf "$DATA_DIR/$SCENARIO"
# rm -f "$DATA_DIR/${SCENARIO}_dynamics_training_HDF5s.tar.gz"

# echo "=========================================="
# echo "Done! Preprocessed data saved to:"
# echo "<dpi_data_dir in paths.yaml>/train/$SCENARIO/"
# echo "=========================================="
