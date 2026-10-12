#!/bin/bash
# GNS-PyG 7-scenario 4-GPU DDP training (the released `without_Drape` model).
# Derived from the original cluster job script; scheduler headers, container image,
# virtualenv and absolute paths removed. Hyper-parameters are unchanged.
#
# Usage (from anywhere):
#   bash scripts/train_4gpu.sh <DATA_ROOT> [OUTF]
#   DATA_ROOT : preprocessed data root containing <Scenario>/{train.txt,valid.txt,<trial>/...}
#   OUTF      : output dir (default ./output_pyg_ddp_4gpu/all_scenarios_bs8_lr1e4_without_Drape)
#   LOG_DIR   : (env, optional) where stdout/stderr/GPU-usage logs go (default ./logs)

set -e

DATA_ROOT="${1:-${DATA_ROOT:-}}"
if [ -z "${DATA_ROOT}" ]; then
  echo "Usage: bash scripts/train_4gpu.sh <DATA_ROOT> [OUTF]" >&2
  exit 1
fi
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"   # resolve before changing directory

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# 创建日志目录
LOG_DIR="${LOG_DIR:-./logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p ${LOG_DIR}
exec > ${LOG_DIR}/stdout.${RUN_ID}.log 2> ${LOG_DIR}/stderr.${RUN_ID}.log

echo "=== RunID=${RUN_ID} host=$(hostname) time=$(date) ==="
echo "=== GNS-PyG 全场景 4 GPU DDP 训练 ==="

# 记录 GPU 状态
nvidia-smi -L || true
nvidia-smi || true

# 每 60 秒记录 GPU 利用率
(
  while true; do
    echo "----- $(date) -----"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 60
  done
) >> ${LOG_DIR}/gpu.${RUN_ID}.log 2>&1 &
GPU_MON_PID=$!
trap "kill ${GPU_MON_PID} 2>/dev/null || true" EXIT

# 训练参数
# Drape
SCENARIO="Dominoes,Collide,Support,Link,Contain,Roll,Drop"
# 多场景训练更稳妥的每卡 batch（先保守，后续可按审查结果再调）
BATCH_SIZE=8
LR=1e-4
N_EPOCH=1000
NPROC=4
FORWARD_TIMES=1
SAMPLING_MODE="scenario_balanced"
TRAIN_SAMPLES_PER_TRIAL=2
CKP_PER_EPOCH=1
MAX_KEEP_CKPT=5
OUTF="${2:-./output_pyg_ddp_4gpu/all_scenarios_bs8_lr1e4_without_Drape}"

echo "模型: GNS-PyG DDP"
echo "场景: ${SCENARIO}"
echo "Data root: ${DATA_ROOT}"
echo "Per-GPU Batch Size: ${BATCH_SIZE}"
echo "Global Batch Size: $((BATCH_SIZE * NPROC))"
echo "forward_times: ${FORWARD_TIMES}"
echo "sampling_mode: ${SAMPLING_MODE}"
echo "train_samples_per_trial: ${TRAIN_SAMPLES_PER_TRIAL}"
echo "Epochs: ${N_EPOCH}"
echo "学习率: ${LR}"
echo "输出目录: ${OUTF}"
echo ""

export PYTHONUNBUFFERED=1

# NCCL 配置
export NCCL_DEBUG=WARN
export MASTER_PORT=29500

echo '>>> Python: '$(which python)
echo '>>> PyTorch: '$(python -c 'import torch; print(torch.__version__)')
echo '>>> CUDA devices: '$(python -c 'import torch; print(torch.cuda.device_count())')
echo ''

torchrun --standalone --nproc_per_node=${NPROC} train_pyg_ddp.py \
  --data_root "${DATA_ROOT}" \
  --scenario ${SCENARIO} \
  --batch_size ${BATCH_SIZE} \
  --forward_times ${FORWARD_TIMES} \
  --sampling_mode ${SAMPLING_MODE} \
  --train_samples_per_trial ${TRAIN_SAMPLES_PER_TRIAL} \
  --ckp_per_epoch ${CKP_PER_EPOCH} \
  --max_keep_ckpt ${MAX_KEEP_CKPT} \
  --lr ${LR} \
  --n_epoch ${N_EPOCH} \
  --outf "${OUTF}"

echo "=== GNS-PyG 全场景 4 GPU DDP 训练完成 time=$(date) ==="
