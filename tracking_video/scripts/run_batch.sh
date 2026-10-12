#!/bin/bash
# ========================================================================
# Batch tracking-video generation (pure SpaTrackerV2 path).
#
# Desensitized version of the job script that produced the training-set
# tracking videos used in the paper (4 x A100-40GB, 1 worker/GPU,
# 24 CPU threads/worker). Cluster-specific scheduler headers, container
# image, virtualenv and personal paths have been removed; set the
# variables below (or export them) before running.
#
# Resumable: re-running skips videos whose two output mp4s already exist.
#
# Usage:
#   SPATRACKER_ROOT=/path/to/SpaTrackerV2 \
#   DATA_DIR=/path/to/videos OUTPUT_DIR=/path/to/tracking \
#   bash tracking_video/scripts/run_batch.sh
# ========================================================================
set -euo pipefail

# ---------------- User config ----------------
NUM_GPUS="${NUM_GPUS:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-24}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the directory containing <video_name>.mp4 files}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the output root directory}"
# Root of the patched SpaTrackerV2 checkout that contains tracking_video/
SPATRACKER_ROOT="${SPATRACKER_ROOT:?Set SPATRACKER_ROOT to the SpaTrackerV2 checkout root}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/job_logs}"
# Optional: export HF_HOME=/path/to/hf_cache   (HuggingFace cache for the SpaTrackerV2 weights)
# ---------------------------------------------

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

JOBID="${PBS_JOBID:-${SLURM_JOB_ID:-local.$(date +%Y%m%d_%H%M%S)}}"
STDOUT_LOG="${LOG_DIR}/job.${JOBID}.log"
GPU_LOG="${LOG_DIR}/gpu.${JOBID}.log"

# Redirect everything from here onward
exec > "${STDOUT_LOG}" 2>&1

echo "=================================================================="
echo "JobID=${JOBID}  Host=$(hostname)  Time=$(date)"
echo "NUM_GPUS=${NUM_GPUS}  WORKERS_PER_GPU=${WORKERS_PER_GPU}"
echo "CPU_THREADS_PER_WORKER=${CPU_THREADS_PER_WORKER}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "=================================================================="

# GPU info (before starting job)
echo ""
echo "=== nvidia-smi ==="
nvidia-smi -L || true
nvidia-smi || true

# Background: sample GPU utilization every 60s
(
  while true; do
    echo "----- $(date) -----"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 60
  done
) >> "${GPU_LOG}" 2>&1 &
GPU_MON_PID=$!
trap "kill ${GPU_MON_PID} 2>/dev/null || true" EXIT

export PYTHONPATH="${SPATRACKER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SPATRACKER_ROOT}"

echo ">>> Python: $(which python)"
echo ">>> PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ">>> CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ">>> GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ">>> CPUs visible: $(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
echo ""

# Parameters identical to the run that produced the paper's training data.
python tracking_video/batch_run_v2.py \
    --num_gpus="${NUM_GPUS}" \
    --workers_per_gpu="${WORKERS_PER_GPU}" \
    --cpu_threads_per_worker="${CPU_THREADS_PER_WORKER}" \
    --data_dir="${DATA_DIR}" \
    --output_base_dir="${OUTPUT_DIR}" \
    --fps=1 \
    --das_fps=24 \
    --grid_size=70 \
    --grid_size_fv=90 \
    --depth_margin=5.0 \
    --das_point_size=2 \
    --tracker=v2 \
    --track_mode=offline \
    --vo_points=384 \
    --conf_threshold_depth=0.0

echo ""
echo "=================================================================="
echo "Batch finished at $(date)"
echo "Summary: ${OUTPUT_DIR}/logs/batch_summary.txt"
echo "=================================================================="
