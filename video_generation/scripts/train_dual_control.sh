#!/bin/bash
# ============================================================
# Dual-tracking control training for Wan2.2-TI2V-5B
# (final "planF" recipe that produced the released checkpoint-10500)
#
# Recipe: patch_embedding lr = 2e-4, all other parameters lr = 2e-5 (low_lr_ratio = 10),
#         max_grad_norm = 0.5, weight decay = 3e-2, tracking_drop_ratio = 0, text_drop_ratio = 0.3,
#         2 GPUs (DDP, bf16, gradient checkpointing).
#
# Run from the ROOT of the patched VideoX-Fun checkout (see README.md, "Installation"):
#
#   bash scripts/wan2.2_fun/train_dual_control.sh \
#       --model_dir  /path/to/Wan2.2-TI2V-5B \
#       --data_root  /path/to/miradata2 \
#       [--train_meta /path/to/miradata2/metadata_dual_control_4999.json] \
#       [--output_dir output_dir_planF_10x_lr_ratio] \
#       [--resume latest | --resume checkpoint-9000] \
#       [--eval_scene_dir /path/to/evaluation_scene] \
#       [--no_validation]
#
#   --model_dir       Wan2.2-TI2V-5B base model directory (https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). Required.
#   --data_root       Dataset root that contains videos/, images/, tracking_4999/ and the metadata json. Required.
#   --train_meta      Metadata json (default: <data_root>/metadata_dual_control_4999.json).
#   --output_dir      Output directory for checkpoints / logs (default: output_dir_planF_10x_lr_ratio).
#   --resume          Optional. Passed to --resume_from_checkpoint ("latest" or "checkpoint-XXXX").
#                     Omit to train from the zero-initialised base model.
#   --eval_scene_dir  Optional. Adds the 6th validation sample used in the paper run (a simulated scene rendered
#                     by the scene_to_tracking module with --suffix _1.5s_fixed).
#   --no_validation   Optional. Disable periodic validation (the 5 MiraData validation ids below are
#                     specific to our local download; drop them if your copy uses different ids).
# ============================================================
set -e
set -o pipefail

MODEL_DIR=""
DATA_ROOT=""
TRAIN_META=""
OUTPUT_DIR="output_dir_planF_10x_lr_ratio"
RESUME=""
EVAL_SCENE_DIR=""
NO_VALIDATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_dir)      MODEL_DIR="$2"; shift 2 ;;
    --data_root)      DATA_ROOT="$2"; shift 2 ;;
    --train_meta)     TRAIN_META="$2"; shift 2 ;;
    --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --resume)         RESUME="$2"; shift 2 ;;
    --eval_scene_dir) EVAL_SCENE_DIR="$2"; shift 2 ;;
    --no_validation)  NO_VALIDATION=1; shift ;;
    -h|--help)        sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_DIR" || -z "$DATA_ROOT" ]]; then
  echo "ERROR: --model_dir and --data_root are required (see --help)."
  exit 1
fi
DATA_ROOT="${DATA_ROOT%/}"
if [[ -z "$TRAIN_META" ]]; then
  TRAIN_META="${DATA_ROOT}/metadata_dual_control_4999.json"
fi

# Optional resume
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint="$RESUME")
fi

# Validation samples (identical to the paper run; paths are now relative to --data_root / --eval_scene_dir)
VALIDATION_ARGS=()
if [[ "$NO_VALIDATION" -eq 0 ]]; then
  VAL_PROMPTS=(
    "A person riding a motorcycle in a video game."
    "A person is walking on a sidewalk with palm trees and buildings in the background."
    "An aerial view of a sandy beach next to the ocean."
    "The sky is blue and there are white clouds in it."
    "There is a person standing on a paddleboard in the ocean at sunset, with mountains in the background."
  )
  VAL_PATHS=()
  VAL_PATHS_2=()
  VAL_IMAGES=()
  for VID in 00000 00118 4014053.0 4167192.0 4266006.0; do
    VAL_PATHS+=("${DATA_ROOT}/tracking_4999/${VID}/first_frame_tracking_video.mp4")
    VAL_PATHS_2+=("${DATA_ROOT}/tracking_4999/${VID}/future_visible_tracking_video.mp4")
    VAL_IMAGES+=("${DATA_ROOT}/images/${VID}.png")
  done
  if [[ -n "$EVAL_SCENE_DIR" ]]; then
    EVAL_SCENE_DIR="${EVAL_SCENE_DIR%/}"
    VAL_PROMPTS+=("The wooden ball rolls and grazes the side of the candle; the candle spins and tips over at an angle while the ball deflects off its original trajectory.")
    VAL_PATHS+=("${EVAL_SCENE_DIR}/visible_tracking_pred_track_1.5s_fixed.mp4")
    VAL_PATHS_2+=("${EVAL_SCENE_DIR}/future_visible_tracking_das_1.5s_fixed.mp4")
    VAL_IMAGES+=("${EVAL_SCENE_DIR}/first_frame_1280x704.png")
  fi
  VALIDATION_ARGS=(
    --validation_steps=500
    --validation_prompts "${VAL_PROMPTS[@]}"
    --validation_paths "${VAL_PATHS[@]}"
    --validation_paths_2 "${VAL_PATHS_2[@]}"
    --validation_start_images "${VAL_IMAGES[@]}"
  )
fi

mkdir -p logs
LOG_FILE="logs/train_dual_control_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"

accelerate launch --num_processes=2 --mixed_precision="bf16" \
  scripts/wan2.2_fun/train_dual_control.py \
  --config_path="config/wan2.2/wan_dual_control_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_DIR" \
  --train_data_dir="${DATA_ROOT}/" \
  --train_data_meta="$TRAIN_META" \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=20 \
  --checkpointing_steps=500 \
  --learning_rate=2e-04 \
  --low_lr_ratio=10 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.5 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="control" \
  --trainable_modules "patch_embedding" \
  --trainable_modules_low_learning_rate "." \
  --tracking_drop_ratio=0.0 \
  --text_drop_ratio=0.3 \
  --checkpoints_total_limit=5 \
  "${VALIDATION_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
