# video_generation: dual-tracking controlled video generation (Wan2.2-TI2V-5B)

## Overview

This module fine-tunes **Wan2.2-TI2V-5B** so that it is conditioned on two tracking videos
in addition to the first frame and the text prompt:

| input | content | how it enters the model |
|---|---|---|
| first frame | RGB image | TI2V blending: the first latent frame is replaced by the VAE latent of the image at every step (its timestep is set to 0) |
| tracking video 1 | visible-point tracking video (anchored at the first frame) | VAE-encoded (48 ch), concatenated to the noisy latent as `y` |
| tracking video 2 | future-visible-point tracking video | VAE-encoded (48 ch), concatenated to the noisy latent as `y` |
| prompt | text | umT5 encoder (unchanged) |

The DiT itself is **not modified**. The config sets `in_dim = 144` (48 noisy + 48 tracking 1 + 48 tracking 2)
and the upstream `Wan2_2Transformer3DModel.from_pretrained` already expands `patch_embedding` for extra input
channels (it copies the pretrained 48 channels and zero-initialises the new 96). Training then updates all
parameters with a 10x larger learning rate on `patch_embedding`.

Where the inputs come from in the full pipeline:

* **Training data**: tracking videos for real MiraData clips, rendered by the `tracking_video` module
  (`first_frame_tracking_video.mp4`, `future_visible_tracking_video.mp4`).
* **Inference**: tracking videos rendered from the simulated rollout by the `scene_to_tracking` module
  (`visible_tracking_das.mp4`, `future_visible_tracking_das.mp4`) plus `first_frame_1280x704.png`.

Contents of this directory:

```
video_generation/
├── README.md
├── requirements.txt                 # upstream VideoX-Fun requirements + versions we used (comments)
├── patches/
│   └── videox_fun_dual_control.patch          # git diff 54bf97a -- videox_fun/data/dataset_image_video.py
├── videox_fun_overrides/
│   └── videox_fun/data/dataset_image_video.py # the same file with the patch already applied
├── config/wan2.2/wan_dual_control_5b.yaml     # in_dim = 144
├── scripts/
│   ├── train_dual_control.py        # training script (derived from upstream scripts/wan2.2_fun/train_control.py)
│   ├── train_dual_control.sh        # exact launcher of the released run ("planF")
│   └── make_metadata_miradata2.py   # builds the training metadata json
└── examples/
    └── predict_dual_control.py      # inference script
```

## Upstream repository, commit and patch

The code is an add-on to **[aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)** at commit
**`54bf97a`** (`54bf97ab661290a55d4ae91da3c0239a95eb6445`, 2026-04-07, Apache-2.0). We do not redistribute the
upstream package. The only change to an upstream file is in `videox_fun/data/dataset_image_video.py`
(`ImageVideoControlDataset` reads an optional second control video `control_file_path_2`, sampled with the same
frame indices as the main video, and returns it as `control_pixel_values_2`; zeros are used when the field is
missing). All other files are new.

### Installation

```bash
git clone https://github.com/aigc-apps/VideoX-Fun.git
cd VideoX-Fun
git checkout 54bf97a

# 1) patch the dataset (equivalently: copy videox_fun_overrides/videox_fun/data/dataset_image_video.py over the original)
git apply /path/to/release/video_generation/patches/videox_fun_dual_control.patch

# 2) copy the new files into the places the scripts expect (they add the repo root to sys.path)
REL=/path/to/release/video_generation
cp $REL/config/wan2.2/wan_dual_control_5b.yaml       config/wan2.2/
cp $REL/scripts/train_dual_control.py                scripts/wan2.2_fun/
cp $REL/scripts/train_dual_control.sh                scripts/wan2.2_fun/
cp $REL/scripts/make_metadata_miradata2.py           scripts/wan2.2_fun/
cp $REL/examples/predict_dual_control.py             examples/wan2.2_fun/
```

All commands below are run **from the root of this patched VideoX-Fun checkout**.

## Dependencies / Environment

* Python 3.10, CUDA 12.4. Versions we used: torch 2.5.1+cu124, torchvision 0.20.1, diffusers 0.37.1,
  transformers 5.5.0, accelerate 1.13.0, decord 0.6.0, opencv-python 4.13, omegaconf 2.3.0,
  flash-attn 2.7.4.post1 (optional). The full list is at the bottom of `requirements.txt`.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt      # upstream requirement list (unchanged)
```

* Base model: **[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)** (original, non-diffusers layout:
  `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, `google/umt5-xxl/`, transformer shards in the repo root).

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /path/to/Wan2.2-TI2V-5B
```

* Hardware: training used 2x NVIDIA H200 (141 GB), DDP + bf16 + gradient checkpointing (no DeepSpeed / FSDP).
  Inference at 1280x704x121 fits on one 80 GB GPU (about 40 GB peak in our measurement with CFG on). With two visible
  GPUs the script puts the DiT on `cuda:0` and the VAE / T5 on `cuda:1`.

## Checkpoints

| checkpoint | content | size |
|---|---|---|
| `checkpoints/video_generation/checkpoint-10500/transformer/` | fine-tuned `Wan2_2Transformer3DModel` (bf16, `in_dim = 144`), 2 safetensors shards + index + `config.json` | 10.0 GB (9.32 GiB) |

Only the transformer is released; the VAE and text encoder are the unmodified ones from the base model.
File list, sha256 and the training provenance are in `checkpoints/video_generation/MANIFEST.md`.

**Path convention**: `predict_dual_control.py` appends `/transformer` to `--transformer_path`
(`os.path.join(args.transformer_path, "transformer")`). Pass the **parent** directory, i.e.
`--transformer_path checkpoints/video_generation/checkpoint-10500`, not `.../checkpoint-10500/transformer`.

## Data format

Training uses **real MiraData clips only** (no simulated data). Each clip is 121 frames, 24 fps, 1280x704, and the two
tracking videos have the same length / fps / resolution (the dataset samples the three videos with the same frame indices).

```
<data_root>/                          # e.g. miradata2/
├── videos/<id>.mp4                   # RGB clip
├── images/<id>.png                   # first frame (only used for validation)
├── videos.txt                        # one line per clip, e.g. "videos/00000.mp4"
├── prompt.txt                        # caption, line-aligned with videos.txt
├── tracking_4999/<id>/first_frame_tracking_video.mp4      # tracking video 1  (from the tracking_video module)
│                     └ future_visible_tracking_video.mp4   # tracking video 2  (from the tracking_video module)
└── metadata_dual_control_4999.json
```

Metadata (`ImageVideoControlDataset` format), one entry per clip; paths are relative to `<data_root>`:

```json
[
  {"file_path": "videos/00000.mp4",
   "text": "A person riding a motorcycle in a video game.",
   "type": "video",
   "control_file_path":   "tracking_4999/00000/first_frame_tracking_video.mp4",
   "control_file_path_2": "tracking_4999/00000/future_visible_tracking_video.mp4"}
]
```

Generate it with:

```bash
python scripts/wan2.2_fun/make_metadata_miradata2.py \
    --root /path/to/miradata2 \
    --tracking-subdir tracking_4999 \
    --out /path/to/miradata2/metadata_dual_control_4999.json
# options: --verify-videos --min-frames 121, --exclude-folders 00000,00118 (e.g. to keep validation clips out),
#          --strict, --dry-run; an audit report is written next to the json (--report-json to change it)
```

A folder is kept only if it contains exactly the two canonical mp4 files, `videos/<id>.mp4` exists and `<id>` has a
caption in `prompt.txt`. Our released model was trained on `metadata_dual_control_4999.json` with **4775 entries**.

## Usage

### Inference (released checkpoint)

Default setting (CFG 3.0, 30 steps):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/wan2.2_fun/predict_dual_control.py \
    --base_model_path /path/to/Wan2.2-TI2V-5B \
    --config_path config/wan2.2/wan_dual_control_5b.yaml \
    --transformer_path /path/to/checkpoints/video_generation/checkpoint-10500 \
    --tracking_video_1 /path/to/scene/visible_tracking_das.mp4 \
    --tracking_video_2 /path/to/scene/future_visible_tracking_das.mp4 \
    --start_image /path/to/scene/first_frame_1280x704.png \
    --prompt "The wooden ball rolls slowly to the right and grazes the side of the candle; the candle spins and tips over at an angle while the ball deflects off its original trajectory." \
    --negative_prompt "" \
    --guidance_scale 3.0 \
    --num_inference_steps 30 \
    --seed 2048 \
    --width 1280 --height 704 --video_length 121 --fps 24 \
    --output_path outputs/scene_cfg3.mp4
```

No-CFG variant (one forward pass per step; the script disables CFG when `--guidance_scale <= 1.0`):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/wan2.2_fun/predict_dual_control.py \
    --base_model_path /path/to/Wan2.2-TI2V-5B \
    --config_path config/wan2.2/wan_dual_control_5b.yaml \
    --transformer_path /path/to/checkpoints/video_generation/checkpoint-10500 \
    --tracking_video_1 /path/to/scene/visible_tracking_das.mp4 \
    --tracking_video_2 /path/to/scene/future_visible_tracking_das.mp4 \
    --start_image /path/to/scene/first_frame_1280x704.png \
    --prompt "<prompt>" \
    --guidance_scale 1.0 \
    --num_inference_steps 30 \
    --seed 2048 \
    --width 1280 --height 704 --video_length 121 --fps 24 \
    --output_path outputs/scene_cfg1.mp4
```

Arguments of `predict_dual_control.py`:

| argument | default in script | value used above |
|---|---|---|
| `--base_model_path` | required | Wan2.2-TI2V-5B directory |
| `--config_path` | `config/wan2.2/wan_dual_control_5b.yaml` | same |
| `--transformer_path` | `None` (= base model with zero-initialised tracking channels) | parent dir of `transformer/` |
| `--tracking_video_1` / `--tracking_video_2` / `--start_image` / `--prompt` | required | see above |
| `--negative_prompt` | `""` | `""` |
| `--guidance_scale` | 5.0 | **3.0** (default setting) or **1.0** (no CFG) |
| `--num_inference_steps` | 50 | **30** |
| `--seed` | 42 | 2048 |
| `--width` / `--height` / `--video_length` / `--fps` | 1280 / 704 / 121 / 24 | same |
| `--output_path` | `output_dual_control.mp4` | any |

Note that the script defaults for `--guidance_scale` (5.0) and `--num_inference_steps` (50) were left unchanged;
pass the values above explicitly.

The scene inputs are produced by the `scene_to_tracking` module (`visible_tracking_das.mp4`,
`future_visible_tracking_das.mp4`; its `--suffix` option appends a suffix to these names) and by the
`scene_reconstruction` module (`first_frame_1280x704.png`, prompt). Tracking video 1 in the training data is the
DAS-style rendering (`visible_tracking_das` -> `first_frame_tracking_video.mp4` in the `tracking_video` module), so
`visible_tracking_das.mp4` is the matching input; the `visible_tracking_pred_track` rendering was also tried
(it is the one used for the simulated validation sample during training).

### Training (reproduces the released run)

```bash
bash scripts/wan2.2_fun/train_dual_control.sh \
    --model_dir /path/to/Wan2.2-TI2V-5B \
    --data_root /path/to/miradata2 \
    --train_meta /path/to/miradata2/metadata_dual_control_4999.json \
    --output_dir output_dir_planF_10x_lr_ratio
# resume:          add  --resume latest   (or --resume checkpoint-9000)
# no validation:   add  --no_validation
# simulated validation sample (6th sample of the paper run): add --eval_scene_dir /path/to/scene
#   (expects visible_tracking_pred_track_1.5s_fixed.mp4, future_visible_tracking_das_1.5s_fixed.mp4 and
#    first_frame_1280x704.png, i.e. scene_to_tracking run with --suffix _1.5s_fixed)
```

The launcher calls `accelerate launch --num_processes=2 --mixed_precision=bf16 scripts/wan2.2_fun/train_dual_control.py`
with the following recipe (identical to the released run):

| hyper-parameter | value |
|---|---|
| init | Wan2.2-TI2V-5B, `patch_embedding` expanded 48 -> 144 input channels, new channels zero-initialised |
| trainable | all parameters: `--trainable_modules patch_embedding` (lr 2e-4) and `--trainable_modules_low_learning_rate .` (lr 2e-4 / `--low_lr_ratio 10` = 2e-5) |
| optimizer | AdamW, betas (0.9, 0.999), weight decay 3e-2, eps 1e-10 |
| schedule | `constant_with_warmup`, 100 warmup steps |
| gradient clipping | `--max_grad_norm 0.5` (script defaults `--initial_grad_norm_ratio 5`, `--abnormal_norm_clip_start 1000`) |
| batch | 1 per GPU x 2 GPUs, gradient accumulation 1 |
| video | 121 frames, stride 1, `--video_sample_size 640 --token_sample_size 640`, `--enable_bucket --random_hw_adapt --training_with_video_token_length` |
| loss weighting | `--weighting_scheme none`, `--uniform_sampling`, `--boundary_type full` |
| dropout | `--tracking_drop_ratio 0.0` (per-track), `--text_drop_ratio 0.3` (empty caption) |
| precision | bf16, gradient checkpointing |
| seed | 42 |
| checkpointing | every 500 steps, keep 5; validation every 500 steps (30 steps, no CFG, 121 frames) |
| released checkpoint | step **10500** (about 1 hour per 500 steps on 2x H200 including validation) |

Other options of `train_dual_control.py` are the upstream `train_control.py` options plus `--tracking_drop_ratio`,
`--low_lr_ratio`, `--text_drop_ratio`, `--validation_paths_2` and `--validation_start_images`.

## Notes / Known limitations

* **Training vs. inference resolution.** Training uses a 640-level token budget with aspect-ratio buckets
  (`--video_sample_size 640 --token_sample_size 640 --random_hw_adapt`), while inference is run at 1280x704.
  This is the setting of the released model; we did not change it.
* **Validation clips are in the training set.** The five MiraData validation clips (`00000`, `00118`, `4014053.0`,
  `4167192.0`, `4266006.0`) are also entries of `metadata_dual_control_4999.json`; validation images during training
  are therefore not held-out. `make_metadata_miradata2.py --exclude-folders` can be used to exclude them.
* **The released run was resumed several times** (step 0 -> 1500 -> 3500 -> 6000 -> 9000 -> 10500). All runs on the
  lineage of checkpoint-10500 used the hyper-parameters above; `--resume_from_checkpoint` was the only argument that
  changed between them. Resuming restores the optimizer / scheduler state, so results of a single uninterrupted run
  may differ slightly. Checkpoint-10500 was chosen by the authors from the saved checkpoints (training reached 11000).
* One training clip (`4219364.0`) has an unreadable tracking video; the dataset logs a read error and resamples another
  index, so it is effectively skipped.
* If an entry has no `control_file_path_2`, the patched dataset silently uses an all-zero second control video.
* The validation loop in `train_dual_control.py` is hard-coded to 30 denoising steps, no CFG and 121 frames.
* The MiraData videos themselves are not redistributed; see the MiraData project for download terms. Only the
  metadata format and the generation script are provided.
* License: the upstream VideoX-Fun code is Apache-2.0; the base model Wan2.2-TI2V-5B is Apache-2.0.
