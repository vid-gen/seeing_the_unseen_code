# tracking_video — 3D-tracking control videos for training

## Overview

This module turns a plain RGB video into the **two tracking control videos** used to train the
video-generation model:

| Output file | Content |
|---|---|
| `first_frame_tracking_video.mp4` | *Visible* tracks: a dense grid of points sampled on frame 0 and tracked through the clip. |
| `future_visible_tracking_video.mp4` | *Future-visible* tracks: points that are **not** visible in frame 0 (they enter the view later or are dis-occluded) and are tracked from the first frame in which they appear. |

Both videos are rendered in the Diffusion-as-Shader (DaS) style: black background, each point is drawn
as a solid square whose RGB color encodes its (normalized) image x / image y / inverse depth, fixed
per point over time, drawn far-to-near. Output resolution and frame count equal the input video
(e.g. 1280x704, 121 frames, 24 fps for our training clips). The file names are the ones expected by
the metadata script of the `video_generation` module.

The tracker is [SpatialTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) (VGGT4Track front end
for depth / camera poses + the offline SpaTrack2 tracker). Two stages:

1. **Stage 1 – visible** (`run_visible_tracking.py`): `grid_size x grid_size` query points on frame 0
   (model resolution), tracked with SpaTrackerV2 in batches of `min(batch_size, vo_points)` queries.
   No point-level filtering; points are rendered on frames where `vis > 0.9`.
2. **Stage 2 – future-visible** (`run_future_visible_tracking.py`): for every frame `t >= 1`, a
   `grid_size_fv x grid_size_fv` grid is lifted to 3D with the predicted point map / camera pose and
   re-projected into frame 0 (`sample_future_visible_queries_inverse` in `utils.py`). A point is
   future-visible if it projects outside frame 0, is occluded there
   (`proj_depth > depth0 * (1 + depth_margin)`), or lies behind the camera; its query frame is the first
   such frame. Candidates within `stage1_filter_radius` pixels of a Stage-1 point with `vis > 0.5` at that
   frame are dropped. The remaining queries are tracked with SpaTrackerV2, then a quality filter keeps a
   track if some frame has `vis > vis_threshold` and `conf > conf_threshold`. Visibility before the
   query frame is set to 0.

```
<video>.mp4 ─► VGGT4Track (depth, poses, intrinsics)
             ├─► Stage 1: frame-0 grid ─► SpaTrackerV2 ─► first_frame_tracking_video.mp4
             └─► Stage 2: inverse projection ─► SpaTrackerV2 ─► quality filter ─► future_visible_tracking_video.mp4
```

### Files

| File | Role |
|---|---|
| `run_all.py` | Single-video entry point (loads models once, runs Stage 1 + Stage 2). |
| `run_visible_tracking.py` | Stage 1 (function `run_visible_tracking`, also a CLI). |
| `run_future_visible_tracking.py` | Stage 2 (function `run_future_visible_tracking`, also a CLI that reads `stage1_results.npz`). |
| `utils.py` | Model loading, video loading, future-visible query sampling, quality filter, result saving. |
| `das_visualizer.py` | DaS-style renderer (writes mp4v, then re-encodes to H.264 with `ffmpeg`). |
| `batch_run_v2.py` | Multi-GPU batch launcher (splits the video list across GPUs/workers, merges logs). |
| `batch_worker_v2.py` | Batch worker (one per GPU slot; resumable; keeps only the two final mp4s). |
| `scripts/run_batch.sh` | The batch job script used to generate the training set (cluster-specific parts removed). |
| `patches/preserve_query_order.patch` | Required patch for SpaTrackerV2 (see below). |

`tracking_video` is a namespace package: the directory must be placed **inside the SpaTrackerV2
checkout** (each script inserts its parent directory, i.e. the SpaTrackerV2 root, into `sys.path` and
imports `models.SpaTrackV2.*` from there).

## Dependencies / Environment

* Python 3.10 (the environment we used; upstream recommends 3.11), CUDA GPU (code uses `"cuda"`).
* Versions we used: `torch 2.4.1+cu121`, `torchvision 0.19.1+cu121`, `xformers 0.0.28`,
  `pycolmap 3.11.1`, `pyceres 2.4`, `kornia 0.8.1`, `utils3d` (commit pinned in upstream
  `requirements.txt`), `decord 0.6.0`, `opencv-python-headless`, `numpy 2.2.6`, `huggingface_hub`, `tqdm`.
  Installing the upstream `requirements.txt` is sufficient.
* `ffmpeg` executable on `PATH` (used to re-encode to H.264/yuv420p; if missing the mp4v file is kept).
* Optional: `HF_HOME` to choose the HuggingFace cache directory for the downloaded weights.

## Upstream repository and patch

Upstream: `https://github.com/henry123-boy/SpaTrackerV2`, commit **`7e12274`** (`add_eval`, 2026-02-27).

`patches/preserve_query_order.patch` modifies two upstream files:

* `models/SpaTrackV2/models/predictor.py` — `Predictor.forward` gets a `preserve_query_order=False`
  argument and forwards it to `forward_stream`.
* `models/SpaTrackV2/models/SpaTrack.py` — `forward_stream` accepts `preserve_query_order`; when `True`
  it skips the "drop uncertain queries + random re-pad" step so that the order of the N query points is
  preserved (our code un-sorts the outputs by index), and it guards the re-pad against zero kept queries.

The patch is **required**: both stages call `Predictor.forward(..., preserve_query_order=True)`, which
raises `TypeError` on unpatched upstream code. (Our local checkout also contained a cosmetic change to
`models/SpaTrackV2/utils/visualizer.py` that only affects the debug `*_pred_track.mp4` videos; it is not
included and does not affect the two DaS control videos.)

```bash
git clone https://github.com/henry123-boy/SpaTrackerV2.git
cd SpaTrackerV2
git checkout 7e12274
git apply /path/to/release/tracking_video/patches/preserve_query_order.patch
# install the environment following the upstream README, e.g.
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
# put this module inside the checkout (copy or symlink)
cp -r /path/to/release/tracking_video ./tracking_video
```

## Checkpoints

No checkpoint is shipped with this module. All weights are the public SpatialTrackerV2 releases and are
downloaded automatically by `from_pretrained` on first use (cached under `HF_HOME`):

| Weight | HuggingFace repo | Used for |
|---|---|---|
| VGGT4Track front end | [`Yuxihenry/SpatialTrackerV2_Front`](https://huggingface.co/Yuxihenry/SpatialTrackerV2_Front) | depth, camera poses, intrinsics |
| SpaTrack2 offline tracker | [`Yuxihenry/SpatialTrackerV2-Offline`](https://huggingface.co/Yuxihenry/SpatialTrackerV2-Offline) | point tracking (`--track_mode offline`, used in the paper) |
| SpaTrack2 online tracker | [`Yuxihenry/SpatialTrackerV2-Online`](https://huggingface.co/Yuxihenry/SpatialTrackerV2-Online) | only for `--track_mode online` (not used) |

## Data format

* **Input (RGB, used in the paper)**: `<data_dir>/<video_name>.mp4`. The batch launcher processes every
  `*.mp4` in `--data_dir`. Our training clips are 1280x704, 121 frames, 24 fps. `--fps` is a **frame
  stride** (`frames[::fps]`), not a frame rate; use `1` to keep all frames.
* Input (RGBD, supported but not used): `<data_dir>/<video_name>.npz` with keys `video` (T,3,H,W in
  [0,1]), `depths`, `intrinsics`, `extrinsics`.
* **Batch output** (`batch_run_v2.py`): `<output_base_dir>/<video_name>/first_frame_tracking_video.mp4`
  and `future_visible_tracking_video.mp4`; logs in `<output_base_dir>/logs/`
  (`video_list_w*.txt`, `batch_log_w*.jsonl`, `batch_log_all.jsonl`, `batch_summary.txt`, `stdout_w*.log`).
* **Single-video output** (`run_all.py`, in `--output_dir`): `visible_tracking_das.mp4` and
  `future_visible_tracking_das.mp4` (the two control videos, i.e. the files that the batch worker renames
  to `first_frame_tracking_video.mp4` / `future_visible_tracking_video.mp4`), debug videos
  `*_pred_track.mp4`, `result_visible.npz` / `result_future_visible.npz` (TAPIP3D-style), and the
  intermediates `stage1_results.npz`, `future_visible_meta.npz`.

## Usage

All commands are run from the SpaTrackerV2 root (the directory that contains `models/` and `tracking_video/`).
Use absolute paths for data/output directories.

**Batch generation (exact settings used to produce the training data, 4 GPUs):**

```bash
cd /path/to/SpaTrackerV2
python tracking_video/batch_run_v2.py \
    --num_gpus=4 --workers_per_gpu=1 --cpu_threads_per_worker=24 \
    --data_dir=/path/to/videos --output_base_dir=/path/to/tracking \
    --fps=1 --das_fps=24 --grid_size=70 --grid_size_fv=90 --depth_margin=5.0 \
    --das_point_size=2 --tracker=v2 --track_mode=offline --vo_points=384 \
    --conf_threshold_depth=0.0
```

`--vis_threshold 0.9`, `--conf_threshold 0.9`, `--batch_size 2500`, `--dilate_pixels 1` and
`--stage1_filter_radius 5.0` are left at their defaults. The same run as a job script:

```bash
SPATRACKER_ROOT=/path/to/SpaTrackerV2 DATA_DIR=/path/to/videos OUTPUT_DIR=/path/to/tracking \
    bash /path/to/SpaTrackerV2/tracking_video/scripts/run_batch.sh
```

**Single video (same settings; note that `run_all.py` has different defaults, so pass them explicitly):**

```bash
cd /path/to/SpaTrackerV2
python tracking_video/run_all.py --data_type=RGB --data_dir=/path/to/videos --video_name=00000 \
    --fps=1 --tracker v2 --track_mode offline \
    --grid_size 70 --grid_size_fv 90 --depth_margin 5.0 --conf_threshold_depth 0.0 \
    --das_point_size 2 --das_fps 24 --output_dir=/path/to/output/00000
```

Stage 1 / Stage 2 can also be run separately (`run_visible_tracking.py`, then
`run_future_visible_tracking.py --stage1_dir=<output_dir of stage 1>`), but `run_all.py` is the tested path.

## Notes / Known limitations

* **Stage 2 with no future-visible point**: `run_future_visible_tracking` returns `None` without writing
  `future_visible_tracking_das.mp4` when no candidate survives the inverse-projection test. The batch worker
  still logs the video as `success`, but because only `first_frame_tracking_video.mp4` exists, the resume
  check treats it as unfinished and the video is reprocessed on every re-run. This behaviour is kept as is.
* **Only the V2 path is released.** `--tracker v1` / `hybrid` remain in the CLI choices of `run_all.py` /
  `run_visible_tracking.py` / `run_future_visible_tracking.py`, but the SpaTracker-V1 adapter is not part of
  this release: `utils._load_models_v1/_hybrid` raise `NotImplementedError` ("v1/hybrid trackers are not
  included in this release; use --tracker v2"). The paper uses `--tracker v2` only.
* `batch_run_v2.py` defaults already match the paper settings. `run_all.py`, `run_visible_tracking.py`
  and `run_future_visible_tracking.py` default to `--vo_points 384` as well, but keep other original
  (different) defaults, e.g. `grid_size=50`, `depth_margin=1.0`, `das_fps=8`, `das_point_size=0`,
  `data_type=RGBD`.
* Stage 1 does not apply the vis/conf quality filter (the `vis_threshold`/`conf_threshold` arguments are
  accepted but unused there); Stage 2 does.
* With `depth_margin=5.0` a point is considered occluded in frame 0 only if its re-projected depth exceeds
  6x the frame-0 depth, so future-visible points come mostly from the out-of-view test.
* The colors of the future-visible video are normalized over the future-visible point set using each
  point's position at its query frame, i.e. they are not in the same color frame as the visible video.
* GPU selection in `batch_worker_v2.py` uses `torch.cuda.set_device(gpu_id)` (the `--gpu_id` help text
  mentions `CUDA_VISIBLE_DEVICES`, and the logged device name is that of GPU 0; logging only).
* `batch_worker_v2.py` deletes all intermediate files (`*_pred_track.mp4`, `*.npz`) after each video.
* Some code comments/log strings are in Chinese and mention `miradata1`, the name of the training-video
  folder on our cluster.
