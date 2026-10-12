# scene_to_tracking: physics rollout + meshes -> tracking control videos

## Overview

This module converts the output of the reconstruction + physics-simulation stage
([`../scene_reconstruction`](../scene_reconstruction)) into 2D point-tracking videos.
Two of these videos are the control inputs of the dual-control video generator in
[`../video_generation`](../video_generation).

What the pipeline does:

```
first frame + SAM masks + posed 3D meshes + GNS particle rollout
  -> grid-sample tracking points on the image, snap them to the nearest
     front-facing mesh vertex (KD-tree)                      (mesh_sampler.py)
  -> per-step rigid transform (R, t) of every object from its rollout
     particles via SVD, applied to the mesh vertices         (motion_tracker.py, utils.py)
  -> visibility = viewport check + back-face culling +
     depth-map based inter-object / self occlusion           (visibility.py)
  -> resample the simulation steps to the target video length / FPS
                                                             (time_resampler.py)
  -> render two styles:
       - "pred_track" (SpaTracker style): rainbow points + short trails on the
         grayscale first frame                               (rainbow_visualizer.py)
       - "das" (Diffusion-as-Shader style): RGB = normalised (x, y, inverse
         depth) at frame 0, black background, painter's-algorithm depth order
                                                             (das_visualizer.py)
```

It produces two kinds of tracks:

- **Visible tracking.** Points on a `grid_size x grid_size` image grid. Each point is either
  matched to a mesh vertex that is visible in frame 0, or treated as a static background point.
- **Future-visible tracking.** Mesh vertices that are hidden in frame 0 (back-facing or
  outside the viewport) and become visible later. For each vertex, the script records the first
  simulation step at which it is visible (its query frame). It then keeps at most one vertex per
  `dense_spacing x dense_spacing` pixel cell, preferring the earliest query frame. Before its
  query frame, a point is invisible and its position is frozen at the query-frame position.

The method is described in the paper section "Tracking Video Generation".

| File | Role |
|---|---|
| `generate_tracking_videos.py` | Entry point (CLI). Runs phase 1 (visible) and/or phase 2 (future visible). |
| `data_loader.py` | Loads rollout, meshes, masks, camera, coordinate transform and first frame; matches masks to meshes. |
| `mesh_sampler.py` | Grid sampling for visible points; query-frame search + cell downsampling for future-visible points. |
| `motion_tracker.py` | SVD rigid transform per object per simulation step; projection to pixels. |
| `visibility.py` | Viewport / back-face visibility and depth-map occlusion between objects. |
| `time_resampler.py` | Maps simulation steps to video frames (linear particle interpolation); remaps query frames. |
| `rainbow_visualizer.py` | SpaTracker-style renderer. |
| `das_visualizer.py` | DaS-style renderer. |
| `utils.py` | `calc_rigid_transform`, `scene_to_physion`, `project_physion_to_pixel`. |

## Dependencies / Environment

The module needs only CPU and plain Python packages. It does not use a GPU or a
deep-learning framework. The authors used Python 3.10.

```bash
pip install -r requirements.txt
```

`requirements.txt` pins numpy, scipy, trimesh, opencv-python, matplotlib (< 3.11, see Notes),
Pillow, imageio and imageio-ffmpeg. imageio-ffmpeg is needed to write H.264 mp4 files.

The scripts use flat, non-package imports (`from utils import ...`). The entry script adds
its own directory to `sys.path`, so run it as a script, e.g.
`python scene_to_tracking/generate_tracking_videos.py ...`. You can run it from any working
directory.

## Upstream repository / patches

None. All code in this module is original and does not depend on third-party repositories.
There are no patches.

## Checkpoints

None. This module has no learned components. The GNS rollout it consumes is produced by
`../scene_reconstruction` with the GNS checkpoint from `../gns_training`.

## Data format

### Input: `--data_dir` (one scene)

```
<data_dir>/
├── first_frame_1280x704.png        # first frame (see "first-frame lookup" below)
├── prompt.txt                      # text prompt for ../video_generation (NOT read here)
├── pipeline_config.json            # written by ../scene_reconstruction (NOT read here)
├── rollout/
│   ├── rollout_positions.npz       # keys step_0 ... step_{T-1}, each (N_particles, 3), Physion space
│   ├── rollout_info.json           # (not read here)
│   └── camera_physion.json         # position, intrinsics_norm, image_width, image_height
├── scene_3dgs/
│   ├── object_{i}_mesh_posed.ply   # posed triangle mesh of object i (scene space)
│   └── mask_{i}.png                # SAM mask of object i
└── trial/
    ├── coord_transform.json        # center_x, ground_y, center_z, scale, ground_clearance,
    │                               # [scene_scale_factor], [domino_rotation_matrix]
    └── phases_dict.pkl             # n_objects, instance_idx (particle range of each object)
```

The script reads only the files listed above, apart from the three marked "not read here".

Correspondence with the output of `../scene_reconstruction`
(`pipeline_3dgs_to_physics.py --image <img> -o <output_dir> ...`):

| Needed here | Produced by scene_reconstruction | Notes |
|---|---|---|
| `rollout/rollout_positions.npz`, `rollout/rollout_info.json` | GNS rollout step (`run_rollout.py`) -> `<output_dir>/rollout/` | `--rollout-steps 167` gives 168 steps (`step_0` ... `step_167`). |
| `rollout/camera_physion.json` | camera restoration step -> `<output_dir>/rollout/` | Do **not** pass `--no-camera` to the pipeline, or this file is missing. |
| `scene_3dgs/object_{i}_mesh_posed.ply`, `scene_3dgs/mask_{i}.png` | `--image` mode: `<output_dir>/scene_3dgs/` | With `--scene-dir <dir>` mode the meshes stay in `<dir>`. Copy or symlink it to `<output_dir>/scene_3dgs`. |
| `trial/coord_transform.json`, `trial/phases_dict.pkl` | `build_physion_input.py` -> `<output_dir>/trial/` | Defines the scene -> Physion transform and the particle ranges per object. |
| `first_frame_1280x704.png` | **not produced**. It is the image you passed to `--image`. | Copy the input image into `<output_dir>`. It must have the same resolution as `camera_physion.json` (`image_width x image_height`, 1280x704 for `../video_generation`). |
| `prompt.txt` | **not produced** | Write it yourself. It is used only as `--prompt` of `../video_generation`. |

In short, point `--data_dir` at the `-o/--output-dir` of `pipeline_3dgs_to_physics.py` after
copying the first frame into it.

First-frame lookup (`data_loader.py`). The loader uses the first of these files that exists:
`1280_704_image.png`, `first_frame_1280x704.png`, `first_frame.png`.

Number of objects. The loader uses objects `0 .. K-1`, where `K` is the number of
consecutive `object_{i}_mesh_posed.ply` files that exist, capped at `phases_dict['n_objects']`.
Mesh `i` must correspond to particle range `instance_idx[i]:instance_idx[i+1]`. A `mask_{i}.png`
must exist for each object used. The loader re-pairs masks with meshes greedily, by the
horizontal centre of each mask's bounding box and of each mesh's projection.

Coordinate frames: scene (mesh space) -> `utils.scene_to_physion` -> Physion
(simulation space) -> `utils.project_physion_to_pixel` -> pixels.

### Output: `--output_dir` (defaults to `--data_dir`)

| File | Style | Frames / FPS | Used by `../video_generation` |
|---|---|---|---|
| `visible_tracking_das{suffix}.mp4` | DaS (RGB-coded position / inverse depth, black bg) | `--video_frames` @ `--video_fps` | **yes**: `--tracking_video_1` |
| `future_visible_tracking_das{suffix}.mp4` | DaS | `--video_frames` @ `--video_fps` | **yes**: `--tracking_video_2` |
| `visible_tracking_pred_track{suffix}.mp4` | SpaTracker rainbow on grayscale first frame | `--video_frames` @ `--video_fps` | no (visualisation only) |
| `future_visible_tracking_pred_track{suffix}.mp4` | SpaTracker rainbow | `--video_frames` @ `--video_fps` | no (visualisation only) |

- All four videos are H.264 (`libx264`, `yuv420p`, CRF 18).
- DaS videos have the camera resolution (`image_width x image_height`). Rainbow videos have the
  first-frame resolution.
- With the defaults, all four videos have 121 frames at 24 FPS. This matches
  `--video_length 121 --fps 24 --width 1280 --height 704` of `../video_generation`.
- Training pairs for `../video_generation` were built with the same order: the
  "first-frame visible" DaS video is control 1 and the "future visible" DaS video is control 2.
- If a scene has no future-visible vertices, phase 2 prints
  `No future visible points found, skipping.` and writes no `future_visible_*` files.

## Usage

Generate all four videos. These are the settings used by the authors (grid 70, dense spacing 4,
24 FPS, 121 frames). Every flag below exists in the current argparse of
`generate_tracking_videos.py`.

```bash
python scene_to_tracking/generate_tracking_videos.py \
  --data_dir   /path/to/scene \
  --output_dir /path/to/scene \
  --grid_size 70 --dense_spacing 4 \
  --video_fps 24 --video_frames 121
```

Only the visible-tracking videos (phase 1), or only the future-visible ones (phase 2):

```bash
python scene_to_tracking/generate_tracking_videos.py --data_dir /path/to/scene --phase 1
python scene_to_tracking/generate_tracking_videos.py --data_dir /path/to/scene --phase 2
```

Feed the two DaS videos to the video generator (see `../video_generation/README.md` for the
remaining flags, checkpoint and guidance settings):

```bash
python examples/wan2.2_fun/predict_dual_control.py \
  --tracking_video_1 /path/to/scene/visible_tracking_das.mp4 \
  --tracking_video_2 /path/to/scene/future_visible_tracking_das.mp4 \
  --start_image      /path/to/scene/first_frame_1280x704.png \
  --prompt "$(cat /path/to/scene/prompt.txt)" \
  ...
```

### All arguments of `generate_tracking_videos.py`

| Argument | Default | Meaning |
|---|---|---|
| `--data_dir` | (required) | Scene directory (see Data format). |
| `--output_dir` | `= --data_dir` | Where the videos are written. Created if missing. |
| `--grid_size` | `70` | Visible tracking: `grid_size x grid_size` sample grid (70 -> 4900 points). |
| `--dense_spacing` | `4` | Future-visible tracking: cell size in pixels for downsampling. |
| `--video_fps` | `24` | FPS written to **all four** videos. |
| `--video_frames` | `121` | Number of output frames. |
| `--sim_dt` | `0.03` | Simulated seconds per rollout step. |
| `--sim_duration` | `None` | Use only the first N simulated seconds. `None` = whole rollout. |
| `--das_point_size` | `2` | Half-size of the DaS square splat (2 -> 5x5 px). |
| `--suffix` | `''` | Appended to every output filename before `.mp4`. |
| `--phase` | `0` | `0` = both, `1` = visible only, `2` = future visible only. |

Time resampling. The first `sim_duration` seconds of the rollout (default: all
`(T-1) * sim_dt` seconds) are stretched linearly onto `video_frames` frames. At each frame, the
particle positions are linearly interpolated between the two nearest simulation steps before
the SVD fit. A point is visible in a frame only if it is visible at both of those steps.
With the defaults (168 steps x 0.03 s = 5.01 s -> 121 frames @ 24 FPS = 5.04 s), playback is
approximately real time. A smaller `--sim_duration` gives slow motion.

### Example scene

The authors' example scene `collide_angle_ball_candle_glancing` ("The wooden ball rolls and
grazes the side of the candle; the candle spins and tips over at an angle while the ball
deflects off its original trajectory.") is **not** included. Its `rollout/ + scene_3dgs/ + trial/`
directories are about 280 MB, mostly from the 3DGS `.ply` files that this module does not read.
The subset this module actually reads is about 31 MB:

```
first_frame_1280x704.png (1.0 MB)
rollout/{rollout_positions.npz, camera_physion.json, rollout_info.json} (1.6 MB)
trial/{coord_transform.json, phases_dict.pkl}
scene_3dgs/{object_0_mesh_posed.ply (19 MB), object_1_mesh_posed.ply (10 MB), mask_0.png, mask_1.png}
```

The example has two reconstructed objects (ball, candle) plus the ground object that
`../scene_reconstruction` appends as the last object (`trial/build_info.json`: `n_objects = 3`).
It also has 430 particles, 168 rollout steps, a 1280x704 image and `ground_clearance = 0.1`.

The authors' copy of this scene also contains `scene_3dgs/object_2_*` and `mask_2.png` from an
earlier reconstruction. These files are older than `object_0/1_*`, and `mesh_info.json` lists
only 2 objects. Because of the loader rule described under "Stale meshes" below, the loader
would pick these files up. Leave them out if you share or rebuild the example.

To reproduce such a scene, run `../scene_reconstruction` on your own image, then run the command
above.

## Notes / Known limitations

- **Old command names.** Earlier internal notes used `--fps 10 --das_fps 24`. These flags no
  longer exist, and passing them makes argparse fail. The single flag `--video_fps` now sets the
  FPS of all four videos, including the rainbow videos, which used to be 10 FPS.
- **matplotlib.** `rainbow_visualizer.py` uses `matplotlib.cm.get_cmap`, which is deprecated
  since matplotlib 3.7 and scheduled for removal in 3.11. Use matplotlib < 3.11.
- **Rigid-body assumption.** Each object's motion is one SVD rigid transform fitted to its
  particles. Deformable motion is not represented.
- **Mask-to-mesh matching** uses only the horizontal centre. Objects with nearly the same
  horizontal centre may be paired with the wrong mask. Masks are used only for grid points that
  have no visible projected vertex within 3 px.
- **Query-frame search** (future-visible points) uses only back-face culling and the viewport
  test. Occlusion by other objects is applied later, in the visibility stage.
- **Stale meshes.** `phases_dict['n_objects']` also counts the ground object, which
  `../scene_reconstruction` appends as the last object. Ground is on by default; disable it with
  `--no-ground`. The loader therefore counts objects as consecutive `object_{i}_mesh_posed.ply`
  files and stops at the first missing one. If `scene_3dgs/` still contains
  `object_{i}_mesh_posed.ply` / `mask_{i}.png` from an earlier reconstruction with more objects,
  those stale files are loaded and paired with the ground particles. Before running, make sure
  `scene_3dgs/` contains exactly the current objects. The `objects` list in
  `scene_3dgs/mesh_info.json` gives the expected set.
- **Resolution.** The first frame must have the same resolution as `camera_physion.json`.
  Otherwise the rainbow and DaS videos come out at different sizes.
- Runtime is CPU-bound. The renderers draw points one at a time in Python loops.

### Known issues found by static review (kept as-is)

The code was reviewed statically while preparing this release. It is released as used for the
paper, so the issues below were **not** fixed. They have not been checked numerically.

1. **Grid-snap offset applied to future-visible points.** `motion_tracker.compute_all_frames`
   adds `grid_uv - tracks_2d[0]` to every frame. This is meant to snap visible points onto their
   grid position. For future-visible points, however, `grid_uv` is the projection at the query
   frame, so the added offset is `uv(query frame) - uv(frame 0)`. That offset can be large. The
   per-step viewport test (`visibility.compute_visibility`) and the occlusion depth-map lookup
   (`visibility.apply_inter_object_occlusion`) for these points are therefore evaluated at
   shifted pixel positions. The rendered positions come from `time_resampler.resample_tracks`,
   which recomputes projections without the offset.
2. **Phase 1 uses the same offset for visibility only.** For visible points the offset is just the
   snap residual (at most 3 px, unless a point falls back to a mask match). It is used in the
   visibility computation, but the rendered tracks are the un-snapped vertex projections.
3. **`time_resampler.remap_query_frames` does not clamp `sim_duration / sim_dt` to
   `n_sim_steps - 1`, while `build_time_mapping` does.** If `--sim_duration` is longer than the
   rollout, the remapped query frames come earlier than the time mapping implies. The default
   (`--sim_duration` unset) is consistent.
4. **With `--sim_duration`, the future-visible query-frame search still scans the whole
   rollout.** Points that first become visible after `sim_duration` are clipped to the last video
   frame.
5. **Stale meshes are picked up by the loader.** See "Stale meshes" above.
