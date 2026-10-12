# Scene Reconstruction and Physics Rollout (single image → 3D scene → GNS rollout)

## Overview

This module turns one RGB image and a text description of the motion into a particle-based
physics rollout. The rollout, together with the reconstructed meshes, is the input of the
`scene_to_tracking` module, which renders the control (tracking) videos for the video
generation model.

`pipeline_3dgs_to_physics.py` is the entry point. It runs these steps:

| Step | What it does | Main file(s) |
|---|---|---|
| 1 | Image → multi-object 3D scene. GPT Vision lists the objects, SAM3 segments them, SAM-3D-Objects reconstructs each object (Gaussians + mesh, with pose), and MoGe-2 estimates the camera intrinsics and the ground plane. | `auto_scene_pipeline.py` |
| – | Initial linear/angular velocities from the motion prompt (GPT, optional; can be given manually) | `pipeline_3dgs_to_physics.py` |
| 2 | Mesh → particles (`binvox -dc` voxelisation, the same as the Physion training preprocessing) → Physion trial format | `mesh_to_particles.py`, `build_physion_input.py`, `binvox_rw.py`, `mesh_thicken.py`, `gs_to_particles.py` |
| 3 | GNS autoregressive rollout with RANSAC rigid-body projection | `run_rollout.py` (+ `../gns_training`) |
| 4 | Diagnostic plots/GIFs (trajectories, centroids, velocities) | `visualize_rollout.py` |
| 5 | Camera restoration: project the rollout back into the input view (`camera_view.gif`, optical flow) | `camera_restore.py`, `visualize_rollout.py` |
| 5.5 | (`--wireframe`) mesh wireframe projected into the input view (`camera_view_wireframe.gif`) | `mesh_wireframe_vis.py`, `generate_ground_mesh.py` |
| 6/7 | Wan-Move track export / generation | **not included in this release** (see Notes) |

Files (all flat, imported by module name):

```
scene_reconstruction/
├── pipeline_3dgs_to_physics.py   # end-to-end entry point
├── auto_scene_pipeline.py        # step 1 (also runnable on its own)
├── build_physion_input.py        # step 2: Physion trial writer
├── mesh_to_particles.py          # step 2: mesh repair + binvox voxelisation
├── mesh_thicken.py               # step 2: optional thin-object thickening (--thicken)
├── gs_to_particles.py            # legacy 3DGS→particles path; still used for coordinate alignment
├── binvox_rw.py                  # third-party binvox reader (GPL-3, see Notes)
├── run_rollout.py                # step 3 (also runnable on its own)
├── visualize_rollout.py          # step 4/5
├── camera_restore.py             # step 5
├── mesh_wireframe_vis.py         # step 5.5
├── generate_ground_mesh.py       # step 5.5
├── patches/sam3d_objects.patch   # local changes to SAM-3D-Objects
└── requirements.txt
```

## Dependencies / Environment

The paper runs used Python 3.10, PyTorch 2.5.1 + CUDA 12.1, on a single GPU with at least 32 GB of memory
(this is the SAM-3D-Objects requirement).

1. **SAM-3D-Objects** (upstream code with a small local patch, see below). Follow its `doc/setup.md`
   to create the environment and download the checkpoints into `<SAM3D_ROOT>/checkpoints/hf/`
   (you need gated access on HuggingFace: `facebook/sam-3d-objects`). MoGe-2
   (`Ruicheng/moge-2-vitl-normal`) is loaded by the SAM-3D-Objects pipeline config and downloaded automatically.
2. **SAM3** for text-prompted segmentation is loaded with `transformers.Sam3Model.from_pretrained("facebook/sam3")`.
   It is a gated HuggingFace model. Request access and run `hf auth login` (or set `HF_TOKEN`) first.
   It needs `transformers>=5`. Set `HF_HOME` if you want a custom cache location.
3. Remaining Python packages:
   ```bash
   pip install -r requirements.txt
   pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   ```
4. **binvox** executable (closed-source freeware, not redistributable): download it from
   <https://www.patrickmin.com/binvox/>, then either put it on `$PATH` or set `BINVOX_PATH=/path/to/binvox`.
   binvox is run through `xvfb-run` (OpenGL depth carving), so `xvfb` and a Mesa `libGL` must be installed.
   The code sets `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1` (Debian/Ubuntu path; see Notes).
5. **GNS code**: the rollout imports `models.GNS_pyg` and `inference_pyg` from the sibling release module
   `../gns_training`. Set `PHYSION_PARTICLES_DIR` to use a different location.
6. **OpenAI API** (optional): GPT Vision object detection (`--gpt-model`, default `gpt-5.2`) and initial velocity
   inference (`--gpt-velocity-model`, default `gpt-5`). The key is read **only** from the environment:
   `export OPENAI_API_KEY=...`. Use `--no-gpt` to run without any OpenAI call (see Usage).

Environment variables used by this module:

| Variable | Required | Meaning |
|---|---|---|
| `SAM3D_ROOT` | yes for `--image` mode | Root of the SAM-3D-Objects checkout (contains `notebook/` and `checkpoints/hf/pipeline.yaml`). If it is not set, the parent directory of `scene_reconstruction/` is used, so you can also place this folder inside the SAM-3D-Objects checkout. |
| `OPENAI_API_KEY` | only when GPT is used | OpenAI key (never hard-coded) |
| `BINVOX_PATH` | if `binvox` is not on `$PATH` | Path to the binvox executable |
| `PHYSION_PARTICLES_DIR` | no | GNS code directory (default `../gns_training`) |
| `HF_HOME` | no | HuggingFace cache location |

## Upstream repository and patch

SAM-3D-Objects: <https://github.com/facebookresearch/sam-3d-objects>, commit
`81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd` (2026-01-06, "Merge pull request #134 ... post-optim").
The local copy we used differs from this commit in exactly two files (+3/−2 lines):

* `notebook/inference.py`: `Inference.__call__` gets a `with_layout_postprocess=False` argument and passes it
  to the pipeline (it was hard-coded to `False`). `auto_scene_pipeline.py` uses it for layout post-optimisation
  (enabled by default, disable with `--no-layout-optimize`).
* `sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py`: `backgrounds=bg_color.unsqueeze(0)`
  (adds the batch dimension that gsplat expects).

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
git checkout 81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd
git apply /path/to/release/scene_reconstruction/patches/sam3d_objects.patch
# then follow doc/setup.md (environment + checkpoints/hf)
export SAM3D_ROOT=$(pwd)
```

## Checkpoints

| What | Where | Source |
|---|---|---|
| GNS (Physion, trained without the Drape scenario) | `../checkpoints/gns/without_Drape/best_model.pth` (default of `--gns-ckpt`) | this release (module `gns_training`, see `checkpoints/gns/without_Drape/MANIFEST.md`) |
| SAM-3D-Objects | `$SAM3D_ROOT/checkpoints/hf/` | HuggingFace `facebook/sam-3d-objects` (gated) |
| SAM3 | HF cache | HuggingFace `facebook/sam3` (gated) |
| MoGe-2 | HF cache | HuggingFace `Ruicheng/moge-2-vitl-normal` (automatic) |

This module has no checkpoints of its own.

## Data format

### Input

One scene folder with the first frame and the motion prompt. The paper scenes are 1280×704:

```
<scene>/
├── first_frame_1280x704.png    # input image (the paper batch used the name 1280_704_image.png)
└── prompt.txt                  # motion description, e.g. "The red ball rolls right and hits the candle."
```

### Output (`-o <scene>`)

```
<scene>/
├── pipeline_config.json        # all CLI arguments of this run
├── scene_3dgs/                 # step 1
│   ├── object_<i>.ply, object_<i>_mesh.ply, object_<i>_mesh_posed.ply, object_<i>_pose.json
│   ├── mask_<i>.png, sam_*_<i>.png, sam3d_input_<i>.png
│   ├── camera_params.json      # MoGe intrinsics, floor_y_scene, RANSAC ground plane
│   ├── detection_result.json, detection_visualization.png, mesh_info.json, reconstruction_map.json
│   ├── scene.ply, scene_posed.ply, scene.gif
│   └── ground_mesh_posed.ply   # only with --wireframe and ground particles enabled
├── trial/                      # step 2: Physion trial format
│   ├── phases_dict.pkl, stat.h5, 0.h5, 3.h5, ... (one per training_fpt frames)
│   ├── coord_transform.json    # scene ↔ Physion coordinate transform
│   ├── build_info.json         # particles per object, velocities actually used, GPT reasoning
│   └── particles_vis.ply
├── rollout/                    # step 3 (+ camera from step 5)
│   ├── rollout_positions.npz   # step_0 ... step_<rollout_steps>
│   ├── centroids_trajectory.npy, rollout_info.json
│   └── camera_physion.json     # written by step 5 (camera restoration)
└── visualization/              # steps 4, 5, 5.5
    ├── camera_view.gif, camera_view_wireframe.gif, camera_view*_step_*.png
    ├── trajectory.gif, centroid_trajectories.png, velocity_analysis.png, ...
    └── optical_flow*.flo, optical_flow_vis.png
```

**Hand-off to `scene_to_tracking`.** That module reads `trial/{coord_transform.json, phases_dict.pkl}`,
`rollout/{rollout_positions.npz, camera_physion.json}`, `scene_3dgs/{object_<i>_mesh_posed.ply, mask_<i>.png}`
and the first frame in the scene root (`1280_704_image.png`, `first_frame_1280x704.png` or `first_frame.png`).
So:
* use the scene folder itself as `-o`, so that the first frame and `prompt.txt` sit next to
  `rollout/ scene_3dgs/ trial/ pipeline_config.json` (this is how the paper batch was run); and
* do **not** pass `--no-camera`, because `rollout/camera_physion.json` is written by the camera-restoration step.

## Usage

All commands are run from `scene_reconstruction/`. The option names below were checked against the argparse
definitions in the scripts.

```bash
cd /path/to/release/scene_reconstruction
export SAM3D_ROOT=/path/to/sam-3d-objects
export OPENAI_API_KEY=...            # only when GPT is used
export BINVOX_PATH=/path/to/binvox   # or put binvox on $PATH
```

### 1. End-to-end (settings used for the paper evaluation scenes)

```bash
SCENE=/path/to/scenes/collide_ball_candle
python pipeline_3dgs_to_physics.py \
    --image "$SCENE/first_frame_1280x704.png" \
    -o "$SCENE" \
    --motion-prompt "$(cat "$SCENE/prompt.txt")" \
    --gns-ckpt ../checkpoints/gns/without_Drape/best_model.pth \
    --enable-angular-velocity \
    --no-ground \
    --ground-clearance 0.004 \
    --wireframe
```

`--gns-ckpt` can be omitted, because this path is its default. `--model-path` is kept as an alias. Other defaults:
`--rollout-steps 167`, `--training-fpt 3`, `--neighbor-radius 0.08`, `--spacing 0.05`, `--velocity-range 20.0`,
`--default-angular-speed 7.0`, `--max-particles-per-obj 600`, `--min-particles-per-obj 20`,
`--target-particles 350`, `--seed 42`.

### 2. Without OpenAI (deterministic object list and velocities)

```bash
python pipeline_3dgs_to_physics.py \
    --image "$SCENE/first_frame_1280x704.png" -o "$SCENE" \
    --no-gpt --text-prompts "ball,candle" \
    --velocity "0:-3,0,0;1:0,0,0" \
    --angular-velocity "0:0,0,7" \
    --no-ground --ground-clearance 0.004 --wireframe
```

* `--text-prompts`: comma-separated SAM3 prompts. They replace GPT object detection.
* `--velocity "idx:vx,vy,vz;..."` (and optionally `--angular-velocity "idx:wx,wy,wz;..."`) replace GPT velocity
  inference. Object indices follow the order of the reconstructed objects (`scene_3dgs/object_<i>*`).
  Coordinates: Y up (gravity −Y), **+X = left in the image**, +Z = away from the camera. Units are m/s and rad/s
  in the normalised simulation space.
* With `--no-gpt`, a missing `--text-prompts` (in `--image` mode) or a missing `--velocity` together with
  `--motion-prompt`/`--chatgpt-velocity` is an error. If neither `--velocity` nor a motion prompt is given,
  all objects start at rest.

### 3. Re-use an existing reconstruction (skip step 1)

```bash
python pipeline_3dgs_to_physics.py --scene-dir "$SCENE/scene_3dgs" -o "$SCENE/rerun" \
    --velocity "0:-3,0,0" --no-ground --ground-clearance 0.004 --wireframe
```

### 4. Run single stages

```bash
# step 1 only (GPT detection, or --text-prompts [--no-gpt])
python auto_scene_pipeline.py "$SCENE/first_frame_1280x704.png" -o "$SCENE/scene_3dgs" --text-prompts "ball,candle" --no-gpt

# step 3 only (note: the standalone default is --rollout-steps 50; the pipeline uses 167)
python run_rollout.py --trial-path "$SCENE/trial" --output-dir "$SCENE/rollout" \
    --gns-ckpt ../checkpoints/gns/without_Drape/best_model.pth --rollout-steps 167
```

## Notes / Known limitations

* **OpenAI non-determinism.** Detection (`gpt-5.2`) and velocity inference (`gpt-5`, a reasoning model,
  so no `temperature` is set) can give different answers on each run. `trial/build_info.json` stores the velocities
  that were actually used (`velocity_spec`, `angular_velocity_spec`) and the GPT reasoning. These stored values
  come after object-index remapping and after the domino alignment rotation done in `build_physion_input.py`.
  So feeding them back through `--velocity` is not an exact round trip when objects were discarded or the domino
  rotation was applied. The raw GPT output is printed only in the run log.
* **Three different `ground_clearance` defaults** (kept unchanged on purpose): `pipeline_3dgs_to_physics.py`
  `--ground-clearance` defaults to `0.025` (also the default of `step2_convert_to_particles`), while
  `build_physion_input.build_from_mesh_output` / `build_from_gs_output` default to `0.020` when called
  directly. The paper evaluation runs passed `--no-ground --ground-clearance 0.004` explicitly. The
  pipeline flag is the user-facing entry point.
* `--use-gpt` is a legacy flag. It defaults to `True` and cannot switch GPT off; use `--no-gpt`.
  Inside `AutoScenePipeline.detect_and_segment_with_sam3`, the `use_gpt` argument does not decide anything: GPT is
  called whenever `text_prompts` is empty. The command-line front ends (`pipeline_3dgs_to_physics.py`,
  `auto_scene_pipeline.py`) check for this combination and stop with an error before it can happen.
  If you call the class directly, pass `text_prompts`.
* The legacy 3DGS→particles path (`--use-3dgs` and the options marked "仅 --use-3dgs 模式":
  `--opacity-threshold`, `--max-particles`, `--min-particles`, `--no-fill`, `--fine-factor`, `--no-fine-fill`,
  `--no-convex-hull`) was not used in the paper. The default is the mesh path. `gs_to_particles.py` is still
  required, because `mesh_to_particles.py` and `build_physion_input.py` import functions from it.
* With `--motion-prompt` and a manual `--velocity` while GPT is enabled, `build_info.json` records
  `velocity_source: chatgpt` even though the manual velocity is used (original behaviour, unchanged). With
  `--no-gpt` the source is recorded as `manual`.
* The module docstring of `pipeline_3dgs_to_physics.py` still lists `gs_to_particles.py` as step 2. The default
  path is `mesh_to_particles.py`.
* Code comments and log messages are mostly in Chinese (original research code, left unchanged).
* **binvox / OpenGL:** `mesh_to_particles.py` runs `xvfb-run -a binvox -aw -dc ...` with
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1`, which is the Mesa path on Debian/Ubuntu. Adjust that path on other
  distributions.
* **Wan-Move export (steps 6/7)** used `export_wan_move_tracks.py` and the Wan-Move code base. It was not used for
  the paper results and is not included. `--export-wan-move` is off by default, and turning it on stops with an
  error. The paper's control videos are produced by the `scene_to_tracking` module.
* **License caveat:** `binvox_rw.py` is third-party code by Daniel Maturana, released under GPL-3.0 (see its file header).
  The modified SAM-3D-Objects files are under the SAM License of the upstream repository.
