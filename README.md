# Seeing the Unseen: Unified Visible–Invisible Motion for Physically Consistent Video Generation

Official code release for our NeurIPS 2026 paper *Seeing the Unseen: Unified Visible–Invisible Motion for Physically Consistent Video Generation*. The system turns a single image plus a motion
description into a physically plausible video: the scene is reconstructed in 3D, its dynamics are
simulated with a learned particle simulator (GNS), the simulated motion is rendered as two
tracking-control videos, and a dual-control video diffusion model (Wan2.2-TI2V-5B fine-tuned)
generates the final video.

```
                 ┌──────────────────────┐    rollout + meshes   ┌──────────────────────┐
 image + prompt ─► scene_reconstruction ├──────────────────────► scene_to_tracking     │
                 │ (SAM-3D-Objects,     │                       │ (mesh → DaS tracking  │
                 │  binvox, GNS rollout)│                       │  videos, 1280×704×121)│
                 └──────────┬───────────┘                       └──────────┬───────────┘
                            │ uses                                         │ visible + future-visible
                 ┌──────────▼───────────┐                       ┌──────────▼───────────┐
                 │ gns_training         │                       │ video_generation      │
                 │ (Physion GNS, PyG)   │                       │ (Wan2.2 dual control) │
                 └──────────────────────┘                       └──────────▲───────────┘
                                                                           │ training data
                                                                ┌──────────┴───────────┐
                                                                │ tracking_video        │
                                                                │ (SpaTrackerV2 on real │
                                                                │  videos, MiraData)    │
                                                                └──────────────────────┘
```

## Modules

| Directory | Role | Upstream dependency |
|---|---|---|
| `scene_reconstruction/` | Image → multi-object 3D scene → particles → GNS rollout (inference entry point) | [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects) @ `81a8237` + patch |
| `gns_training/` | Training / evaluation of the particle-based simulator (PyG GNS, DDP) | [Physion-particles](https://github.com/htung0101/Physion-particles) @ `5ac002a` |
| `scene_to_tracking/` | Rollout + meshes → `visible` and `future-visible` DaS tracking videos (inference-time control signals) | none |
| `tracking_video/` | Generates the same two control videos from real videos (training data for the video model) | [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) @ `7e12274` + patch |
| `video_generation/` | Dual-control fine-tuning and inference of Wan2.2-TI2V-5B | [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) @ `54bf97a` + patch |

Each module has its own `README.md` (environment, upstream setup, data format, commands and known
limitations).

## Checkpoints

The `checkpoints/` directory is not tracked by git; weights are published on Hugging Face (see the
module READMEs). `checkpoints/*/MANIFEST.md` lists file names, sizes, sha256 and training provenance.

| Checkpoint | Used by | Size |
|---|---|---|
| `checkpoints/gns/without_Drape/best_model.pth` | `scene_reconstruction`, `gns_training` | 6 MB |
| `checkpoints/video_generation/checkpoint-10500/transformer/` | `video_generation` (planF, step 10500) | 10 GB |

Base model for video generation: [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## End-to-end inference

```bash
# 1. image → 3D scene → GNS rollout
cd scene_reconstruction
export OPENAI_API_KEY=...        # object detection + velocity inference (or use --no-gpt, see README)
python pipeline_3dgs_to_physics.py --image input.png -o out/scene --motion-prompt "..."

# 2. rollout → two control videos
cd ../scene_to_tracking
python generate_tracking_videos.py --data_dir ../scene_reconstruction/out/scene

# 3. control videos → final video
cd ../video_generation
python examples/predict_dual_control.py \
  --base_model_path <Wan2.2-TI2V-5B> \
  --transformer_path ../checkpoints/video_generation/checkpoint-10500 \
  --tracking_video_1 ../scene_reconstruction/out/scene/visible_tracking_das.mp4 \
  --tracking_video_2 ../scene_reconstruction/out/scene/future_visible_tracking_das.mp4 \
  --start_image ../scene_reconstruction/out/scene/first_frame_1280x704.png \
  --prompt "..." --guidance_scale 3.0 --num_inference_steps 30
```

Each step is described in detail in the corresponding module README. The modules use different
Python environments (see each README); they communicate only through files on disk.

## License

TBD. Note that `binvox_rw.py` (used by `scene_reconstruction` and `gns_training`) is GPL-3.0.
