# GNS particle-dynamics model (PyG re-implementation) — training, inference, preprocessing

## Overview

This module trains the particle-based physics simulator used by our pipeline: a **Graph Network-based
Simulator (GNS)** re-implemented with PyTorch Geometric, trained on 7 rigid-body scenarios of the
**Physion** `dynamics_training` set (Dominoes, Collide, Support, Link, Contain, Roll, Drop; Drape excluded).
The released checkpoint (`checkpoints/gns/without_Drape/best_model.pth`) is the model consumed by the
scene-reconstruction / rollout module.

```
Physion TDW hdf5 ──preprocessing/──▶ particle trials (phases_dict.pkl, stat.h5, <t>.h5)
                                     + data_splits/<Scenario>/{train,valid}.txt
        └──▶ train_pyg_ddp.py (4-GPU DDP) ──▶ best_model.pth ──▶ inference_pyg.py (autoregressive rollout, RMSE, GIFs)
```

| Path | Role |
|---|---|
| `models/GNS_pyg.py` | GNS model (`GNS_PyG`, `create_gns_model`): encoder / 10-step message passing / residual propagator, optional LayerNorm & deep propagator (both off in the released model) |
| `models/__init__.py` | empty package init (legacy upstream models are not imported) |
| `data_pyg.py` | `PhysionPyGDataset`: loads particle trials, subsamples to ≤3000 particles, adds a floor node, builds radius graph (cKDTree, r=0.08), injects noise |
| `train_pyg.py` | single-GPU trainer; `compute_loss_per_graph` / `str2bool` are reused by the DDP trainer |
| `train_pyg_ddp.py` | DDP trainer used for the released model (scenario-balanced sampling, plateau LR schedule, resume, ckpt pruning) |
| `inference_pyg.py` | rollout + RMSE evaluation + optional RANSAC rigid projection + GIF / error curves; also imported as a function library by the downstream rollout code |
| `utils_geom.py`, `binvox_rw.py` | upstream helper files (only `calc_rigid_transform` is used by `inference_pyg.py`; `binvox_rw` is required because `utils_geom` imports it at module level) |
| `scripts/train_4gpu.sh` | exact training command of the released model |
| `scripts/inference.sh` | 7-scenario rollout evaluation (5 validation trials per scenario) |
| `data_splits/<Scenario>/{train,valid}.txt` | the exact train/valid split used for training (7 scenarios) |
| `preprocessing/` | the preprocessing code actually used to build the training data (binvox `-e` variant) |
| `patches/preprocessing_vs_upstream_5ac002a.patch` | our modifications to the upstream preprocessing files (for reference; already applied in `preprocessing/`) |

## Dependencies / Environment

Reference environment: Python 3.10.12, CUDA 12.4, PyTorch 2.6.0, PyG 2.7.0, torch_scatter 2.1.2, numpy 2.2.6
(see `requirements.txt`; numpy ≥ 1.24 is handled by an `np.int = int` shim in `inference_pyg.py`).

```bash
conda create -n gns python=3.10 -y && conda activate gns
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -r requirements.txt
```

* `inference_pyg.py` imports `utils_geom.py`, which imports `trimesh`, `matplotlib` and `ipdb` at module level,
  so these are required even for inference.
* Preprocessing additionally needs the **`binvox` executable** (https://www.patrickmin.com/binvox/) on `PATH`
  (or pass `BINVOX_DIR=<dir>`), and `Xvfb`: `pipeline_preprocess.sh` starts `Xvfb :99` if it is not running and
  exports `DISPLAY=:99` (binvox needs an X display).
* Training hardware of the released model: 4x NVIDIA H200 (per-GPU batch 8, ≈112 s / epoch, ≈31 h for 1000 epochs).

## Upstream repository, patch and license

This code builds on **htung0101/Physion-particles @ `5ac002a`** (official particle-based GNS / DPI-Net baselines
of the Physion benchmark, https://github.com/htung0101/Physion-particles).

* `models/GNS_pyg.py`, `data_pyg.py`, `train_pyg.py`, `train_pyg_ddp.py`, `inference_pyg.py`,
  `preprocessing/preprocessing_tdw_parallel.py`, `preprocessing/extract_with_progress.py`,
  `preprocessing/pipeline_preprocess.sh` are **new files** written by us (the GNS-PyG model mirrors the
  structure of the upstream `models/GNS.py`).
* `utils_geom.py`, `binvox_rw.py` (top level) and `preprocessing/utils.py` are **upstream files, unmodified**
  (apart from a provenance header and replacing two example paths inside comments).
* `preprocessing/{utils_geom.py, binvox_rw.py, data.py, create_train_valid.py}` are **upstream files with our
  modifications** — most importantly binvox is called with `-e` (exact voxelization) instead of `-pb`, the
  binvox header parser skips comment lines written by binvox ≥ 1.38, `np.int`/`np.bool` were replaced for
  numpy ≥ 1.24, and `create_train_valid.py` covers all 8 scenarios, skips missing ones and fixes the
  `os.path.join` argument order.
  The full diff is in `patches/preprocessing_vs_upstream_5ac002a.patch`; it applies cleanly to a pristine
  upstream checkout and reproduces the files in `preprocessing/` (minus our provenance headers):
  ```bash
  git clone https://github.com/htung0101/Physion-particles.git && cd Physion-particles
  git checkout 5ac002a
  git apply /path/to/gns_training/patches/preprocessing_vs_upstream_5ac002a.patch
  ```
  You do **not** need to do this to use this module — `preprocessing/` is self-contained.
* `preprocessing/files/models_full_check_window.txt` is an upstream data file (unmodified).

**License status:** the upstream repository does **not** contain a LICENSE file; redistribution terms of the
upstream-derived files must be confirmed with the upstream authors before publication. `binvox_rw.py`
(binvox-rw-py, Daniel Maturana) is GPL-3.0.

## Checkpoints

| Checkpoint | Location | Notes |
|---|---|---|
| `without_Drape/best_model.pth` | `../checkpoints/gns/without_Drape/best_model.pth` | **released model**; epoch 718 / 1000, valid loss 0.022298 |
| `without_Drape/ckpt_epoch_995..999.pth` | same directory | last 5 regular checkpoints (resume / inspection only) |

sha256 sums, checkpoint keys, and the full argument list are in
`../checkpoints/gns/without_Drape/MANIFEST.md`. The checkpoint stores `model_state_dict` (532,603 parameters)
plus optimizer/scheduler states; model hyper-parameters are not stored, so always instantiate with
`create_gns_model(200, 300, 200, pstep=10, residual=True, use_layernorm=False, deep_propagator=False)`
(the defaults of `inference_pyg.py`).

### Hyper-parameters of `without_Drape` (complete)

| Group | Value |
|---|---|
| Scenarios | `Dominoes,Collide,Support,Link,Contain,Roll,Drop` (Drape excluded) |
| Model | `nf_particle 200`, `nf_relation 300`, `nf_effect 200`, `pstep 10`, `residual true`, `use_layernorm false`, `deep_propagator false` (attr_dim 3, state_dim 6) |
| Graph / data | `neighbor_radius 0.08`, `training_fpt 3` (dt = 0.03 s), `floor_cheat true`, `subsample_limit 3000`, `noise_std 3e-4`, `stat_path None` (no normalization) |
| Sampling | `sampling_mode scenario_balanced`, `train_samples_per_trial 2` (22,134 train samples / epoch, 1,233 valid trials) |
| Optimization | Adam (`beta1 0.9`, beta2 0.999), `lr 1e-4`, `weight_decay 0`, `grad_clip 0`, `forward_times 1` (no grad accumulation) |
| Schedule | `lr_schedule plateau` = ReduceLROnPlateau(factor 0.8, patience 10, min_lr 1e-6) on valid loss; `n_epoch 1000` |
| Batch | per-GPU `batch_size 8`, 4 GPUs → global batch 32 |
| Misc | `seed 0` (+rank), `num_workers 8`, `ckp_per_epoch 1`, `max_keep_ckpt 5`, trained from scratch |
| Selection | best validation loss → epoch 718 (valid loss 0.022298) |

## Data format

**Raw data.** Physion `dynamics_training` HDF5 archives `<Scenario>_dynamics_training_HDF5s.tar.gz`
(Physion benchmark, Bear et al., NeurIPS 2021 Datasets & Benchmarks; download links in the official Physion
repository https://github.com/cogtoolslab/physics-benchmarking-neurips2021).

**Preprocessed data** (`<DATA_ROOT>` = `<dpi_data_dir>/train`, default `./data/preprocessed/train`):
```
<DATA_ROOT>/<Scenario>/
    train.txt  valid.txt               # one trial name per line (copy from data_splits/<Scenario>/)
    <trial>/
        phases_dict.pkl                # instance_idx, material, obj_points, n_objects, n_particles,
                                       # time_step, dt (=0.01), root_des_radius, clusters, instance, ...
        stat.h5                        # positions / velocities statistics (unused: training used stat_path=None)
        0.h5 ... <T-1>.h5              # rigid: obj_positions, obj_rotations   (Drape: particle_positions, particle_velocities)
```
Particles are obtained by voxelizing every object mesh with binvox at spacing 0.05.

**Splits.** `data_pyg.py` reads `<DATA_ROOT>/<Scenario>/{train,valid}.txt`; if these files are missing it
silently falls back to *all* trial directories. The original split was generated by `create_train_valid.py`
with an unseeded `random.shuffle`, so it cannot be regenerated — **always copy the released split files**:
```bash
for s in Dominoes Collide Support Link Contain Roll Drop; do cp data_splits/$s/*.txt <DATA_ROOT>/$s/; done
```
| Scenario | train | valid |
|---|---|---|
| Dominoes | 1800 | 200 |
| Collide | 1792 | 200 |
| Support | 655 | 73 |
| Link | 1801 | 201 |
| Contain | 1418 | 158 |
| Roll | 1801 | 201 |
| Drop | 1800 | 200 |

## Usage

All commands are run from `gns_training/`.

### 1. Preprocessing (optional — only if you rebuild the particle data)

```bash
cd preprocessing
# edit paths.yaml: training_data_dir (raw archives / extracted <Scenario>/) and dpi_data_dir (output)
PHYSION_RAW_DIR=../data/physion_raw/dynamics_training BINVOX_DIR=/path/to/binvox_dir \
    bash pipeline_preprocess.sh Dominoes 32        # <Scenario> [num_workers]
# repeat for Collide Support Link Contain Roll Drop
cd ..
# then overwrite the freshly generated (random) splits with the released ones (see "Splits" above)
```
`pipeline_preprocess.sh` = (1) extract the archive with `extract_with_progress.py`, (2) run
`preprocessing_tdw_parallel.py --scenario <S> --mode train --num_workers <N>` (a trial is skipped if no
progress is made for `--timeout` = 60 s),
(3) run `create_train_valid.py`. `PHYSION_RAW_DIR` must point to the same directory as `training_data_dir`
in `paths.yaml` (relative paths in `paths.yaml` are resolved from `preprocessing/`).

### 2. Training (released model)

```bash
bash scripts/train_4gpu.sh <DATA_ROOT> [OUTF]
# logs: ./logs/{stdout,stderr,gpu}.<RUN_ID>.log ; checkpoints: OUTF (default ./output_pyg_ddp_4gpu/all_scenarios_bs8_lr1e4_without_Drape)
```
which runs
```bash
torchrun --standalone --nproc_per_node=4 train_pyg_ddp.py --data_root <DATA_ROOT> \
  --scenario Dominoes,Collide,Support,Link,Contain,Roll,Drop \
  --batch_size 8 --forward_times 1 --sampling_mode scenario_balanced --train_samples_per_trial 2 \
  --ckp_per_epoch 1 --max_keep_ckpt 5 --lr 1e-4 --n_epoch 1000 \
  --outf ./output_pyg_ddp_4gpu/all_scenarios_bs8_lr1e4_without_Drape
```
Resume an interrupted run with `--resume auto` (latest regular checkpoint in `--outf`) or `--resume <ckpt>`;
`--resume_weights_only true` loads only the model weights (fine-tuning).

### 3. Rollout evaluation

All 7 scenarios, 5 random **validation** trials each, full rollout from timestep 45 with RANSAC rigid projection:
```bash
bash scripts/inference.sh <DATA_ROOT> [CHECKPOINT] [OUTPUT_BASE]
# CHECKPOINT default ../checkpoints/gns/without_Drape/best_model.pth ; OUTPUT_BASE default ./inference_output_all
```
Single scenario:
```bash
python inference_pyg.py \
  --checkpoint ../checkpoints/gns/without_Drape/best_model.pth \
  --data_root <DATA_ROOT> --scenario Collide --split valid --n_trials 5 \
  --full_rollout --start_timestep 45 --training_fpt 3 --neighbor_radius 0.08 \
  --ransac_on_pred 1 --visualize --gif_fps 4 --seed 42 \
  --outf ./inference_output_all/Collide
```
Outputs per scenario: `results.json` (per-step RMSE / MSE / max error for each trial), and with `--visualize`
one GIF per trial plus `error_curves.png`.

`--split` (new): `valid` (default) samples trials only from `<Scenario>/valid.txt`; `train` from `train.txt`;
`all` reproduces the original behaviour (all trial directories, *including training trials*).
Other options: `--rollout_steps N` (when `--full_rollout` is not given), `--cloth_proj 1` (+ `--cloth_proj_iters`,
`--cloth_proj_stiffness`, `--cloth_proj_radius_scale`; experimental cloth distance projection, off by default and
not used for the released results).

## Notes / Known limitations

* **Node-attribute semantics (working-tree version).** `data_pyg.py` / `inference_pyg.py` are released in their
  latest version, which sets `attr[:, 1] = 1` for non-rigid (cloth / fluid) particles. The released checkpoint was
  trained with an earlier version in which every particle had `attr = 0`. For the 7 rigid-body scenarios (and the
  downstream pipeline, which marks every object as `rigid`) the two versions produce **identical inputs**; they
  differ only on Drape / non-rigid data, for which the corresponding weights were never trained.
* **Inference trial sampling changed.** Originally `inference_pyg.py` sampled from *all* trials of a scenario
  (including training trials). It now samples from `valid.txt` by default (`--split valid`). Consequently, with the
  same `--seed 42` the selected trials differ from the original internal runs; use `--split all` to reproduce the
  old selection.
* `utils_geom.py` still imports `ipdb`, `trimesh`, `matplotlib`, `binvox_rw` at module level (kept unmodified);
  install them even if you only run inference.
* **Preprocessing fallback.** `preprocessing/utils_geom.py` silently falls back to a pure-Python trimesh
  voxelization when `binvox` is not on `PATH`; the released data were produced **with binvox (`-e`)**, and
  `pipeline_preprocess.sh` aborts if binvox is missing. Running `preprocessing_tdw_parallel.py` directly without
  binvox will produce different particles.
* `create_train_valid.py` reshuffles (unseeded) and **overwrites the splits of every scenario found** each time
  `pipeline_preprocess.sh` runs; re-copy `data_splits/` afterwards.
* The package versions of the environment used for preprocessing were not recorded separately (the
  `requirements.txt` versions are those of the training environment); bit-exact regeneration of the particle data
  is not guaranteed.
* A model fine-tuned on synthetic "Slide" data (`slide_finetune_A`) existed internally; its exact training command
  was not recorded, so it is **not** released or documented here.
* Exploratory variants (multi-step pushforward training, warm restarts, Drape material bit, cloth projection
  experiments) are not part of this release.
* No per-scenario rollout RMSE is reported in the paper for this model; `inference_pyg.py` is provided for
  qualitative inspection and sanity checks.
