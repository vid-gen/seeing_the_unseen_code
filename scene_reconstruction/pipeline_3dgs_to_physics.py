#!/usr/bin/env python3
"""
pipeline_3dgs_to_physics.py - 完整 3DGS → 物理仿真 Pipeline

串联所有步骤:
  1. [可选] auto_scene_pipeline.py: 图像 → 3DGS 场景
  2. gs_to_particles.py: 3DGS → 粒子
  3. build_physion_input.py: 粒子 → Physion 格式
  4. run_rollout.py: GNS rollout 推理
  5. visualize_rollout.py: 结果可视化
  6. [可选] export_wan_move_tracks.py: 导出 Wan-Move 轨迹 (not included in the open-source release)
  7. [可选] Wan-Move 视频生成: 加载模型生成视频 (not included in the open-source release)

支持两种启动模式:
  A. 从图像开始 (--image): 运行完整 Pipeline
  B. 从 3DGS 场景开始 (--scene-dir): 跳过 3D 重建步骤
"""

import os
import sys
import argparse
import json
import time
import pickle
import numpy as np

# 确保 my_code/ 在 sys.path 最前面 (优先于 Physion-particles 的旧 binvox_rw.py 等)
MY_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MY_CODE_DIR)

# 添加 Physion-particles 到 path (在 my_code/ 之后)
# GNS code directory (models/GNS_pyg.py, inference_pyg.py). Defaults to the sibling
# release module ../gns_training; override with the PHYSION_PARTICLES_DIR env variable.
PHYSION_DIR = os.environ.get(
    'PHYSION_PARTICLES_DIR',
    os.path.join(os.path.dirname(MY_CODE_DIR), 'gns_training'))
sys.path.insert(1, PHYSION_DIR)

# Default GNS checkpoint: <release root>/checkpoints/gns/without_Drape/best_model.pth
DEFAULT_GNS_CKPT = os.path.join(
    os.path.dirname(MY_CODE_DIR), 'checkpoints', 'gns', 'without_Drape', 'best_model.pth')

# numpy compatibility
if not hasattr(np, "int"):
    np.int = int


def step1_reconstruct_scene(image_path, scene_dir, max_objects=5,
                             device='cuda', text_prompts=None,
                             use_gpt=True, gpt_model='gpt-5.2',
                             max_depth_ratio=2.0, no_depth_filter=False,
                             layout_optimize=True, seed=42,
                             sam3_threshold=0.5):
    """
    Step 1: 从图像重建 3DGS 场景

    调用 auto_scene_pipeline.py 的 AutoScenePipeline
    """
    print("\n" + "=" * 70)
    print("  Step 1: Image → 3DGS Scene Reconstruction")
    print("=" * 70)

    # 动态导入 (避免不需要时加载大量模型)
    my_code_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, my_code_dir)
    from auto_scene_pipeline import AutoScenePipeline

    pipeline = AutoScenePipeline(device=device)
    scene_gs = pipeline.run(
        image_path=image_path,
        output_dir=scene_dir,
        max_objects=max_objects,
        save_intermediate=True,
        save_gif=True,
        text_prompts=text_prompts,
        use_gpt=use_gpt,
        gpt_model=gpt_model,
        max_depth_ratio=max_depth_ratio,
        no_depth_filter=no_depth_filter,
        layout_optimize=layout_optimize,
        seed=seed,
        sam3_threshold=sam3_threshold,
    )

    if scene_gs is None:
        raise RuntimeError("3DGS scene reconstruction failed")

    # 清理 GPU 内存
    import torch
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  3DGS 场景保存到: {scene_dir}")
    return scene_dir


def step2_convert_to_particles(scene_dir, trial_dir, velocity_spec=None,
                                spacing=0.05, opacity_threshold=0.5,
                                max_particles=3000, min_particles=150,
                                discard_threshold=10, training_fpt=3,
                                fill_interior=True, fine_factor=2,
                                use_convex_hull=True,
                                chatgpt_velocity=False,
                                motion_prompt=None,
                                max_velocity=20.0,
                                image_path=None, gpt_model='gpt-5',
                                add_ground=True, ground_margin=0.3,
                                ground_clearance=0.025,
                                use_mesh=True,
                                max_particles_per_obj=600,
                                min_particles_per_obj=20,
                                target_particles=350,
                                enable_angular_velocity=False,
                                default_angular_speed=7.0,
                                manual_angular_velocity_spec=None,
                                thicken=False,
                                thicken_min_layers=3,
                                align_z=False):
    """
    Step 2-3: 3DGS/Mesh → 粒子 → Physion 格式

    use_mesh=True (默认): 使用 mesh 进行粒子转换 (更可靠的内部填充)
    use_mesh=False: 使用旧的 3DGS 方案 (binary_fill_holes + 凸包兜底)
    """
    source_type = "Mesh" if use_mesh else "3DGS"
    print("\n" + "=" * 70)
    print(f"  Step 2: {source_type} → Particles → Physion Format")
    print("=" * 70)

    # ChatGPT 智能速度推断
    velocity_reasoning = None
    angular_velocity_spec = manual_angular_velocity_spec
    if chatgpt_velocity and velocity_spec is None:
        velocity_spec, gpt_angular_spec, velocity_reasoning = infer_velocity_with_chatgpt(
            scene_dir, image_path=image_path, gpt_model=gpt_model,
            motion_prompt=motion_prompt, max_velocity=max_velocity,
            enable_angular_velocity=enable_angular_velocity,
            default_angular_speed=default_angular_speed)
        if angular_velocity_spec is None:
            angular_velocity_spec = gpt_angular_spec

    # 加载地面参数（优先 RANSAC 平面，回退 floor_y）
    floor_y_scene = None
    ground_plane = None
    camera_params_path = os.path.join(scene_dir, 'camera_params.json')
    if os.path.exists(camera_params_path):
        with open(camera_params_path, 'r') as f:
            cp = json.load(f)
        ground_plane = cp.get('ground_plane', None)
        floor_y_scene = cp.get('floor_y_scene', None)
        if ground_plane is not None:
            print(f"  Loaded ground plane (RANSAC): normal={[round(n,4) for n in ground_plane['normal']]}")
        if floor_y_scene is not None:
            print(f"  Loaded floor_y_scene: {floor_y_scene:.4f}")

    # 检测是否有 mesh 文件可用
    if use_mesh:
        import glob as _glob
        mesh_files = _glob.glob(os.path.join(scene_dir, "object_*_mesh_posed.ply"))
        if not mesh_files:
            print(f"  警告: 未找到 mesh 文件, 回退到 3DGS 方案")
            use_mesh = False

    if use_mesh:
        from build_physion_input import build_from_mesh_output
        trial_path, scene_info = build_from_mesh_output(
            scene_dir=scene_dir,
            output_dir=trial_dir,
            velocity_spec=velocity_spec,
            angular_velocity_spec=angular_velocity_spec,
            spacing=spacing,
            discard_threshold=discard_threshold,
            training_fpt=training_fpt,
            add_ground=add_ground,
            ground_margin=ground_margin,
            ground_clearance=ground_clearance,
            max_particles_per_obj=max_particles_per_obj,
            min_particles_per_obj=min_particles_per_obj,
            target_particles=target_particles,
            floor_y_scene=floor_y_scene,
            ground_plane=ground_plane,
            thicken=thicken,
            thicken_min_layers=thicken_min_layers,
            align_z=align_z,
        )
    else:
        from build_physion_input import build_from_gs_output
        trial_path, scene_info = build_from_gs_output(
            scene_dir=scene_dir,
            output_dir=trial_dir,
            velocity_spec=velocity_spec,
            angular_velocity_spec=angular_velocity_spec,
            spacing=spacing,
            opacity_threshold=opacity_threshold,
            max_particles=max_particles,
            min_particles=min_particles,
            discard_threshold=discard_threshold,
            training_fpt=training_fpt,
            fill_interior=fill_interior,
            fine_factor=fine_factor,
            use_convex_hull=use_convex_hull,
            add_ground=add_ground,
            ground_margin=ground_margin,
            ground_clearance=ground_clearance,
            floor_y_scene=floor_y_scene,
        )

    # 记录速度来源信息到 build_info.json
    build_info_path = os.path.join(trial_dir, 'build_info.json')
    if os.path.exists(build_info_path):
        with open(build_info_path, 'r') as f:
            build_info = json.load(f)
        if chatgpt_velocity:
            build_info['velocity_source'] = 'chatgpt'
            build_info['motion_prompt'] = motion_prompt
            build_info['gpt_velocity_model'] = gpt_model
            build_info['velocity_reasoning'] = velocity_reasoning
            if angular_velocity_spec:
                build_info['angular_velocity_source'] = 'chatgpt'
        elif velocity_spec:
            build_info['velocity_source'] = 'manual'
        else:
            build_info['velocity_source'] = 'zero'
        with open(build_info_path, 'w') as f:
            json.dump(build_info, f, indent=2)

    print(f"\n  Physion trial 保存到: {trial_path}")
    return trial_path, scene_info


def step3_run_rollout(trial_dir, model_path, rollout_dir,
                      rollout_steps=50, training_fpt=3,
                      device='cuda', neighbor_radius=0.08,
                      ransac=True):
    """
    Step 4: GNS Rollout 推理
    """
    print("\n" + "=" * 70)
    print("  Step 3: GNS Rollout Inference")
    print("=" * 70)

    import torch
    from run_rollout import load_model, run_rollout, save_rollout_results

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # 加载模型
    print(f"\n  Loading model: {model_path}")
    model = load_model(model_path, device)

    # 运行 rollout
    positions, centroids = run_rollout(
        model=model,
        trial_path=trial_dir,
        rollout_steps=rollout_steps,
        training_fpt=training_fpt,
        device=device,
        neighbor_radius=neighbor_radius,
        ransac_on_pred=ransac,
    )

    # 保存结果
    save_rollout_results(
        output_dir=rollout_dir,
        positions=positions,
        centroids_per_step=centroids,
        phases_dict_path=trial_dir,
        extra_info={
            'model_path': model_path,
            'rollout_steps': rollout_steps,
            'training_fpt': training_fpt,
            'ransac': ransac,
        }
    )

    # 清理 GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Rollout 结果保存到: {rollout_dir}")
    return positions, centroids


def step4_visualize(trial_dir, rollout_dir, vis_dir,
                    training_fpt=3, dt=0.01, fps=6):
    """
    Step 5: 可视化
    """
    print("\n" + "=" * 70)
    print("  Step 4: Visualization")
    print("=" * 70)

    from visualize_rollout import (
        load_rollout_data, generate_all_visualizations
    )

    positions_list, centroids_array, rollout_info = load_rollout_data(rollout_dir)

    phases_dict_path = os.path.join(trial_dir, 'phases_dict.pkl')
    with open(phases_dict_path, 'rb') as f:
        phases_dict = pickle.load(f)

    effective_dt = training_fpt * dt

    generate_all_visualizations(
        positions_list, centroids_array, phases_dict,
        vis_dir, dt=effective_dt, fps=fps)

    print(f"\n  可视化保存到: {vis_dir}")


def step5_restore_camera(scene_dir, trial_dir, rollout_dir, vis_dir,
                          image_path=None):
    """
    Step 5: 摄像头位置还原 + 原始视角渲染

    从 camera_params.json + coord_transform.json 还原相机位置，
    并从原始视角渲染物理仿真轨迹。
    """
    print("\n" + "=" * 70)
    print("  Step 5: Camera Restoration & Original View Rendering")
    print("=" * 70)

    from camera_restore import CameraInPhysion, visualize_flow, save_flo

    # 查找 camera_params.json
    camera_params_path = os.path.join(scene_dir, 'camera_params.json')
    if not os.path.exists(camera_params_path):
        print(f"  警告: {camera_params_path} 不存在, 跳过相机还原")
        return None

    # 查找 coord_transform.json
    coord_transform_path = os.path.join(trial_dir, 'coord_transform.json')
    if not os.path.exists(coord_transform_path):
        # 尝试从 scene_info.json 读取
        scene_info_path = os.path.join(trial_dir, 'scene_info.json')
        if os.path.exists(scene_info_path):
            camera = CameraInPhysion.from_scene_info(
                camera_params_path, scene_info_path)
        else:
            print(f"  警告: 未找到坐标变换参数, 跳过相机还原")
            return None
    else:
        camera = CameraInPhysion.from_files(
            camera_params_path, coord_transform_path)

    print(f"  {camera}")

    # 保存相机参数到 rollout 目录
    os.makedirs(rollout_dir, exist_ok=True)
    camera_physion_path = os.path.join(rollout_dir, 'camera_physion.json')
    camera.save(camera_physion_path)

    # 如果有 rollout 结果，生成原始视角渲染
    rollout_pos_path = os.path.join(rollout_dir, 'rollout_positions.npz')
    if not os.path.exists(rollout_pos_path):
        print("  无 rollout 结果, 跳过原始视角渲染")
        return camera

    os.makedirs(vis_dir, exist_ok=True)

    # 加载 rollout 数据
    data = np.load(rollout_pos_path, allow_pickle=True)
    positions_list = [data[f'step_{i}'] for i in range(len(data.files))]

    phases_dict_path = os.path.join(trial_dir, 'phases_dict.pkl')
    with open(phases_dict_path, 'rb') as f:
        phases_dict = pickle.load(f)

    n_objects = phases_dict['n_objects']
    instance_idx = np.array(phases_dict['instance_idx'])

    trajectory = np.array(positions_list)  # (n_steps, N, 3)

    # 1. 初始帧粒子投影叠加到原始图像
    if not image_path or not os.path.exists(image_path):
        image_path = find_scene_image(scene_dir)
    if image_path and os.path.exists(image_path):
        from PIL import Image as PILImage
        orig_image = np.array(PILImage.open(image_path).convert("RGB"))
    else:
        orig_image = None

    # 2. 绘制 2D 轨迹图
    print("  生成 2D 轨迹图...")
    trajectory_path = os.path.join(vis_dir, 'trajectory_2d.png')
    camera.render_trajectory_2d(
        trajectory, instance_idx, n_objects,
        image=orig_image,
        output_path=trajectory_path,
    )

    # 3. 生成光流
    if len(positions_list) >= 2:
        print("  生成光流...")
        # 第一帧到最后一帧的光流
        flow, valid_mask = camera.compute_optical_flow(
            positions_list[0], positions_list[-1])

        # 保存光流
        flow_path = os.path.join(vis_dir, 'optical_flow.flo')
        save_flo(flow, flow_path)
        print(f"  保存光流: {flow_path}")

        # 光流可视化
        flow_vis = visualize_flow(flow)
        import cv2
        cv2.imwrite(os.path.join(vis_dir, 'optical_flow_vis.png'), flow_vis[:, :, ::-1])
        print(f"  保存光流可视化: optical_flow_vis.png")

        # 逐帧光流
        for t in range(min(len(positions_list) - 1, 10)):
            flow_t, _ = camera.compute_optical_flow(
                positions_list[t], positions_list[t + 1])
            flow_t_path = os.path.join(vis_dir, f'optical_flow_{t:02d}.flo')
            save_flo(flow_t, flow_t_path)

        print(f"  保存逐帧光流: optical_flow_00.flo ~ optical_flow_{min(len(positions_list)-2, 9):02d}.flo")

    # 4. 初始帧投影叠加
    if orig_image is not None:
        print("  生成初始帧投影叠加...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(
            camera.image_width/100, camera.image_height/100), dpi=150)
        ax.imshow(orig_image)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                  '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

        for obj_id in range(n_objects):
            st = instance_idx[obj_id]
            ed = instance_idx[obj_id + 1]
            obj_pts = positions_list[0][st:ed]
            uv, depth, visible = camera.project_with_visibility(obj_pts)
            color = colors[obj_id % len(colors)]
            if len(uv) > 0:
                ax.scatter(uv[:, 0], uv[:, 1], c=color, s=3, alpha=0.5)

        ax.set_xlim(0, camera.image_width)
        ax.set_ylim(camera.image_height, 0)
        ax.axis('off')
        plt.tight_layout(pad=0)

        overlay_path = os.path.join(vis_dir, 'overlay_initial.png')
        fig.savefig(overlay_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        print(f"  保存初始帧叠加: {overlay_path}")

    # 5. 原始相机视角动画 GIF
    print("  生成原始相机视角动画...")
    from visualize_rollout import create_camera_view_gif
    camera_gif_path = os.path.join(vis_dir, 'camera_view.gif')
    create_camera_view_gif(
        positions_list, phases_dict, camera,
        camera_gif_path,
        orig_image=orig_image,
        fps=8)

    print(f"\n  相机还原完成! 输出: {vis_dir}")
    return camera


# ============================================================
# ChatGPT 智能速度推断
# ============================================================

# OpenAI API key is read from the OPENAI_API_KEY environment variable (never hard-coded)
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 不可动物体关键词 — 这些物体速度始终为 0
# 已清空：所有物体的速度完全由 ChatGPT 根据 motion_prompt 推断
IMMOVABLE_KEYWORDS = set()


def find_scene_image(scene_dir):
    """
    在场景目录中查找原始输入图像

    搜索顺序:
    1. detection_result.json 中的 image_path 字段
    2. scene_dir 下常见图像文件（排除中间结果）

    Returns:
        image_path: str 或 None
    """
    # 1. 从 detection_result.json 读取
    det_path = os.path.join(scene_dir, 'detection_result.json')
    if os.path.exists(det_path):
        with open(det_path, 'r') as f:
            det = json.load(f)
        img_path = det.get('image_path')
        if img_path and os.path.exists(img_path):
            return img_path

    # 2. 搜索场景目录中的图像文件（排除中间结果）
    skip_prefixes = ('sam_', 'mask_', 'detection_', 'sam3d_input_')
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        import glob
        for f in sorted(glob.glob(os.path.join(scene_dir, ext))):
            basename = os.path.basename(f)
            if not any(basename.startswith(p) for p in skip_prefixes):
                return f

    return None


def collect_scene_context(scene_dir):
    """
    收集场景上下文信息，用于构建 GPT prompt

    确保返回的物体数与实际 .ply 文件数一致。如果 detection_result.json
    包含的物体多于实际重建的（深度过滤或重建失败导致），
    使用 reconstruction_map.json 精确映射或截断。

    Returns:
        context: dict with keys:
          'objects': list of {label, bbox, centroid, ...}
          'n_objects': int
          'image_size': (W, H) 或 None
          'positions': list of [x,y,z] 或 None
    """
    context = {'objects': [], 'n_objects': 0, 'positions': None}

    # 1. detection_result.json → 物体标签、bbox
    det_path = os.path.join(scene_dir, 'detection_result.json')
    if os.path.exists(det_path):
        with open(det_path, 'r') as f:
            det = json.load(f)
        context['objects'] = det.get('objects', [])
        context['n_objects'] = len(context['objects'])

    # 2. 确保物体数与实际 .ply 文件数一致
    n_ply = 0
    while os.path.exists(os.path.join(scene_dir, f'object_{n_ply}.ply')):
        n_ply += 1

    if n_ply > 0 and context['n_objects'] > n_ply:
        # detection_result.json 包含的物体多于实际重建的
        map_path = os.path.join(scene_dir, 'reconstruction_map.json')
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                rmap = json.load(f)
            ply_to_det = rmap.get('ply_to_detection', [])
            original_objects = context['objects']
            context['objects'] = [original_objects[di] for di in ply_to_det
                                  if di < len(original_objects)]
            print(f"  使用 reconstruction_map.json 映射: "
                  f"{len(original_objects)} 检测 → {len(context['objects'])} 重建")
        else:
            # 无映射文件（旧数据）— 截断到 .ply 数量
            print(f"  警告: detection_result.json 有 {context['n_objects']} 物体，"
                  f"但只有 {n_ply} 个 .ply 文件，截断到 {n_ply}")
            context['objects'] = context['objects'][:n_ply]
        context['n_objects'] = len(context['objects'])

    # 3. scene_info.json → 3D 位置
    scene_info_path = os.path.join(scene_dir, 'scene_info.json')
    if os.path.exists(scene_info_path):
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        context['positions'] = scene_info.get('centroids')
        # 使用 scene_info 的物体名补充（如果 detection_result 没有）
        if not context['objects'] and 'object_names' in scene_info:
            context['objects'] = [{'label': n} for n in scene_info['object_names']]
            context['n_objects'] = len(context['objects'])

    # 4. camera_params.json → 图像尺寸
    cam_path = os.path.join(scene_dir, 'camera_params.json')
    if os.path.exists(cam_path):
        with open(cam_path, 'r') as f:
            cam = json.load(f)
        context['image_size'] = (cam.get('image_width'), cam.get('image_height'))

    return context


def validate_and_clamp_velocities(velocities, n_objects, max_vel=20.0,
                                   object_labels=None):
    """
    验证和 clamp GPT 返回的速度

    步骤:
    1. 对已知不可动物体强制设为 0
    2. 每轴 clamp 到 [-max_vel, max_vel]
    3. 整体 magnitude clamp 到 max_vel * 1.5

    Args:
        velocities: (n_objects, 3) ndarray
        n_objects: int
        max_vel: 每轴最大速度
        object_labels: list of str

    Returns:
        velocities: 经过 clamp 的速度数组
        clamp_log: list of str, clamp 日志
    """
    velocities = np.array(velocities, dtype=np.float32)
    clamp_log = []

    for i in range(n_objects):
        label = object_labels[i] if object_labels else f'object_{i}'
        label_lower = label.lower().strip()

        # 1. 不可动物体检查
        is_immovable = any(kw in label_lower for kw in IMMOVABLE_KEYWORDS)
        if is_immovable and np.any(np.abs(velocities[i]) > 0.01):
            msg = f"  [CLAMP] 物体 {i} \"{label}\" 是不可动物体, 速度 [{velocities[i, 0]:.2f},{velocities[i, 1]:.2f},{velocities[i, 2]:.2f}] → [0,0,0]"
            clamp_log.append(msg)
            velocities[i] = 0.0
            continue

        # 2. 每轴 clamp
        for axis, axis_name in enumerate(['vx', 'vy', 'vz']):
            if abs(velocities[i, axis]) > max_vel:
                old = velocities[i, axis]
                velocities[i, axis] = np.clip(old, -max_vel, max_vel)
                msg = f"  [CLAMP] 物体 {i} \"{label}\" {axis_name}: {old:.2f} → {velocities[i, axis]:.2f}"
                clamp_log.append(msg)

        # 3. Magnitude clamp
        mag = np.linalg.norm(velocities[i])
        mag_limit = max_vel * 1.5
        if mag > mag_limit:
            old_vel = velocities[i].copy()
            velocities[i] = velocities[i] / mag * mag_limit
            msg = f"  [CLAMP] 物体 {i} \"{label}\" magnitude: {mag:.2f} → {mag_limit:.2f}"
            clamp_log.append(msg)

    return velocities, clamp_log


def parse_gpt_velocity_response(response_text, n_objects,
                                 default_angular_speed=7.0):
    """
    解析 GPT 返回的速度响应

    支持两种格式:
    1. JSON 格式 (推荐): {"velocities": [{"object_id": 0, "vx": ..., "vy": ..., "vz": ..., "reasoning": ...}]}
    2. 回退: velocity_spec 字符串格式 "0:vx,vy,vz;1:vx,vy,vz"

    Returns:
        velocities: (n_objects, 3) ndarray
        angular_velocity_spec: str 或 None (角速度规格字符串)
        reasoning: list of str 或 None
    """
    import re

    velocities = np.zeros((n_objects, 3), dtype=np.float32)
    reasoning = [None] * n_objects
    angular_velocity_spec = None

    text = response_text.strip()

    # 去除 markdown code fences
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # rotation_axis → (wx, wy, wz) 方向映射
    # 推导: omega=(0,0,wz), r_top=(0,h,0) → v_top = omega×r = (-wz*h, 0, 0)
    # tilt_left 需 top 往 +X(左) → -wz*h > 0 → wz < 0
    AXIS_MAP = {
        "roll_right":    (0, 0, 1),     # 向右滚(-X): wz>0
        "roll_left":     (0, 0, -1),    # 向左滚(+X): wz<0
        "roll_forward":  (1, 0, 0),     # 向前滚(+Z): wx>0
        "roll_backward": (-1, 0, 0),    # 向后滚(-Z): wx<0
        "tilt_left":     (0, 0, -1),    # 向左倾倒: top→+X, wz<0
        "tilt_right":    (0, 0, 1),     # 向右倾倒: top→-X, wz>0
        "tilt_forward":  (1, 0, 0),     # 向前倾倒: top→+Z, wx>0
        "tilt_backward": (-1, 0, 0),    # 向后倾倒: top→-Z, wx<0
        "spin_cw":       (0, -1, 0),
        "spin_ccw":      (0, 1, 0),
    }

    # 尝试 JSON 解析
    try:
        data = json.loads(text)

        vel_list = None
        if isinstance(data, dict) and 'velocities' in data:
            vel_list = data['velocities']
        elif isinstance(data, list):
            vel_list = data

        if vel_list:
            angular_parts = []
            for item in vel_list:
                obj_id = item.get('object_id', item.get('id', -1))
                if 0 <= obj_id < n_objects:
                    velocities[obj_id, 0] = float(item.get('vx', 0))
                    velocities[obj_id, 1] = float(item.get('vy', 0))
                    velocities[obj_id, 2] = float(item.get('vz', 0))
                    reasoning[obj_id] = item.get('reasoning', None)

                    # 解析角速度
                    if item.get('needs_rotation', False):
                        axis = item.get('rotation_axis', '')
                        direction = AXIS_MAP.get(axis, None)
                        if direction:
                            speed = default_angular_speed
                            wx = direction[0] * speed
                            wy = direction[1] * speed
                            wz = direction[2] * speed
                            angular_parts.append(f"{obj_id}:{wx:.1f},{wy:.1f},{wz:.1f}")
                            print(f"  角速度: 物体 {obj_id} → {axis} → omega=[{wx:.1f},{wy:.1f},{wz:.1f}]")

            if angular_parts:
                angular_velocity_spec = ";".join(angular_parts)

            return velocities, angular_velocity_spec, reasoning

    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # 回退: 尝试解析 velocity_spec 字符串格式
    # 匹配模式: 0:1.0,-2.0,0.5 或 0:1.0,-2.0,0.5;1:0,0,0
    spec_pattern = r'(\d+)\s*:\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)'
    matches = re.findall(spec_pattern, text)

    if matches:
        for m in matches:
            obj_id = int(m[0])
            if 0 <= obj_id < n_objects:
                velocities[obj_id] = [float(m[1]), float(m[2]), float(m[3])]
        return velocities, None, None

    # 最后尝试: 找任何 3 个连续数字作为速度
    num_pattern = r'[-+]?\d*\.?\d+'
    numbers = re.findall(num_pattern, text)
    if len(numbers) >= 3:
        # 只给第一个物体赋速度
        velocities[0] = [float(numbers[0]), float(numbers[1]), float(numbers[2])]
        return velocities, None, None

    print(f"  警告: 无法解析 GPT 响应: {text[:200]}")
    return velocities, None, None


def _build_system_prompt(max_vel, enable_angular_velocity=False):
    """构建 GPT system prompt"""
    base = f"""You are a physics simulation expert. You analyze 3D scenes and assign physically plausible initial velocities to objects for a particle-based physics simulation (Graph Network Simulator).

COORDINATE SYSTEM:
- Y-up: gravity acts in -Y direction
- Positive X = leftward in image, Negative X = rightward in image
- Positive Z = forward (away from camera), Negative Z = backward (toward camera)
- Scene is normalized: typical object height ~ 0.1-0.5 units
- Example: "move left to right" means negative X velocity (vx < 0)

VELOCITY CONSTRAINTS:
- Unit: meters per second in normalized simulation space
- Valid range per axis: [-{max_vel}, {max_vel}]
- Speed categories:
  * Very slow (0~3 m/s): gentle movements, slow slides, soft drops
  * Slow (3~6 m/s): normal drops, moderate pushes
  * Medium (6~10 m/s): normal throws, typical impacts
  * Fast (10~15 m/s): strong impacts, fast throws, forceful collisions
  * Very fast (15~20 m/s): extreme impacts, explosive launches
- Recommended magnitudes by motion type:
  * Stationary: 0, 0, 0
  * Gentle fall (gravity only): vy = -1.0 to -3.0 (very slow)
  * Normal drop: vy = -3.0 to -6.0 (slow)
  * Thrown/fast fall: vy = -6.0 to -10.0 (medium)
  * Horizontal push/slide: vx or vz = 1.0 to 6.0 (very slow to slow)
  * Rolling: vx or vz = 1.0 to 3.0 (very slow)
  * Strong impact: up to 10.0-15.0 per axis (fast)
  * Extreme impact: up to 15.0-20.0 per axis (very fast)

CRITICAL RULE — INITIAL STATE ONLY:
You are assigning INITIAL velocities BEFORE the simulation starts. The physics engine (GNS) will handle all collisions, reactions, and secondary effects automatically.
- Only assign non-zero velocity to objects that are ACTIVELY MOVING at the start (e.g., a thrown ball, a pushed object).
- Objects that are STATIONARY and will be HIT by other objects must have velocity (0, 0, 0). Their motion after collision will be computed by the physics engine.
- Example: "ball knocks over a vase" → ball gets velocity, vase gets (0,0,0). The vase toppling is a RESULT of the collision, not an initial condition.
- Example: "dominos fall in sequence" → only the FIRST domino gets a push velocity, all others are (0,0,0).
- Do NOT predict post-collision outcomes. Do NOT give target objects any "reaction" velocity.

IMPORTANT: Any object can be moved if the user's motion description explicitly says it is moving at the start. Assign velocities based on the user's intent, not on object category. If the user says "the chair moves left", give the chair a velocity. Only keep objects stationary if they are not described as actively moving at the start."""

    if enable_angular_velocity:
        base += """

ANGULAR VELOCITY (rotation):

In addition to translational velocity, determine whether each object needs an initial angular velocity (rotation).

CRITICAL: Same initial-state-only rule applies to angular velocity. Only assign rotation to objects that are ALREADY rotating at the start (e.g., a rolling ball). Do NOT assign rotation to objects that will START rotating only AFTER being hit — the physics engine handles that.

Criteria:
- Ball/sphere/cylinder actively rolling on ground at the start → needs rotation
- Object being actively pushed/tipped at the start → needs rotation
- Object that will be HIT and then topple (passive target) → NO rotation (physics engine handles it)
- Object in free fall / stationary / floating / waiting to be hit → no rotation
- Ground/wall/table/floor → no rotation

You do NOT need to compute the exact angular velocity value. Just determine:
1. Whether the object needs rotation (needs_rotation: true/false)
2. The rotation direction (rotation_axis), choose from:
   - "roll_right": rolling rightward in image (negative X direction)
   - "roll_left": rolling leftward in image (positive X direction)
   - "roll_forward": rolling away from camera (positive Z direction)
   - "roll_backward": rolling toward camera (negative Z direction)
   - "tilt_left": tilting/toppling leftward (positive X direction, like a domino falling left)
   - "tilt_right": tilting/toppling rightward (negative X direction)
   - "tilt_forward": tilting forward (positive Z direction)
   - "tilt_backward": tilting backward (negative Z direction)
   - "spin_cw": spinning clockwise around Y axis (top view)
   - "spin_ccw": spinning counter-clockwise around Y axis (top view)

RESPONSE FORMAT (strict JSON, no markdown fences, no extra text):
{
  "velocities": [
    {
      "object_id": 0,
      "label": "object_name",
      "vx": 0.0,
      "vy": 0.0,
      "vz": 0.0,
      "needs_rotation": true,
      "rotation_axis": "tilt_left",
      "reasoning": "brief explanation"
    }
  ]
}"""
    else:
        base += """

RESPONSE FORMAT (strict JSON, no markdown fences, no extra text):
{
  "velocities": [
    {
      "object_id": 0,
      "label": "object_name",
      "vx": 0.0,
      "vy": -2.0,
      "vz": 0.0,
      "reasoning": "brief explanation"
    }
  ]
}"""

    return base


def _build_user_message(motion_prompt, objects, positions=None,
                         image_b64=None, max_vel=20.0):
    """构建 GPT user message"""
    parts = []

    # 用户意图
    if motion_prompt:
        parts.append(f"USER MOTION DESCRIPTION: {motion_prompt}")
        parts.append("")

    # 场景物体
    parts.append("SCENE OBJECTS:")
    for i, obj in enumerate(objects):
        label = obj.get('label', f'object_{i}')
        bbox_info = ""
        if 'x0' in obj:
            w = obj['x1'] - obj['x0']
            h = obj['y1'] - obj['y0']
            bbox_info = f" (bbox: [{obj['x0']},{obj['y0']},{obj['x1']},{obj['y1']}], size: {w}x{h}px)"
        parts.append(f"  Object {i}: \"{label}\"{bbox_info}")
    parts.append("")

    # 3D 位置信息
    if positions and len(positions) == len(objects):
        parts.append("3D POSITIONS (Y-up, ground at Y~0):")
        for i, (obj, pos) in enumerate(zip(objects, positions)):
            label = obj.get('label', f'object_{i}')
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                parts.append(f"  Object {i} \"{label}\": centroid=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        parts.append("")

    # 约束提醒
    parts.append(f"VELOCITY RANGE: each axis must be in [-{max_vel}, {max_vel}]")
    if not motion_prompt:
        parts.append("No specific motion was requested. Infer reasonable physical behavior from the scene.")
        parts.append("Most objects should be stationary (0,0,0) unless they appear to be falling or in motion.")
    parts.append("Respond with ONLY the JSON object, nothing else.")

    text = "\n".join(parts)

    # 如果有图像，构建 multimodal message
    if image_b64:
        return [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": text}
        ]
    return text


def infer_velocity_with_chatgpt(scene_dir, image_path=None,
                                 gpt_model='gpt-5',
                                 motion_prompt=None,
                                 max_velocity=20.0,
                                 enable_angular_velocity=False,
                                 default_angular_speed=7.0):
    """
    使用 ChatGPT 推断物体初始速度（和角速度）

    结合图像、场景信息和用户文本提示，为每个物体赋予合理的初始速度。
    速度范围经过 clamp 确保在 GNS 模型训练分布内。

    Args:
        scene_dir: 3DGS 场景目录 (包含 detection_result.json)
        image_path: 原始图像路径 (可选, 用于 GPT Vision)
        gpt_model: GPT 模型名称
        motion_prompt: 用户文本提示 (如 "bear falls off the bed")
        max_velocity: 每轴最大速度上限 (默认 5.0)
        enable_angular_velocity: 是否启用角速度推断
        default_angular_speed: 角速度大小 (rad/s, 默认 7.0)

    Returns:
        velocity_spec: 速度规格字符串, 如 "0:0,-2,0;1:0,0,0"
        angular_velocity_spec: 角速度规格字符串, 如 "0:0,0,7" (或 None)
        reasoning_dict: dict, 每个物体的速度推理 (或 None)
    """
    print("\n  ChatGPT 智能速度推断...")
    if motion_prompt:
        print(f"  用户意图: \"{motion_prompt}\"")

    # 收集场景上下文
    context = collect_scene_context(scene_dir)
    objects = context['objects']

    if not objects:
        print("  警告: 没有检测到物体, 使用零速度")
        return None, None, None

    n_objects = len(objects)
    object_labels = [obj.get('label', f'object_{i}') for i, obj in enumerate(objects)]
    print(f"  场景物体: {object_labels}")

    # 查找图像
    if not image_path:
        image_path = find_scene_image(scene_dir)

    # 编码图像 (如果有)
    image_b64 = None
    if image_path and os.path.exists(image_path):
        import base64
        from PIL import Image as PILImage
        # 压缩到 1024px 以内减少 token 消耗
        img = PILImage.open(image_path).convert('RGB')
        max_dim = max(img.size)
        if max_dim > 1024:
            scale = 1024 / max_dim
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, PILImage.LANCZOS)
        import io
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        print(f"  使用图像: {image_path} ({img.size[0]}x{img.size[1]})")

    # 构建 GPT 消息
    system_prompt = _build_system_prompt(max_velocity,
                                         enable_angular_velocity=enable_angular_velocity)
    user_content = _build_user_message(
        motion_prompt=motion_prompt,
        objects=objects,
        positions=context.get('positions'),
        image_b64=image_b64,
        max_vel=max_velocity,
    )

    if not _OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, or pass --velocity "
            "(and optionally --angular-velocity) together with --no-gpt.")

    # 调用 GPT API (最多重试 2 次)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            import openai
            client = openai.OpenAI(api_key=_OPENAI_API_KEY)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            print(f"  调用 {gpt_model}... (尝试 {attempt + 1}/{max_retries + 1})")
            # 推理型模型(gpt-5, o1, o3)需要更大 token 预算（reasoning tokens 占大头）
            is_reasoning_model = any(k in gpt_model for k in ('gpt-5', 'o1', 'o3'))
            api_kwargs = dict(
                model=gpt_model,
                messages=messages,
                max_completion_tokens=16000 if is_reasoning_model else 2000,
            )
            # 只有非推理型模型(gpt-4o等)支持 temperature
            if not is_reasoning_model:
                api_kwargs['temperature'] = 0.3
            response = client.chat.completions.create(**api_kwargs)

            raw_response = response.choices[0].message.content.strip()
            print(f"  GPT 原始响应: {raw_response[:300]}")

            # 解析响应
            velocities, gpt_angular_spec, reasoning = parse_gpt_velocity_response(
                raw_response, n_objects,
                default_angular_speed=default_angular_speed)

            # 验证和 clamp
            velocities, clamp_log = validate_and_clamp_velocities(
                velocities, n_objects, max_vel=max_velocity,
                object_labels=object_labels)

            for msg in clamp_log:
                print(msg)

            # 构建 velocity_spec 字符串
            spec_parts = []
            for i in range(n_objects):
                spec_parts.append(f"{i}:{velocities[i, 0]:.3f},{velocities[i, 1]:.3f},{velocities[i, 2]:.3f}")
            velocity_spec = ";".join(spec_parts)

            # 打印结果摘要
            print(f"\n  速度分配结果:")
            print(f"  {'─' * 60}")
            reasoning_dict = {}
            for i in range(n_objects):
                label = object_labels[i]
                v = velocities[i]
                r = reasoning[i] if reasoning else None
                mag = np.linalg.norm(v)
                status = "静止" if mag < 0.01 else f"运动 (|v|={mag:.2f})"
                reason_str = f" — {r}" if r else ""
                print(f"  物体 {i} \"{label}\": [{v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}] {status}{reason_str}")
                if r:
                    reasoning_dict[str(i)] = r
            print(f"  {'─' * 60}")
            print(f"  velocity_spec: {velocity_spec}")
            if gpt_angular_spec:
                print(f"  angular_velocity_spec: {gpt_angular_spec}")

            return velocity_spec, gpt_angular_spec, reasoning_dict if reasoning_dict else None

        except Exception as e:
            print(f"  警告: 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries:
                print("  重试中...")
                continue
            else:
                print("  所有尝试均失败, 使用零速度")
                return None, None, None


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='3DGS → Physics Simulation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 从已有 3DGS 场景运行 (最常用)
  python pipeline_3dgs_to_physics.py --scene-dir depth_ratio_bear/ -o output_physics

  # 从图像开始完整 Pipeline
  python pipeline_3dgs_to_physics.py --image bear.jpg -o output_physics

  # 设置初始速度 (物体0向下)
  python pipeline_3dgs_to_physics.py --scene-dir scene/ -o output/ --velocity "0:0,-2,0"

  # 使用 ChatGPT 推断速度 (自动模式)
  python pipeline_3dgs_to_physics.py --scene-dir scene/ -o output/ --chatgpt-velocity

  # 使用 text prompt 描述运动 (自动启用 ChatGPT)
  python pipeline_3dgs_to_physics.py --scene-dir scene/ -o output/ --motion-prompt "bear falls off the bed"

  # 自定义速度范围
  python pipeline_3dgs_to_physics.py --scene-dir scene/ -o output/ --motion-prompt "ball hits wall" --velocity-range 3.0

  # 自定义 rollout 参数
  python pipeline_3dgs_to_physics.py --scene-dir scene/ -o output/ --rollout-steps 100
        """)

    # 输入模式 (二选一)
    input_group = parser.add_argument_group('Input (choose one)')
    input_group.add_argument('--image', type=str, default=None,
                             help='Input image path (full pipeline)')
    input_group.add_argument('--scene-dir', type=str, default=None,
                             help='Existing 3DGS scene directory (skip reconstruction)')

    # 输出
    parser.add_argument('--output-dir', '-o', type=str, default='output_physics',
                        help='Output directory')

    # 3D 重建参数 (仅 --image 模式)
    recon_group = parser.add_argument_group('3D Reconstruction (--image mode)')
    recon_group.add_argument('--max-objects', type=int, default=5)
    recon_group.add_argument('--text-prompts', type=str, default=None,
                             help='Comma-separated SAM3 text prompts')
    recon_group.add_argument('--use-gpt', action='store_true', default=True,
                             help='Use GPT Vision for detection')
    recon_group.add_argument('--no-gpt', dest='use_gpt', action='store_false',
                             help='Disable all OpenAI calls. --image mode then requires '
                                  '--text-prompts, and initial velocities must be given via '
                                  '--velocity (optionally --angular-velocity)')
    recon_group.add_argument('--gpt-model', type=str, default='gpt-5.2',
                             help='GPT model for detection')
    recon_group.add_argument('--max-depth-ratio', type=float, default=2,
                             help='深度过滤倍数（相对于最近物体，默认 2）')
    recon_group.add_argument('--no-depth-filter', action='store_true',
                             help='禁用深度过滤（重建所有检测到的物体）')
    recon_group.add_argument('--no-layout-optimize', action='store_true',
                             help='禁用 Layout Post-Optimization（默认启用）')
    recon_group.add_argument('--seed', type=int, default=42,
                             help='随机种子 (default: 42)')
    recon_group.add_argument('--sam3-threshold', type=float, default=0.5,
                             help='SAM3 检测阈值 (default: 0.5)')

    # 粒子化参数
    particle_group = parser.add_argument_group('Particle Conversion')
    particle_group.add_argument('--use-mesh', dest='use_mesh', action='store_true',
                                default=True,
                                help='使用 Mesh 进行粒子转换 (默认, 更可靠的内部填充)')
    particle_group.add_argument('--use-3dgs', dest='use_mesh', action='store_false',
                                help='使用旧的 3DGS 方案进行粒子转换 (回退)')
    # --no-repair-mesh 已移除: binvox -dc 不需要 mesh 修复
    particle_group.add_argument('--spacing', type=float, default=0.05,
                                help='Particle spacing (default: 0.05)')
    particle_group.add_argument('--opacity-threshold', type=float, default=0.5,
                                help='Opacity filter threshold (仅 --use-3dgs 模式)')
    particle_group.add_argument('--max-particles', type=int, default=3000,
                                help='Max particles per object')
    particle_group.add_argument('--min-particles', type=int, default=150,
                                help='Min particles per object (仅 --use-3dgs 模式)')
    particle_group.add_argument('--discard-threshold', type=int, default=10,
                                help='Discard objects with fewer particles than this')
    particle_group.add_argument('--no-fill', action='store_true',
                                help='Skip interior filling (仅 --use-3dgs 模式)')
    particle_group.add_argument('--fine-factor', type=int, default=2,
                                help='Fine grid factor for two-pass filling (仅 --use-3dgs 模式)')
    particle_group.add_argument('--no-fine-fill', action='store_true',
                                help='Disable two-pass fine filling (仅 --use-3dgs 模式)')
    particle_group.add_argument('--no-convex-hull', action='store_true',
                                help='Disable convex hull fallback filling (仅 --use-3dgs 模式)')
    particle_group.add_argument('--max-particles-per-obj', type=int, default=600,
                                help='单物体最大粒子数上限，超过则缩小场景 (仅 Mesh 模式)')
    particle_group.add_argument('--min-particles-per-obj', type=int, default=20,
                                help='最大物体粒子数下限，低于则放大场景 (仅 Mesh 模式)')
    particle_group.add_argument('--target-particles', type=int, default=350,
                                help='放大场景时最大物体目标粒子数 (仅 Mesh 模式)')
    particle_group.add_argument('--thicken', action='store_true',
                                help='自动检测并加厚薄物体（如手机），使其至少有 N 层粒子')
    particle_group.add_argument('--thicken-min-layers', type=int, default=3,
                                help='加厚目标最少粒子层数 (默认 3, 即厚度 >= 3*spacing)')
    particle_group.add_argument('--align-z', action='store_true',
                                help='将所有物体的 Z 中心对齐（修正深度估计偏差，多米诺场景必用）')

    # 速度参数
    vel_group = parser.add_argument_group('Velocity')
    vel_group.add_argument('--velocity', type=str, default=None,
                           help='Velocity spec: "obj:vx,vy,vz" (semicolon-separated)')
    vel_group.add_argument('--chatgpt-velocity', action='store_true',
                           help='Use ChatGPT to infer initial velocities')
    vel_group.add_argument('--motion-prompt', type=str, default=None,
                           help='描述物体运动行为的文本提示 '
                                '(如 "bear falls off the bed", 自动启用 ChatGPT)')
    vel_group.add_argument('--velocity-range', type=float, default=20.0,
                           help='每轴最大允许速度 m/s (默认: 20.0)')
    vel_group.add_argument('--gpt-velocity-model', type=str, default='gpt-5',
                           help='用于速度推断的 GPT 模型 (默认: gpt-5)')
    vel_group.add_argument('--angular-velocity', type=str, default=None,
                           help='手动角速度: "obj:wx,wy,wz" (分号分隔多个物体)')
    vel_group.add_argument('--enable-angular-velocity', action='store_true',
                           help='启用角速度推断（让 GPT 判断哪些物体需要旋转）。'
                                '需配合 --motion-prompt 或 --chatgpt-velocity 使用')
    vel_group.add_argument('--default-angular-speed', type=float, default=7.0,
                           help='GPT 推断时使用的默认角速度大小 (rad/s)，'
                                '默认 7.0（匹配 Physion 训练数据）')

    # Rollout 参数
    rollout_group = parser.add_argument_group('GNS Rollout')
    rollout_group.add_argument('--gns-ckpt', '--model-path', dest='model_path', type=str,
                               default=DEFAULT_GNS_CKPT,
                               help='GNS model checkpoint (default: '
                                    'checkpoints/gns/without_Drape/best_model.pth under the release root)')
    rollout_group.add_argument('--rollout-steps', type=int, default=167,
                               help='Number of rollout steps (167 steps = 5s, matches Wan-Move)')
    rollout_group.add_argument('--training-fpt', type=int, default=3,
                               help='Frame interval (must match training)')
    rollout_group.add_argument('--neighbor-radius', type=float, default=0.08,
                               help='Neighbor search radius')
    rollout_group.add_argument('--no-ransac', dest='ransac', action='store_false',
                               default=True, help='Disable rigid body projection')

    # 可视化参数
    vis_group = parser.add_argument_group('Visualization')
    vis_group.add_argument('--no-visualize', action='store_true',
                           help='Skip visualization')
    vis_group.add_argument('--fps', type=int, default=6,
                           help='GIF frame rate')

    # 地面粒子参数
    ground_group = parser.add_argument_group('Ground Plane')
    ground_group.add_argument('--no-ground', dest='add_ground',
                              action='store_false', default=True,
                              help='不添加地面粒子（默认添加）')
    ground_group.add_argument('--ground-margin', type=float, default=0.3,
                              help='地面 XZ 方向超出物体范围的额外覆盖')
    ground_group.add_argument('--ground-clearance', type=float, default=0.025,
                              help='物体与地面的初始间距 (默认 0.025, 匹配训练数据落地间距中位数)')

    # 相机还原参数
    cam_group = parser.add_argument_group('Camera Restoration')
    cam_group.add_argument('--no-camera', action='store_true',
                           help='Skip camera restoration')

    # Mesh 线框可视化
    wf_group = parser.add_argument_group('Wireframe Visualization')
    wf_group.add_argument('--wireframe', action='store_true',
                          help='生成 mesh 线框投影可视化 (camera_view_wireframe.gif)')
    wf_group.add_argument('--wireframe-faces', type=int, default=3000,
                          help='线框 mesh 简化目标面数 (默认 3000)')
    wf_group.add_argument('--no-backface-cull', action='store_true',
                          help='禁用线框背面剔除')
    wf_group.add_argument('--wireframe-particles', action='store_true',
                          help='线框模式同时显示粒子散点')

    # Wan-Move 导出
    wan_group = parser.add_argument_group('Wan-Move Export')
    wan_group.add_argument('--export-wan-move', action='store_true',
                           help='导出 Wan-Move 格式轨迹文件')
    wan_group.add_argument('--wan-move-mode', type=str,
                           default='centroid', choices=['centroid', 'dense'],
                           help='轨迹模式: centroid(质心) 或 dense(密集采样)')
    wan_group.add_argument('--wan-move-samples', type=int, default=10,
                           help='dense 模式下每个物体的采样点数')
    wan_group.add_argument('--depth-tolerance', type=float, default=0.05,
                           help='z-buffer 遮挡检测的深度容差比例 (默认 0.05)')
    wan_group.add_argument('--min-visibility', type=float, default=0.7,
                           help='dense 模式最低可见性阈值 (默认 0.7)')
    wan_group.add_argument('--wan-move-prompt', type=str, default=None,
                           help='Wan-Move 视频生成的文本提示')
    wan_group.add_argument('--wan-move-steps', type=int, default=40,
                           help='Wan-Move 采样步数 (默认 40)')
    wan_group.add_argument('--wan-move-guide', type=float, default=5.0,
                           help='Wan-Move CFG scale (默认 5.0)')
    wan_group.add_argument('--wan-move-seed', type=int, default=42,
                           help='Wan-Move 随机种子')
    wan_group.add_argument('--wan-move-size', type=str, default='480*832',
                           choices=['480*832', '832*480'],
                           help='Wan-Move 输出分辨率')
    wan_group.add_argument('--no-wan-move-video', action='store_true',
                           help='仅导出轨迹，不生成视频（保留旧行为）')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-rollout', action='store_true',
                        help='Only convert to particles, skip rollout')

    args = parser.parse_args()

    # 验证输入
    if args.image is None and args.scene_dir is None:
        parser.error("必须指定 --image 或 --scene-dir")

    # --motion-prompt 自动启用 chatgpt_velocity
    if args.motion_prompt and not args.chatgpt_velocity:
        args.chatgpt_velocity = True

    # OpenAI usage check (the key is read from the OPENAI_API_KEY environment variable)
    needs_gpt_detection = args.image is not None and not args.text_prompts
    needs_gpt_velocity = args.chatgpt_velocity and args.velocity is None
    if not args.use_gpt:
        if needs_gpt_detection:
            parser.error("--no-gpt requires --text-prompts in --image mode")
        if needs_gpt_velocity:
            parser.error("--no-gpt disables GPT velocity inference: pass --velocity "
                         "(optionally --angular-velocity), or drop --motion-prompt/--chatgpt-velocity")
        args.chatgpt_velocity = False
    elif (needs_gpt_detection or needs_gpt_velocity) and not os.environ.get('OPENAI_API_KEY'):
        parser.error("OPENAI_API_KEY is not set (required for GPT object detection / velocity "
                     "inference). Export it, or use --no-gpt with --text-prompts and --velocity.")

    # Wan-Move export (Step 6/7) is not part of the open-source release
    if args.export_wan_move:
        import importlib.util
        if importlib.util.find_spec('export_wan_move_tracks') is None:
            parser.error("--export-wan-move requires export_wan_move_tracks.py (Wan-Move export), "
                         "which is not included in this release. Run without --export-wan-move.")

    # 设置目录结构
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.image is not None:
        scene_dir = os.path.join(output_dir, 'scene_3dgs')
    else:
        scene_dir = args.scene_dir

    trial_dir = os.path.join(output_dir, 'trial')
    rollout_dir = os.path.join(output_dir, 'rollout')
    vis_dir = os.path.join(output_dir, 'visualization')

    # 确定速度模式描述
    if args.velocity:
        vel_mode = f'manual ({args.velocity})'
    elif args.motion_prompt:
        vel_mode = f'ChatGPT + prompt ("{args.motion_prompt}")'
    elif args.chatgpt_velocity:
        vel_mode = 'ChatGPT (auto)'
    else:
        vel_mode = 'zero'

    # 记录配置
    config = {
        'image': args.image,
        'scene_dir': scene_dir,
        'output_dir': output_dir,
        'max_depth_ratio': args.max_depth_ratio,
        'no_depth_filter': args.no_depth_filter,
        'layout_optimize': not args.no_layout_optimize,
        'seed': args.seed,
        'sam3_threshold': args.sam3_threshold,
        'use_gpt': args.use_gpt,
        'use_mesh': args.use_mesh,
        'spacing': args.spacing,
        'velocity': args.velocity,
        'chatgpt_velocity': args.chatgpt_velocity,
        'motion_prompt': args.motion_prompt,
        'velocity_range': args.velocity_range,
        'gpt_velocity_model': args.gpt_velocity_model,
        'rollout_steps': args.rollout_steps,
        'model_path': args.model_path,
        'training_fpt': args.training_fpt,
        'add_ground': args.add_ground,
        'ground_margin': args.ground_margin,
        'ground_clearance': args.ground_clearance,
        'export_wan_move': args.export_wan_move,
        'wan_move_mode': args.wan_move_mode,
        'wan_move_samples': args.wan_move_samples,
        'depth_tolerance': args.depth_tolerance,
        'min_visibility': args.min_visibility,
        'wan_move_prompt': args.wan_move_prompt,
        'wan_move_steps': args.wan_move_steps,
        'wan_move_guide': args.wan_move_guide,
        'wan_move_seed': args.wan_move_seed,
        'wan_move_size': args.wan_move_size,
        'no_wan_move_video': args.no_wan_move_video,
        'enable_angular_velocity': args.enable_angular_velocity,
        'default_angular_speed': args.default_angular_speed,
    }
    with open(os.path.join(output_dir, 'pipeline_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    start_time = time.time()

    print("=" * 70)
    print("  3DGS → Physics Simulation Pipeline")
    print("=" * 70)
    print(f"  Input:  {args.image or args.scene_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Velocity: {vel_mode}")
    if args.motion_prompt:
        print(f"  Motion:   \"{args.motion_prompt}\"")
    print(f"  Vel Range: ±{args.velocity_range} m/s")
    if args.enable_angular_velocity:
        print(f"  Angular:  enabled (speed={args.default_angular_speed} rad/s)")
    print(f"  Particles: {'Mesh (default)' if args.use_mesh else '3DGS (legacy)'}")
    print(f"  Particle limits: max={args.max_particles_per_obj}, min={args.min_particles_per_obj}, target={args.target_particles}")
    print(f"  Ground:   {'enabled (margin={:.1f})'.format(args.ground_margin) if args.add_ground else 'disabled'}")
    print(f"  Rollout: {args.rollout_steps} steps")

    # ========================================
    # Step 1: 3D 重建 (可选)
    # ========================================
    if args.image is not None:
        text_prompts = None
        if args.text_prompts:
            text_prompts = [p.strip() for p in args.text_prompts.split(',')]

        scene_dir = step1_reconstruct_scene(
            image_path=args.image,
            scene_dir=scene_dir,
            max_objects=args.max_objects,
            device=args.device,
            text_prompts=text_prompts,
            use_gpt=args.use_gpt,
            gpt_model=args.gpt_model,
            max_depth_ratio=args.max_depth_ratio,
            no_depth_filter=args.no_depth_filter,
            layout_optimize=not args.no_layout_optimize,
            seed=args.seed,
            sam3_threshold=args.sam3_threshold,
        )

    # ========================================
    # Step 2: 3DGS → 粒子 → Physion
    # ========================================
    # fine_factor: 1 表示禁用两遍填充(单遍), >=2 表示启用
    fine_factor = 1 if args.no_fine_fill else args.fine_factor
    step2_convert_to_particles(
        scene_dir=scene_dir,
        trial_dir=trial_dir,
        velocity_spec=args.velocity,
        spacing=args.spacing,
        opacity_threshold=args.opacity_threshold,
        max_particles=args.max_particles,
        min_particles=args.min_particles,
        discard_threshold=args.discard_threshold,
        training_fpt=args.training_fpt,
        fill_interior=not args.no_fill,
        fine_factor=fine_factor,
        use_convex_hull=not args.no_convex_hull,
        chatgpt_velocity=args.chatgpt_velocity,
        motion_prompt=args.motion_prompt,
        max_velocity=args.velocity_range,
        image_path=args.image,
        gpt_model=args.gpt_velocity_model,
        add_ground=args.add_ground,
        ground_margin=args.ground_margin,
        ground_clearance=args.ground_clearance,
        use_mesh=args.use_mesh,
        max_particles_per_obj=args.max_particles_per_obj,
        min_particles_per_obj=args.min_particles_per_obj,
        target_particles=args.target_particles,
        enable_angular_velocity=args.enable_angular_velocity,
        default_angular_speed=args.default_angular_speed,
        manual_angular_velocity_spec=args.angular_velocity,
        thicken=args.thicken,
        thicken_min_layers=args.thicken_min_layers,
        align_z=args.align_z,
    )

    if args.skip_rollout:
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"  Pipeline 完成 (跳过 rollout)")
        print(f"  用时: {elapsed:.1f}s")
        print(f"  Trial: {trial_dir}")
        print(f"{'=' * 70}")
        return

    # ========================================
    # Step 3: GNS Rollout
    # ========================================
    positions, centroids = step3_run_rollout(
        trial_dir=trial_dir,
        model_path=args.model_path,
        rollout_dir=rollout_dir,
        rollout_steps=args.rollout_steps,
        training_fpt=args.training_fpt,
        device=args.device,
        neighbor_radius=args.neighbor_radius,
        ransac=args.ransac,
    )

    # ========================================
    # Step 4: 可视化
    # ========================================
    if not args.no_visualize:
        step4_visualize(
            trial_dir=trial_dir,
            rollout_dir=rollout_dir,
            vis_dir=vis_dir,
            training_fpt=args.training_fpt,
            fps=args.fps,
        )

    # ========================================
    # Step 5: 摄像头还原 + 原始视角渲染
    # ========================================
    if not args.no_camera:
        step5_restore_camera(
            scene_dir=scene_dir,
            trial_dir=trial_dir,
            rollout_dir=rollout_dir,
            vis_dir=vis_dir,
            image_path=args.image,
        )

    # ========================================
    # Step 5.5: Mesh 线框可视化 (可选)
    # ========================================
    if args.wireframe and not args.no_camera:
        from mesh_wireframe_vis import generate_wireframe_visualizations
        from camera_restore import CameraInPhysion

        # 加载相机和 rollout 数据
        camera_params_path = os.path.join(scene_dir, 'camera_params.json')
        coord_transform_path = os.path.join(trial_dir, 'coord_transform.json')
        rollout_pos_path = os.path.join(rollout_dir, 'rollout_positions.npz')
        phases_dict_path = os.path.join(trial_dir, 'phases_dict.pkl')

        if (os.path.exists(camera_params_path)
                and os.path.exists(coord_transform_path)
                and os.path.exists(rollout_pos_path)):
            wf_camera = CameraInPhysion.from_files(
                camera_params_path, coord_transform_path)
            wf_data = np.load(rollout_pos_path)
            wf_positions = [wf_data[f'step_{i}']
                            for i in range(len(wf_data.files))]
            with open(phases_dict_path, 'rb') as f:
                wf_phases = pickle.load(f)

            # 加载原图
            wf_image = None
            wf_img_path = args.image
            if not wf_img_path or not os.path.exists(str(wf_img_path)):
                wf_img_path = find_scene_image(scene_dir)
            if wf_img_path and os.path.exists(wf_img_path):
                from PIL import Image as PILImage
                wf_image = np.array(PILImage.open(wf_img_path).convert("RGB"))

            # 生成地面 mesh (如果还没有)
            ground_mesh_path = os.path.join(scene_dir, 'ground_mesh_posed.ply')
            if not os.path.exists(ground_mesh_path) and args.add_ground:
                from generate_ground_mesh import generate_ground_mesh
                generate_ground_mesh(
                    scene_dir=args.output_dir,
                    output_path=ground_mesh_path,
                )

            generate_wireframe_visualizations(
                positions_list=wf_positions,
                phases_dict=wf_phases,
                camera=wf_camera,
                scene_dir=scene_dir,
                trial_dir=trial_dir,
                output_dir=vis_dir,
                orig_image=wf_image,
                fps=8,
                target_faces=args.wireframe_faces,
                backface_cull=not args.no_backface_cull,
                show_particles=args.wireframe_particles,
            )
        else:
            print("  警告: 线框可视化所需文件不完整, 跳过")

    # ========================================
    # Step 6: 导出 Wan-Move 轨迹 (可选)
    # ========================================
    if args.export_wan_move:
        from export_wan_move_tracks import export_wan_move_tracks
        from export_wan_move_tracks import generate_wan_move_command

        wan_move_dir = os.path.join(output_dir, 'wan_move')

        wan_result = export_wan_move_tracks(
            rollout_dir=rollout_dir,
            trial_dir=trial_dir,
            scene_dir=scene_dir,
            output_dir=wan_move_dir,
            image_path=args.image,
            mode=args.wan_move_mode,
            samples_per_object=args.wan_move_samples,
            prompt=args.wan_move_prompt,
            depth_tolerance=args.depth_tolerance,
            min_visibility=args.min_visibility,
        )

        if wan_result:
            cmd = generate_wan_move_command(
                wan_move_dir, prompt=args.wan_move_prompt)
            print(f"\n  Wan-Move 备用命令:\n  {cmd}")

    # ========================================
    # Step 7: Wan-Move 视频生成 (可选)
    # ========================================
    if args.export_wan_move and not args.no_wan_move_video:
        from export_wan_move_tracks import run_wan_move_generate

        wan_move_dir = os.path.join(output_dir, 'wan_move')

        video_path = run_wan_move_generate(
            wan_move_dir=wan_move_dir,
            prompt=args.wan_move_prompt,
            size=args.wan_move_size,
            sample_steps=args.wan_move_steps,
            sample_shift=3.0 if '480' in args.wan_move_size else 5.0,
            guide_scale=args.wan_move_guide,
            seed=args.wan_move_seed,
            t5_cpu=True,
            offload_model=True,
            dtype='bf16',
        )

        if video_path:
            print(f"\n  Wan-Move 视频已生成: {video_path}")

    # ========================================
    # 完成
    # ========================================
    elapsed = time.time() - start_time

    # 打印摘要
    centroids_array = np.array(centroids)
    phases_dict_path = os.path.join(trial_dir, 'phases_dict.pkl')
    with open(phases_dict_path, 'rb') as f:
        phases_dict = pickle.load(f)

    print(f"\n{'=' * 70}")
    print(f"  Pipeline 完成!")
    print(f"  用时: {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"  输出目录: {output_dir}")
    print(f"    scene:    {scene_dir}")
    print(f"    trial:    {trial_dir}")
    print(f"    rollout:  {rollout_dir}")
    if not args.no_visualize:
        print(f"    vis:      {vis_dir}")
    if args.export_wan_move:
        wan_move_dir = os.path.join(output_dir, 'wan_move')
        print(f"    wan_move: {wan_move_dir}")
    print(f"\n  物体轨迹:")
    for obj_id in range(phases_dict['n_objects']):
        name = phases_dict['instance'][obj_id]
        if isinstance(name, bytes):
            name = name.decode()
        traj = centroids_array[:, obj_id, :]
        displacement = np.linalg.norm(traj[-1] - traj[0])
        print(f"    {name}: start={traj[0]} → end={traj[-1]}, "
              f"displacement={displacement:.4f}")


if __name__ == '__main__':
    main()
