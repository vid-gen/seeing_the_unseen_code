"""
visualize_rollout.py - Rollout 结果可视化模块

功能:
- 多视角粒子轨迹 GIF
- 每物体质心轨迹曲线
- 位移/速度统计图
- 多帧快照对比
"""

import os
import argparse
import json
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 颜色方案
# ============================================================
OBJECT_COLORS = [
    '#e74c3c',  # red
    '#3498db',  # blue
    '#2ecc71',  # green
    '#f39c12',  # orange
    '#9b59b6',  # purple
    '#1abc9c',  # teal
    '#e67e22',  # dark orange
    '#34495e',  # dark gray
]


def compute_camera_view_angles(camera):
    """从 CameraInPhysion 计算匹配原图视角的 matplotlib view_init 参数.

    matplotlib 坐标映射 (render_particles_frame):
        matplot_X = -Physion_X (right)
        matplot_Y = Physion_Z  (depth/forward)
        matplot_Z = Physion_Y  (up)

    Returns:
        (elev, azim) for ax.view_init()
    """
    pos = np.array(camera.position)  # Physion 空间相机位置
    # 转到 matplotlib 坐标: matX=-physX, matY=physZ, matZ=physY
    mx, my, mz = -pos[0], pos[2], pos[1]
    # azim: 观察者在 XY 平面的角度 (从 +X 轴量起)
    azim = np.degrees(np.arctan2(my, mx))
    # elev: 基于相机高度计算，但适当提高以补偿 matplotlib 正交投影
    # 与透视投影 (FOV ~60°) 的视觉差异
    horiz_dist = np.sqrt(mx**2 + my**2)
    elev = np.degrees(np.arctan2(mz, horiz_dist))
    elev = max(elev, 15)  # 至少 15° 保证 3D 图可读性
    return elev, azim


def get_color(obj_id):
    """获取物体颜色"""
    return OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]


# ============================================================
# 核心渲染函数
# ============================================================
def render_particles_frame(ax, positions, instance_idx, n_objects,
                           view_elev=25, view_azim=45,
                           x_range=None, y_range=None, z_range=None,
                           point_size=3, alpha=0.6, title=''):
    """渲染单帧粒子 3D 散点图 (Y-up → matplotlib Z, X 取反匹配原图)"""
    ax.clear()

    for obj_id in range(n_objects):
        st = instance_idx[obj_id]
        ed = instance_idx[obj_id + 1]
        obj_pos = positions[st:ed]
        color = get_color(obj_id)
        # Physion 坐标 X-left → 取反为 X-right 匹配原始图像
        # (-X, Y-up, Z) → matplotlib (-X, Z, Y)
        ax.scatter(-obj_pos[:, 0], obj_pos[:, 2], obj_pos[:, 1],
                   c=color, s=point_size, alpha=alpha, edgecolors='none')

    if x_range:
        ax.set_xlim(x_range)
    if z_range:
        ax.set_ylim(z_range)  # matplotlib Y = Physion Z
    if y_range:
        ax.set_zlim(y_range)  # matplotlib Z = Physion Y

    ax.set_xlabel('X (right)')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y (up)')
    ax.set_title(title)
    ax.view_init(elev=view_elev, azim=view_azim)


def compute_global_bounds(positions_list, margin=0.1):
    """计算所有帧的全局坐标边界（X 取反以匹配原始图像 X-right 视觉直觉）"""
    all_pos = np.concatenate(positions_list, axis=0)
    # Physion 空间是 X-left，显示时取反为 X-right
    x_neg = -all_pos[:, 0]
    x_range = (x_neg.min() - margin, x_neg.max() + margin)
    y_range = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    z_range = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)
    return x_range, y_range, z_range


# ============================================================
# 可视化 1: 粒子轨迹 GIF
# ============================================================
def create_trajectory_gif(positions_list, phases_dict, output_path,
                          fps=6, max_frames=40, figsize=(10, 8),
                          view_elev=25, view_azim=45):
    """
    生成粒子轨迹 GIF 动画

    Args:
        positions_list: list of (n_particles, 3), 每步的粒子位置
        phases_dict: Physion 格式的 phases_dict
        output_path: GIF 保存路径
        fps: 帧率
        max_frames: 最大帧数
        figsize: 图形尺寸
        view_elev: 视角仰角
        view_azim: 视角方位角
    """
    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']

    x_range, y_range, z_range = compute_global_bounds(positions_list)

    # 绘制地面
    ground_x = np.array([[x_range[0], x_range[1]], [x_range[0], x_range[1]]])
    ground_z = np.array([[z_range[0], z_range[0]], [z_range[1], z_range[1]]])
    ground_y = np.array([[0, 0], [0, 0]])

    # 采样帧
    n_steps = len(positions_list)
    if n_steps > max_frames:
        step_interval = n_steps / max_frames
        selected = [int(i * step_interval) for i in range(max_frames)]
        if (n_steps - 1) not in selected:
            selected.append(n_steps - 1)
    else:
        selected = list(range(n_steps))

    frames = []
    for step_idx in selected:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        render_particles_frame(
            ax, positions_list[step_idx], instance_idx, n_objects,
            view_elev=view_elev, view_azim=view_azim,
            x_range=x_range, y_range=y_range, z_range=z_range,
            title=f'Step {step_idx}/{n_steps-1}')

        # 地面
        ax.plot_surface(ground_x, ground_z, ground_y,
                        alpha=0.08, color='gray')

        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())
        plt.close(fig)

    _save_gif(frames, output_path, fps)
    print(f"  轨迹 GIF: {output_path} ({len(frames)} 帧)")


# ============================================================
# 可视化 2: 多视角快照
# ============================================================
def create_multiview_snapshot(positions_list, phases_dict, output_path,
                              steps=None, figsize=(20, 10)):
    """
    生成多视角快照 (初始、中间、最终)

    Args:
        positions_list: 粒子位置列表
        phases_dict: phases_dict
        output_path: PNG 保存路径
        steps: 要显示的步骤 [step0, step1, ...], 默认 [0, mid, end]
    """
    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']
    n_steps = len(positions_list)

    if steps is None:
        steps = [0, n_steps // 2, n_steps - 1]
    steps = [min(s, n_steps - 1) for s in steps]

    x_range, y_range, z_range = compute_global_bounds(positions_list)

    views = [
        (25, 45, 'View 1 (Front)'),
        (25, 135, 'View 2 (Side)'),
        (90, 0, 'View 3 (Top)'),
    ]

    n_rows = len(steps)
    n_cols = len(views)
    fig = plt.figure(figsize=figsize)

    for row, step_idx in enumerate(steps):
        for col, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1,
                                 projection='3d')
            title = f'Step {step_idx} - {view_name}'
            render_particles_frame(
                ax, positions_list[step_idx], instance_idx, n_objects,
                view_elev=elev, view_azim=azim,
                x_range=x_range, y_range=y_range, z_range=z_range,
                title=title, point_size=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  多视角快照: {output_path}")


# ============================================================
# 可视化 3: 质心轨迹曲线
# ============================================================
def plot_centroid_trajectories(centroids_array, phases_dict, output_path,
                                figsize=(14, 10)):
    """
    绘制每物体质心轨迹曲线

    Args:
        centroids_array: (n_steps, n_objects, 3) 质心位置
        phases_dict: phases_dict
        output_path: PNG 保存路径
    """
    n_steps, n_objects, _ = centroids_array.shape
    object_names = _get_object_names(phases_dict)
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # X 轨迹 (取反为 X-right)
    ax = axes[0, 0]
    for obj_id in range(n_objects):
        ax.plot(steps, -centroids_array[:, obj_id, 0],
                color=get_color(obj_id), label=object_names[obj_id], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('X (right)')
    ax.set_title('X Position (right+)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y 轨迹 (高度)
    ax = axes[0, 1]
    for obj_id in range(n_objects):
        ax.plot(steps, centroids_array[:, obj_id, 1],
                color=get_color(obj_id), label=object_names[obj_id], linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Ground')
    ax.set_xlabel('Step')
    ax.set_ylabel('Y (height)')
    ax.set_title('Y Position (Height)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z 轨迹
    ax = axes[1, 0]
    for obj_id in range(n_objects):
        ax.plot(steps, centroids_array[:, obj_id, 2],
                color=get_color(obj_id), label=object_names[obj_id], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Z')
    ax.set_title('Z Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 总位移
    ax = axes[1, 1]
    for obj_id in range(n_objects):
        displacement = np.linalg.norm(
            centroids_array[:, obj_id, :] - centroids_array[0, obj_id, :],
            axis=1)
        ax.plot(steps, displacement,
                color=get_color(obj_id), label=object_names[obj_id], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Displacement')
    ax.set_title('Total Displacement from Start')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Object Centroid Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  质心轨迹: {output_path}")


# ============================================================
# 可视化 4: 速度和加速度
# ============================================================
def plot_velocity_stats(centroids_array, phases_dict, output_path,
                        dt=0.03, figsize=(14, 5)):
    """
    绘制每物体速度变化

    Args:
        centroids_array: (n_steps, n_objects, 3) 质心位置
        phases_dict: phases_dict
        output_path: PNG 保存路径
        dt: 时间步长
    """
    n_steps, n_objects, _ = centroids_array.shape
    object_names = _get_object_names(phases_dict)

    # 计算速度 (差分)
    velocity = np.diff(centroids_array, axis=0) / dt  # (n_steps-1, n_objects, 3)
    speed = np.linalg.norm(velocity, axis=2)  # (n_steps-1, n_objects)
    steps = np.arange(1, n_steps)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 速率 (标量)
    ax = axes[0]
    for obj_id in range(n_objects):
        ax.plot(steps, speed[:, obj_id],
                color=get_color(obj_id), label=object_names[obj_id], linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Speed (units/s)')
    ax.set_title('Speed')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Y 方向速度 (重力方向)
    ax = axes[1]
    for obj_id in range(n_objects):
        ax.plot(steps, velocity[:, obj_id, 1],
                color=get_color(obj_id), label=object_names[obj_id], linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('V_y (units/s)')
    ax.set_title('Vertical Velocity (Y)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 最大粒子位移
    ax = axes[2]
    for obj_id in range(n_objects):
        disp = np.linalg.norm(
            centroids_array[1:, obj_id, :] - centroids_array[:-1, obj_id, :],
            axis=1)
        ax.plot(steps, disp,
                color=get_color(obj_id), label=object_names[obj_id], linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Step Displacement')
    ax.set_title('Per-step Displacement')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Velocity & Motion Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  速度分析: {output_path}")


# ============================================================
# 可视化 5: 初始 vs 最终对比
# ============================================================
def plot_initial_vs_final(positions_list, phases_dict, output_path,
                          figsize=(16, 6)):
    """并排对比初始和最终粒子状态"""
    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']
    object_names = _get_object_names(phases_dict)

    x_range, y_range, z_range = compute_global_bounds(positions_list)

    fig = plt.figure(figsize=figsize)

    for i, (step_idx, label) in enumerate([(0, 'Initial'), (-1, 'Final')]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        pos = positions_list[step_idx]

        for obj_id in range(n_objects):
            st = instance_idx[obj_id]
            ed = instance_idx[obj_id + 1]
            obj_pos = pos[st:ed]
            color = get_color(obj_id)
            # X 取反匹配原始图像
            ax.scatter(-obj_pos[:, 0], obj_pos[:, 2], obj_pos[:, 1],
                       c=color, s=4, alpha=0.6, label=object_names[obj_id],
                       edgecolors='none')

        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_zlim(y_range)
        ax.set_xlabel('X (right)')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (up)')
        step_num = step_idx if step_idx >= 0 else len(positions_list) + step_idx
        ax.set_title(f'{label} (Step {step_num})')
        ax.legend(fontsize=8, loc='upper right')
        ax.view_init(elev=25, azim=45)

        # 地面
        gx = np.array([[x_range[0], x_range[1]], [x_range[0], x_range[1]]])
        gz = np.array([[z_range[0], z_range[0]], [z_range[1], z_range[1]]])
        gy = np.array([[0, 0], [0, 0]])
        ax.plot_surface(gx, gz, gy, alpha=0.08, color='gray')

    plt.suptitle('Initial vs Final State', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  初始 vs 最终: {output_path}")


# ============================================================
# 可视化 6: 原始相机视角动画
# ============================================================
def render_camera_frame(ax, positions, instance_idx, n_objects,
                        camera, orig_image=None, object_names=None,
                        point_size=6, alpha=0.6, step_idx=0, n_steps=1,
                        show_ground=False):
    """
    渲染单帧: 将粒子投影到 2D 并叠加在原始图像上

    Args:
        ax: matplotlib axes
        positions: (n_particles, 3) 当前帧粒子位置 (Physion 空间)
        instance_idx: 物体粒子索引边界
        n_objects: 物体数
        camera: CameraInPhysion 实例
        orig_image: 原始图像 (H, W, 3), None 则白色背景
        object_names: 物体名称列表
        point_size: 粒子大小
        alpha: 粒子透明度
        step_idx: 当前步骤索引
        n_steps: 总步骤数
        show_ground: 是否渲染地面粒子
    """
    ax.clear()

    W, H = camera.image_width, camera.image_height

    # 背景
    if orig_image is not None:
        ax.imshow(orig_image)
    else:
        ax.set_facecolor('white')

    # 按物体渲染粒子
    legend_handles = []
    for obj_id in range(n_objects):
        name = object_names[obj_id] if object_names else f'obj_{obj_id}'
        # 跳过地面粒子
        if not show_ground and name.lower() in ('cube', 'ground'):
            continue

        st = instance_idx[obj_id]
        ed = instance_idx[obj_id + 1]
        obj_pts = positions[st:ed]

        # 投影到 2D
        uv, depth, visible = camera.project_with_visibility(obj_pts)
        if len(uv) == 0:
            continue

        color = get_color(obj_id)
        scatter = ax.scatter(uv[:, 0], uv[:, 1], c=color, s=point_size,
                             alpha=alpha, edgecolors='none', zorder=3)
        legend_handles.append((scatter, f'{name} ({len(uv)})'))

    # 帧标注
    ax.text(12, 28, f'Step {step_idx} / {n_steps - 1}',
            fontsize=11, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            zorder=5)

    # 图例
    if legend_handles:
        ax.legend([h[0] for h in legend_handles],
                  [h[1] for h in legend_handles],
                  loc='upper right', fontsize=7, framealpha=0.7,
                  markerscale=1.5)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.axis('off')


def create_camera_view_gif(positions_list, phases_dict, camera,
                           output_path, orig_image=None,
                           fps=8, max_frames=40, point_size=6,
                           alpha=0.6, show_ground=False):
    """
    生成原始相机视角的粒子动画 GIF

    Args:
        positions_list: list of (n_particles, 3), 每步粒子位置
        phases_dict: Physion 格式 phases_dict
        camera: CameraInPhysion 实例
        output_path: GIF 保存路径
        orig_image: 原始图像 (H, W, 3), None 则白色背景
        fps: 帧率
        max_frames: 最大帧数
        point_size: 粒子大小
        alpha: 粒子透明度
        show_ground: 是否渲染地面粒子
    """
    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']
    object_names = _get_object_names(phases_dict)

    W, H = camera.image_width, camera.image_height
    # 使用与原图匹配的 figsize (英寸) 和 dpi
    dpi = 100
    figsize = (W / dpi, H / dpi)

    # 采样帧
    n_steps = len(positions_list)
    if n_steps > max_frames:
        step_interval = n_steps / max_frames
        selected = [int(i * step_interval) for i in range(max_frames)]
        if (n_steps - 1) not in selected:
            selected.append(n_steps - 1)
    else:
        selected = list(range(n_steps))

    frames = []
    for i, step_idx in enumerate(selected):
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        render_camera_frame(
            ax, positions_list[step_idx], instance_idx, n_objects,
            camera, orig_image=orig_image, object_names=object_names,
            point_size=point_size, alpha=alpha,
            step_idx=step_idx, n_steps=n_steps,
            show_ground=show_ground)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"    渲染帧 {i + 1}/{len(selected)}...")

    _save_gif(frames, output_path, fps)
    print(f"  相机视角 GIF: {output_path} ({len(frames)} 帧, {W}x{H})")

    # 同时保存关键帧 PNG
    key_steps = [selected[0], selected[len(selected) // 2], selected[-1]]
    base_dir = os.path.dirname(output_path)
    for ks_idx, ks in enumerate(key_steps):
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        render_camera_frame(
            ax, positions_list[ks], instance_idx, n_objects,
            camera, orig_image=orig_image, object_names=object_names,
            point_size=point_size, alpha=alpha,
            step_idx=ks, n_steps=n_steps,
            show_ground=show_ground)

        key_path = os.path.join(base_dir, f'camera_view_step_{ks:03d}.png')
        fig.savefig(key_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  关键帧: {key_path}")


# ============================================================
# 辅助函数
# ============================================================
def _get_object_names(phases_dict):
    """从 phases_dict 获取物体名称"""
    names = []
    for name in phases_dict['instance']:
        if isinstance(name, bytes):
            name = name.decode()
        names.append(name)
    return names


def _save_gif(frames, output_path, fps=6):
    """保存帧列表为 GIF"""
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
    except ImportError:
        print("  警告: PIL 不可用, 跳过 GIF 生成")


def load_rollout_data(rollout_dir):
    """
    加载 rollout 结果数据

    Args:
        rollout_dir: run_rollout.py 的输出目录

    Returns:
        positions_list: list of (n_particles, 3)
        centroids_array: (n_steps, n_objects, 3)
        rollout_info: dict
    """
    # 加载粒子位置
    npz_path = os.path.join(rollout_dir, 'rollout_positions.npz')
    data = np.load(npz_path)
    n_steps = len(data.files)
    positions_list = [data[f'step_{i}'] for i in range(n_steps)]

    # 加载质心轨迹
    centroids_path = os.path.join(rollout_dir, 'centroids_trajectory.npy')
    centroids_array = np.load(centroids_path)

    # 加载元信息
    info_path = os.path.join(rollout_dir, 'rollout_info.json')
    with open(info_path, 'r') as f:
        rollout_info = json.load(f)

    return positions_list, centroids_array, rollout_info


def generate_all_visualizations(positions_list, centroids_array,
                                 phases_dict, output_dir,
                                 dt=0.03, fps=6,
                                 camera=None, orig_image=None):
    """
    生成所有可视化

    Args:
        positions_list: list of (n_particles, 3), 每步粒子位置
        centroids_array: (n_steps, n_objects, 3) 质心轨迹
        phases_dict: Physion 格式 phases_dict
        output_dir: 输出目录
        dt: 时间步长
        fps: GIF 帧率
        camera: CameraInPhysion 实例 (可选, 提供时生成相机视角动画)
        orig_image: 原始图像 (H, W, 3) (可选, 配合 camera 使用)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n生成可视化 → {output_dir}")

    # 1. 轨迹 GIF (有相机时自动匹配原图视角)
    traj_elev, traj_azim = 25, 45  # 默认视角
    if camera is not None:
        try:
            traj_elev, traj_azim = compute_camera_view_angles(camera)
            print(f"  使用相机匹配视角: elev={traj_elev:.1f}, azim={traj_azim:.1f}")
        except Exception as e:
            print(f"  计算相机视角失败, 使用默认: {e}")
    create_trajectory_gif(
        positions_list, phases_dict,
        os.path.join(output_dir, 'trajectory.gif'),
        fps=fps,
        view_elev=traj_elev, view_azim=traj_azim)

    # 2. 多视角快照
    create_multiview_snapshot(
        positions_list, phases_dict,
        os.path.join(output_dir, 'multiview_snapshot.png'))

    # 3. 质心轨迹
    plot_centroid_trajectories(
        centroids_array, phases_dict,
        os.path.join(output_dir, 'centroid_trajectories.png'))

    # 4. 速度分析
    if len(centroids_array) > 2:
        plot_velocity_stats(
            centroids_array, phases_dict,
            os.path.join(output_dir, 'velocity_analysis.png'),
            dt=dt)

    # 5. 初始 vs 最终
    plot_initial_vs_final(
        positions_list, phases_dict,
        os.path.join(output_dir, 'initial_vs_final.png'))

    # 6. 原始相机视角动画
    if camera is not None:
        print("  生成原始相机视角动画...")
        create_camera_view_gif(
            positions_list, phases_dict, camera,
            os.path.join(output_dir, 'camera_view.gif'),
            orig_image=orig_image,
            fps=fps)

    # 7. 保存摘要
    _save_summary(positions_list, centroids_array, phases_dict,
                  os.path.join(output_dir, 'visualization_summary.json'))

    print(f"  可视化完成: {len(os.listdir(output_dir))} 个文件")


def _save_summary(positions_list, centroids_array, phases_dict, output_path):
    """保存可视化摘要 JSON"""
    n_steps = len(positions_list)
    n_objects = phases_dict['n_objects']
    object_names = _get_object_names(phases_dict)

    summary = {
        'n_steps': n_steps,
        'n_particles': int(positions_list[0].shape[0]),
        'n_objects': n_objects,
        'objects': [],
    }

    for obj_id in range(n_objects):
        traj = centroids_array[:, obj_id, :]
        displacement = float(np.linalg.norm(traj[-1] - traj[0]))
        max_speed = 0.0
        if n_steps > 1:
            vel = np.diff(traj, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            max_speed = float(speed.max())

        summary['objects'].append({
            'name': object_names[obj_id],
            'start_position': traj[0].tolist(),
            'end_position': traj[-1].tolist(),
            'total_displacement': displacement,
            'max_step_speed': max_speed,
        })

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Rollout 结果可视化')
    parser.add_argument('--rollout-dir', type=str, required=True,
                        help='run_rollout.py 输出的 rollout 目录')
    parser.add_argument('--trial-path', type=str, default=None,
                        help='Trial 目录 (用于获取 phases_dict)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='可视化输出目录 (默认: rollout_dir/vis)')
    parser.add_argument('--fps', type=int, default=6,
                        help='GIF 帧率')
    parser.add_argument('--dt', type=float, default=0.03,
                        help='时间步长')

    # 相机视角参数
    cam_group = parser.add_argument_group('Camera View (原始相机视角动画)')
    cam_group.add_argument('--camera-params', type=str, default=None,
                           help='camera_params.json 路径')
    cam_group.add_argument('--coord-transform', type=str, default=None,
                           help='coord_transform.json 路径')
    cam_group.add_argument('--image', type=str, default=None,
                           help='原始图像路径 (作为动画背景)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.rollout_dir, 'vis')

    # 加载 rollout 数据
    print("加载 rollout 数据...")
    positions_list, centroids_array, rollout_info = load_rollout_data(args.rollout_dir)
    print(f"  {len(positions_list)} 步, {positions_list[0].shape[0]} 粒子")

    # 加载 phases_dict
    if args.trial_path is None:
        args.trial_path = rollout_info.get('phases_dict_path', args.rollout_dir)
        # 也可能 trial 就是 rollout 的上层目录
        if not os.path.exists(os.path.join(args.trial_path, 'phases_dict.pkl')):
            args.trial_path = os.path.dirname(args.rollout_dir)

    phases_dict_path = os.path.join(args.trial_path, 'phases_dict.pkl')
    with open(phases_dict_path, 'rb') as f:
        phases_dict = pickle.load(f)
    print(f"  加载 phases_dict: {phases_dict['n_objects']} 物体")

    # 加载相机参数 (可选)
    camera = None
    orig_image = None

    if args.camera_params:
        from camera_restore import CameraInPhysion
        coord_transform_path = args.coord_transform
        if coord_transform_path is None:
            # 自动查找 trial 目录下的 coord_transform.json
            coord_transform_path = os.path.join(args.trial_path, 'coord_transform.json')
        if os.path.exists(coord_transform_path):
            camera = CameraInPhysion.from_files(args.camera_params, coord_transform_path)
            print(f"  加载相机: {camera}")
        else:
            print(f"  警告: 未找到 coord_transform.json, 跳过相机视角动画")

    if args.image and os.path.exists(args.image):
        from PIL import Image as PILImage
        orig_image = np.array(PILImage.open(args.image).convert("RGB"))
        print(f"  加载原始图像: {args.image} ({orig_image.shape[1]}x{orig_image.shape[0]})")

    # 生成可视化
    generate_all_visualizations(
        positions_list, centroids_array, phases_dict,
        args.output_dir, dt=args.dt, fps=args.fps,
        camera=camera, orig_image=orig_image)


if __name__ == '__main__':
    main()
