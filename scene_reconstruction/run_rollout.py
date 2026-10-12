"""
run_rollout.py - 使用训练好的 GNS 模型对自定义场景进行 rollout 推理

功能:
- 加载 build_physion_input.py 生成的 trial 数据
- 加载训练好的 GNS 模型
- 执行自回归 rollout 预测
- 保存粒子轨迹和可视化
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch

# 添加 Physion-particles 到 path
# GNS code directory (models/GNS_pyg.py, inference_pyg.py). Defaults to the sibling
# release module ../gns_training; override with the PHYSION_PARTICLES_DIR env variable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHYSION_DIR = os.environ.get(
    'PHYSION_PARTICLES_DIR',
    os.path.join(os.path.dirname(_THIS_DIR), 'gns_training'))
sys.path.insert(0, PHYSION_DIR)

# Default GNS checkpoint: <release root>/checkpoints/gns/without_Drape/best_model.pth
DEFAULT_GNS_CKPT = os.path.join(
    os.path.dirname(_THIS_DIR), 'checkpoints', 'gns', 'without_Drape', 'best_model.pth')

# numpy compatibility
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from models.GNS_pyg import GNS_PyG, create_gns_model
from inference_pyg import (
    load_phases_dict, load_positions, load_stat,
    build_pyg_data, apply_rigid_projection,
    normalize, denormalize
)


def load_model(checkpoint_path, device, nf_particle=200, nf_relation=300,
               nf_effect=200, pstep=10):
    """加载训练好的 GNS 模型"""
    model = create_gns_model(
        nf_particle=nf_particle,
        nf_relation=nf_relation,
        nf_effect=nf_effect,
        pstep=pstep,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        loss = checkpoint.get('valid_loss', checkpoint.get('loss', '?'))
        print(f"  加载 checkpoint: epoch={epoch}, valid_loss={loss}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  加载 state_dict")

    model = model.to(device)
    model.eval()
    return model


def inference_step(model, pos_t, pos_prev, phases_dict, stat, dt, device,
                   neighbor_radius=0.08, floor_cheat=True, ransac_on_pred=False):
    """单步推理"""
    data = build_pyg_data(pos_t, pos_prev, phases_dict, stat, dt,
                          neighbor_radius, floor_cheat)
    data = data.to(device)

    with torch.no_grad():
        pred = model(data)

    # 提取粒子预测 (不含 floor)
    n_particles = data.n_particles
    pred_vel_norm = pred[:n_particles].cpu().numpy()

    # 反归一化
    pred_vel = denormalize(pred_vel_norm, stat[1])

    # 预测位置
    pos_pred = pos_t + pred_vel * dt

    if ransac_on_pred:
        pos_pred = apply_rigid_projection(pos_t, pos_pred, phases_dict)

    return pos_pred


def run_rollout(model, trial_path, rollout_steps=50, training_fpt=3,
                device='cuda', neighbor_radius=0.08, ransac_on_pred=True,
                phases_dict=None, stat=None):
    """
    对自定义 trial 执行 rollout 推理

    Args:
        model: GNS 模型
        trial_path: trial 目录 (包含 phases_dict.pkl, .h5, stat.h5)
        rollout_steps: 预测步数
        training_fpt: 帧间隔
        device: 计算设备
        neighbor_radius: 邻居搜索半径
        ransac_on_pred: 是否做刚体投影
        phases_dict: 可选，预加载的 phases_dict（避免重复加载）
        stat: 可选，预加载的归一化统计量

    Returns:
        positions: list of (n_particles, 3), 长度 rollout_steps+1 (包含初始帧)
        centroids_per_step: list of (n_objects, 3), 每步每物体质心
    """
    if phases_dict is None:
        phases_dict = load_phases_dict(trial_path)
    if stat is None:
        stat = load_stat(trial_path)

    dt = training_fpt * 0.01
    n_particles = phases_dict['n_particles']
    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']

    print(f"  Trial: {n_objects} 物体, {n_particles} 粒子")
    print(f"  dt={dt}, fpt={training_fpt}, rollout_steps={rollout_steps}")

    # 加载初始两帧
    t_prev = 0
    t_curr = training_fpt

    pos_prev = load_positions(trial_path, t_prev, phases_dict)
    pos_t = load_positions(trial_path, t_curr, phases_dict)

    vel_init = (pos_t - pos_prev) / dt
    print(f"  初始速度范数: {np.linalg.norm(vel_init, axis=1).mean():.4f} (均值)")

    # 存储结果
    positions = [pos_t.copy()]  # 包含初始帧
    centroids_per_step = []

    # 计算初始质心
    step_centroids = []
    for obj_id in range(n_objects):
        st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
        step_centroids.append(pos_t[st:ed].mean(axis=0))
    centroids_per_step.append(np.array(step_centroids))

    print(f"\n  Rollout 推理中...")
    for step in range(rollout_steps):
        pos_pred = inference_step(
            model, pos_t, pos_prev, phases_dict, stat, dt, device,
            neighbor_radius, floor_cheat=True, ransac_on_pred=ransac_on_pred)

        positions.append(pos_pred.copy())

        # 计算质心
        step_centroids = []
        for obj_id in range(n_objects):
            st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
            step_centroids.append(pos_pred[st:ed].mean(axis=0))
        centroids_per_step.append(np.array(step_centroids))

        # 自回归更新
        pos_prev = pos_t
        pos_t = pos_pred

        if (step + 1) % 10 == 0 or step == 0:
            # 打印进度
            max_displacement = np.max(np.linalg.norm(pos_pred - positions[0], axis=1))
            print(f"    Step {step+1:3d}/{rollout_steps}: "
                  f"max_displacement={max_displacement:.4f}")

    return positions, centroids_per_step


def save_rollout_results(output_dir, positions, centroids_per_step,
                         phases_dict_path, extra_info=None):
    """保存 rollout 结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存粒子位置 (numpy)
    np.savez_compressed(
        os.path.join(output_dir, 'rollout_positions.npz'),
        **{f'step_{i}': pos for i, pos in enumerate(positions)}
    )

    # 保存质心轨迹
    centroids_array = np.array(centroids_per_step)  # (steps, n_objects, 3)
    np.save(os.path.join(output_dir, 'centroids_trajectory.npy'), centroids_array)

    # 保存元信息
    info = {
        'n_steps': len(positions),
        'n_particles': int(positions[0].shape[0]),
        'phases_dict_path': phases_dict_path,
    }
    if extra_info:
        info.update(extra_info)
    with open(os.path.join(output_dir, 'rollout_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  结果保存到: {output_dir}")


def visualize_rollout(positions, centroids_per_step, phases_dict,
                      output_path='rollout.gif', fps=4):
    """生成 rollout 可视化 GIF"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  警告: matplotlib 不可用, 跳过可视化")
        return

    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']

    # 计算全局边界
    all_pos = np.concatenate(positions, axis=0)
    margin = 0.2
    x_range = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    y_range = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    z_range = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown']
    frames = []

    # 每隔几帧采样 (避免 GIF 太大)
    step_interval = max(1, len(positions) // 30)
    selected_steps = list(range(0, len(positions), step_interval))
    if (len(positions) - 1) not in selected_steps:
        selected_steps.append(len(positions) - 1)

    for step_idx in selected_steps:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        pos = positions[step_idx]
        for obj_id in range(n_objects):
            st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
            obj_pos = pos[st:ed]
            color = colors[obj_id % len(colors)]
            ax.scatter(obj_pos[:, 0], obj_pos[:, 2], obj_pos[:, 1],
                       c=color, s=3, alpha=0.6)

        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_zlim(y_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f'Step {step_idx}')

        # 绘制地面
        ax.plot_surface(
            np.array([[x_range[0], x_range[1]], [x_range[0], x_range[1]]]),
            np.array([[z_range[0], z_range[0]], [z_range[1], z_range[1]]]),
            np.array([[0, 0], [0, 0]]),
            alpha=0.1, color='gray'
        )

        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())
        plt.close(fig)

    # 保存 GIF
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
        print(f"  GIF 保存到: {output_path}")
    except ImportError:
        print("  警告: PIL 不可用, 跳过 GIF 生成")
        # 保存关键帧为 PNG
        for i, frame in enumerate(frames[::max(1, len(frames) // 5)]):
            from matplotlib.image import imsave
            imsave(output_path.replace('.gif', f'_frame{i}.png'), frame)


def visualize_rollout_topdown(positions, phases_dict,
                              output_path='rollout_topdown.gif', fps=4):
    """生成俯视图 (X-Z) GIF，便于检查左右关系"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  警告: matplotlib 不可用, 跳过俯视图可视化")
        return

    instance_idx = phases_dict['instance_idx']
    n_objects = phases_dict['n_objects']
    object_names = phases_dict.get('instance', [f'obj_{i}' for i in range(n_objects)])

    all_pos = np.concatenate(positions, axis=0)
    margin = 0.2
    x_range = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    z_range = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown']
    frames = []

    step_interval = max(1, len(positions) // 30)
    selected_steps = list(range(0, len(positions), step_interval))
    if (len(positions) - 1) not in selected_steps:
        selected_steps.append(len(positions) - 1)

    for step_idx in selected_steps:
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = positions[step_idx]

        for obj_id in range(n_objects):
            st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
            obj_pos = pos[st:ed]
            color = colors[obj_id % len(colors)]
            name = object_names[obj_id]
            if isinstance(name, bytes):
                name = name.decode()

            ax.scatter(obj_pos[:, 0], obj_pos[:, 2], c=color, s=3, alpha=0.55)
            centroid = obj_pos.mean(axis=0)
            ax.scatter(centroid[0], centroid[2], c=color, s=70, marker='*',
                       edgecolors='black', linewidths=0.5)
            ax.text(centroid[0], centroid[2], f'{obj_id}:{name}',
                    fontsize=8, color=color, ha='center', va='bottom')

        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.grid(True, alpha=0.2)
        ax.set_title(f'Top-down X-Z | Step {step_idx}')
        ax.text(0.01, 0.98, 'left/right = world X axis',
                transform=ax.transAxes, fontsize=8,
                va='top', ha='left', color='dimgray')

        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())
        plt.close(fig)

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
        print(f"  俯视图 GIF 保存到: {output_path}")
    except ImportError:
        print("  警告: PIL 不可用, 跳过俯视图 GIF 生成")
        for i, frame in enumerate(frames[::max(1, len(frames) // 5)]):
            from matplotlib.image import imsave
            imsave(output_path.replace('.gif', f'_frame{i}.png'), frame)


def main():
    parser = argparse.ArgumentParser(description='GNS Rollout 推理')
    parser.add_argument('--trial-path', type=str, required=True,
                        help='Trial 数据目录')
    parser.add_argument('--gns-ckpt', '--model-path', dest='model_path', type=str,
                        default=DEFAULT_GNS_CKPT,
                        help='训练好的模型路径 (default: '
                             'checkpoints/gns/without_Drape/best_model.pth under the release root)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: trial_path/rollout)')
    parser.add_argument('--rollout-steps', type=int, default=50,
                        help='Rollout 步数')
    parser.add_argument('--training-fpt', type=int, default=3,
                        help='帧间隔')
    parser.add_argument('--neighbor-radius', type=float, default=0.08,
                        help='邻居搜索半径')
    parser.add_argument('--ransac', action='store_true', default=True,
                        help='对 rigid 物体做刚体投影')
    parser.add_argument('--no-ransac', dest='ransac', action='store_false')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化 GIF')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')

    # 模型参数
    parser.add_argument('--nf-particle', type=int, default=200)
    parser.add_argument('--nf-relation', type=int, default=300)
    parser.add_argument('--nf-effect', type=int, default=200)
    parser.add_argument('--pstep', type=int, default=10)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.trial_path, 'rollout')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    phases_dict = load_phases_dict(args.trial_path)
    stat = load_stat(args.trial_path)

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = load_model(
        args.model_path, device,
        nf_particle=args.nf_particle,
        nf_relation=args.nf_relation,
        nf_effect=args.nf_effect,
        pstep=args.pstep,
    )

    # 运行 rollout
    print(f"\n运行 Rollout:")
    positions, centroids = run_rollout(
        model=model,
        trial_path=args.trial_path,
        rollout_steps=args.rollout_steps,
        training_fpt=args.training_fpt,
        device=device,
        neighbor_radius=args.neighbor_radius,
        ransac_on_pred=args.ransac,
        phases_dict=phases_dict,
        stat=stat,
    )

    # 保存结果
    print(f"\n保存结果:")
    save_rollout_results(
        output_dir=args.output_dir,
        positions=positions,
        centroids_per_step=centroids,
        phases_dict_path=args.trial_path,
        extra_info={
            'model_path': args.model_path,
            'rollout_steps': args.rollout_steps,
            'training_fpt': args.training_fpt,
            'ransac': args.ransac,
        }
    )

    # 可视化
    if args.visualize:
        print(f"\n生成可视化:")
        visualize_rollout(
            positions, centroids, phases_dict,
            output_path=os.path.join(args.output_dir, 'rollout.gif'),
        )
        visualize_rollout_topdown(
            positions, phases_dict,
            output_path=os.path.join(args.output_dir, 'rollout_topdown.gif'),
        )

    # 打印轨迹摘要
    print(f"\n轨迹摘要:")
    centroids_array = np.array(centroids)
    for obj_id in range(phases_dict['n_objects']):
        obj_name = phases_dict['instance'][obj_id]
        if isinstance(obj_name, bytes):
            obj_name = obj_name.decode()
        traj = centroids_array[:, obj_id, :]
        displacement = np.linalg.norm(traj[-1] - traj[0])
        print(f"  物体 {obj_id} ({obj_name}): "
              f"起始={traj[0]}, 终止={traj[-1]}, "
              f"位移={displacement:.4f}")


if __name__ == '__main__':
    main()
