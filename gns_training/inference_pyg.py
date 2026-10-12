"""
inference_pyg.py - GNS-PyG 推理脚本

功能:
- 加载训练好的 GNS-PyG checkpoint
- 从指定数据集中选取 N 个 trial
- 对每个 trial 进行 autoregressive rollout (多步推理)
- 计算预测 vs 真实的误差 (MSE/RMSE)
- 可选: 保存 3D 可视化和误差曲线
"""

import os
import argparse
import json
import pickle
import copy
import numpy as np
import torch
import h5py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from models.GNS_pyg import GNS_PyG, create_gns_model
from torch_geometric.data import Data

# numpy>=1.24 compatibility for vendor utils using np.int.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from utils_geom import calc_rigid_transform


# ========== 数据处理工具函数 (从 data_pyg.py 提取) ==========

def load_phases_dict(trial_path, subsample_limit=3000):
    """加载并处理 phases_dict"""
    pkl_path = os.path.join(trial_path, 'phases_dict.pkl')
    with open(pkl_path, 'rb') as f:
        phases_dict = pickle.load(f)
    phases_dict['trial_dir'] = trial_path
    _correct_bad_chair(phases_dict)
    _remove_large_obstacles(phases_dict)
    _subsample_particles(phases_dict, subsample_limit)
    return phases_dict


def load_positions(trial_path, timestep, phases_dict):
    """加载指定时间步的粒子位置"""
    h5_path = os.path.join(trial_path, f'{timestep}.h5')
    is_drape = 'Drape' in trial_path

    with h5py.File(h5_path, 'r') as f:
        if is_drape and 'particle_positions' in f.keys():
            positions = np.array(f['particle_positions'])
            positions = positions * 0.05 / 0.035
            if phases_dict.get('is_subsample', False):
                obj_subsample_idx = phases_dict['obj_subsample_idx']
                instance_idx_old = phases_dict['instance_idx_before_subsample']
                n_objects = phases_dict['n_objects']
                new_positions = []
                for obj_id in range(n_objects):
                    st = instance_idx_old[obj_id]
                    ed = instance_idx_old[obj_id + 1]
                    obj_pos = positions[st:ed]
                    new_positions.append(obj_pos[obj_subsample_idx[obj_id]])
                positions = np.concatenate(new_positions, axis=0)
        else:
            obj_positions = np.array(f['obj_positions'])
            obj_rotations = np.array(f['obj_rotations'])
            obj_points = phases_dict['obj_points']
            ok_obj_id = phases_dict.get('ok_obj_id', range(len(obj_points)))
            all_positions = []
            for idx, obj_id in enumerate(ok_obj_id):
                rot = R.from_quat(obj_rotations[obj_id]).as_matrix()
                trans = obj_positions[obj_id]
                pts = obj_points[idx]
                transformed_pts = np.dot(rot, pts.T).T + trans
                all_positions.append(transformed_pts)
            positions = np.concatenate(all_positions, axis=0)

    return positions.astype(np.float32)


def normalize(data, stat):
    mean = stat[:, 0]
    std = stat[:, 1]
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std


def denormalize(data, stat):
    mean = stat[:, 0]
    std = stat[:, 1]
    std = np.where(std == 0, 1.0, std)
    return data * std + mean


def build_edges(pos_raw, pos_norm, n_particles, neighbor_radius=0.08):
    """用原始坐标构建邻居边，edge_attr 用归一化坐标"""
    pos = pos_raw[:n_particles, :3]
    tree = cKDTree(pos)
    neighbors_list = tree.query_ball_point(pos, neighbor_radius, p=2)

    senders, receivers = [], []
    for i, neighbors in enumerate(neighbors_list):
        for j in neighbors:
            senders.append(j)
            receivers.append(i)

    if len(senders) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    edge_index = np.array([senders, receivers], dtype=np.int64)

    pos_r = pos_norm[receivers, :3]
    pos_s = pos_norm[senders, :3]
    relative_pos = pos_r - pos_s
    distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)
    edge_attr = np.concatenate([relative_pos, distance], axis=1).astype(np.float32)

    return edge_index, edge_attr


def build_floor_edges(pos_raw, pos_norm, n_particles):
    """floor 判断用原始坐标，edge_attr 用归一化坐标"""
    floor_idx = pos_raw.shape[0] - 1
    y_raw = pos_raw[:n_particles, 1]
    near_floor = np.where(y_raw < 0.1)[0]

    if len(near_floor) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    senders = np.full(len(near_floor), floor_idx, dtype=np.int64)
    receivers = near_floor.astype(np.int64)
    edge_index = np.stack([senders, receivers], axis=0)

    pos_r = pos_norm[receivers, :3]
    pos_s = pos_norm[senders, :3]
    relative_pos = pos_r - pos_s
    distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)
    edge_attr = np.concatenate([relative_pos, distance], axis=1).astype(np.float32)

    return edge_index, edge_attr


def build_object_id(phases_dict, n_particles, floor_cheat=True):
    n_nodes = n_particles + (1 if floor_cheat else 0)
    object_id = np.zeros(n_nodes, dtype=np.int64)
    is_rigid = np.zeros(n_nodes, dtype=bool)

    instance_idx = phases_dict['instance_idx']
    materials = phases_dict['material']

    for obj_id in range(len(instance_idx) - 1):
        st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
        object_id[st:ed] = obj_id
        if materials[obj_id] == 'rigid':
            is_rigid[st:ed] = True

    if floor_cheat:
        object_id[-1] = -1

    return object_id, is_rigid


def build_node_attr(phases_dict, n_particles, floor_cheat=True):
    n_nodes = n_particles + (1 if floor_cheat else 0)
    attr = np.zeros((n_nodes, 3), dtype=np.float32)
    # dim1: 软体材质位 (cloth/fluid), 与 data_pyg.py::_build_node_attr 一致
    instance_idx = phases_dict['instance_idx']
    materials = phases_dict['material']
    for obj_id in range(len(instance_idx) - 1):
        if materials[obj_id] != 'rigid':
            st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
            attr[st:ed, 1] = 1
    if floor_cheat:
        attr[-1, 2] = 1
    return attr


def load_stat(trial_path):
    """从 trial 目录加载统计量"""
    stat_path = os.path.join(trial_path, 'stat.h5')
    if os.path.exists(stat_path):
        with h5py.File(stat_path, 'r') as f:
            positions_stat = np.array(f['positions'])
            velocities_stat = np.array(f['velocities'])
        return [positions_stat, velocities_stat]
    return [np.zeros((3, 3)), np.zeros((3, 3))]


# ========== data_pyg.py 的辅助方法 ==========

def _correct_bad_chair(phases_dict):
    if len(phases_dict["instance_idx"]) - 1 != phases_dict["n_objects"]:
        obj_points = []
        opt_ids = []
        for opt_id, opts in enumerate(phases_dict["obj_points"]):
            if opts.shape[0] != 0:
                obj_points.append(opts)
                opt_ids.append(opt_id)
        phases_dict["obj_points"] = obj_points
        phases_dict["before_fix_n_objects"] = phases_dict["n_objects"]
        phases_dict["n_objects"] = len(obj_points)
        phases_dict["ok_obj_id"] = opt_ids


def _remove_large_obstacles(phases_dict):
    if phases_dict["n_particles"] <= 3000:
        return
    critical_objects = [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube',
                       b'cylinder', b'dumbbell', b'octahedron', b'pentagon',
                       b'pipe', b'platonic', b'pyramid', b'sphere', b'torus',
                       b'triangular_prism']
    n_objects = phases_dict["n_objects"]
    objs_npts = []
    ok_obj_id = []
    is_removed = False
    for obj_id in range(n_objects):
        obj_npts = phases_dict["instance_idx"][obj_id + 1] - phases_dict["instance_idx"][obj_id]
        if phases_dict["instance"][obj_id] not in critical_objects and obj_npts > 3000:
            is_removed = True
            continue
        objs_npts.append(obj_npts)
        ok_obj_id.append(obj_id)
    if is_removed:
        current_particle_id = 0
        new_instance_idx = [current_particle_id]
        for obj_npts in objs_npts:
            current_particle_id += obj_npts
            new_instance_idx.append(current_particle_id)
        list_items = ["root_des_radius", "root_num", "clusters", "instance", "material", "obj_points"]
        for item in list_items:
            if item in phases_dict:
                phases_dict[item] = [phases_dict[item][a] for a in ok_obj_id]
        phases_dict["n_objects"] = len(ok_obj_id)
        phases_dict["n_particles"] = sum(objs_npts)
        phases_dict["instance_idx"] = new_instance_idx
        phases_dict["ok_obj_id"] = ok_obj_id


def _subsample_particles(phases_dict, limit=3000):
    n_objects = phases_dict["n_objects"]
    obj_names = phases_dict.get("instance", [])
    obj_subsample_idx = []
    obj_points = phases_dict["obj_points"]
    new_instance_idx = [0]
    is_subsample = False
    for obj_id in range(n_objects):
        obj_npts = phases_dict["instance_idx"][obj_id + 1] - phases_dict["instance_idx"][obj_id]
        obj_name = obj_names[obj_id] if obj_id < len(obj_names) else b''
        if obj_npts > limit and obj_name != b'cloth_square':
            selected_idxs = np.random.choice(obj_npts, limit, replace=False)
            obj_points[obj_id] = obj_points[obj_id][selected_idxs]
            obj_subsample_idx.append(selected_idxs)
            is_subsample = True
            if "clusters" in phases_dict and obj_id < len(phases_dict["clusters"]):
                phases_dict["clusters"][obj_id][0][0] = \
                    phases_dict["clusters"][obj_id][0][0][selected_idxs]
        else:
            obj_subsample_idx.append(np.arange(obj_npts))
        current_particle_idx = new_instance_idx[-1] + obj_points[obj_id].shape[0]
        new_instance_idx.append(current_particle_idx)
    phases_dict["n_particles"] = new_instance_idx[-1]
    phases_dict["obj_points"] = obj_points
    phases_dict["instance_idx_before_subsample"] = copy.deepcopy(phases_dict["instance_idx"])
    phases_dict["instance_idx"] = new_instance_idx
    phases_dict["obj_subsample_idx"] = obj_subsample_idx
    phases_dict["is_subsample"] = is_subsample


# ========== 推理核心逻辑 ==========

def build_pyg_data(pos_t, pos_prev, phases_dict, stat, dt,
                   neighbor_radius=0.08, floor_cheat=True):
    """
    从位置构建 PyG Data 对象 (推理用, 不加噪声)

    Args:
        pos_t: (n_particles, 3) 当前帧位置
        pos_prev: (n_particles, 3) 前一帧位置
        phases_dict: 场景信息
        stat: [positions_stat, velocities_stat]
        dt: 时间步长
        neighbor_radius: 邻居搜索半径
        floor_cheat: 是否使用 floor 节点

    Returns:
        data: PyG Data 对象
    """
    vel_t = (pos_t - pos_prev) / dt
    n_particles = pos_t.shape[0]

    # 保存原始坐标 (用于构边)
    pos_t_raw = pos_t.copy()

    # 添加 floor 节点
    if floor_cheat:
        pos_t_raw = np.vstack([pos_t_raw, np.zeros((1, 3))])
        pos_t_with_floor = np.vstack([pos_t, np.zeros((1, 3))])
        vel_t = np.vstack([vel_t, np.zeros((1, 3))])
    else:
        pos_t_with_floor = pos_t

    # 归一化
    pos_t_norm = normalize(pos_t_with_floor, stat[0])
    vel_t_norm = normalize(vel_t, stat[1])

    # 构建 object_id 和 is_rigid
    object_id, is_rigid = build_object_id(phases_dict, n_particles, floor_cheat)

    # 构建节点属性
    attr = build_node_attr(phases_dict, n_particles, floor_cheat)

    # 构建边
    edge_index, edge_attr = build_edges(pos_t_raw, pos_t_norm, n_particles, neighbor_radius)

    # floor 边
    if floor_cheat:
        floor_edge_index, floor_edge_attr = build_floor_edges(pos_t_raw, pos_t_norm, n_particles)
        if floor_edge_index.shape[1] > 0:
            edge_index = np.hstack([edge_index, floor_edge_index])
            edge_attr = np.vstack([edge_attr, floor_edge_attr])

    # 节点特征: attr + vel_masked
    n_nodes = pos_t_norm.shape[0]
    vel_masked = np.zeros((n_nodes, 6), dtype=np.float32)
    vel_masked[:, 3:6] = vel_t_norm

    node_features = np.concatenate([attr, vel_masked], axis=1)  # (N, 9)

    # state
    state = np.concatenate([pos_t_norm, vel_t_norm], axis=1)  # (N, 6)

    data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        n_particles=n_particles,
        state=torch.FloatTensor(state),
        pos_raw=torch.FloatTensor(pos_t_raw),
        object_id=torch.LongTensor(object_id),
        is_rigid=torch.BoolTensor(is_rigid),
        batch=torch.zeros(n_nodes, dtype=torch.long),  # 单图推理, batch 全为 0
    )

    return data


def apply_rigid_projection(pos_prev, pos_pred, phases_dict):
    """
    对 rigid 物体做刚体投影 (对齐 eval.py --ransac_on_pred 逻辑)。
    """
    projected = pos_pred.copy()
    instance_idx = phases_dict['instance_idx']
    materials = phases_dict['material']

    for obj_id in range(len(instance_idx) - 1):
        st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
        if materials[obj_id] != 'rigid':
            continue

        prev_obj = pos_prev[st:ed]
        pred_obj = pos_pred[st:ed]
        R, T = calc_rigid_transform(prev_obj, pred_obj)
        projected[st:ed] = (np.dot(R, prev_obj.T) + T).T

    return projected


def apply_translation_projection(pos_prev, pos_pred, phases_dict):
    """
    仅平移投影：所有粒子以质心位移统一平移，不旋转。
    保持物体形状不变，避免 SVD 拟合出的大角度旋转导致翻转。
    """
    projected = pos_pred.copy()
    instance_idx = phases_dict['instance_idx']
    materials = phases_dict['material']

    for obj_id in range(len(instance_idx) - 1):
        st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
        if materials[obj_id] != 'rigid':
            continue

        prev_obj = pos_prev[st:ed]
        pred_obj = pos_pred[st:ed]

        centroid_prev = prev_obj.mean(axis=0)
        centroid_pred = pred_obj.mean(axis=0)
        translation = centroid_pred - centroid_prev

        projected[st:ed] = prev_obj + translation

    return projected


def build_cloth_constraints(pos_start, phases_dict, radius_scale=1.5, max_rest_scale=2.0):
    """
    对 cloth 物体在起始帧构建保距约束对 (PBD 距离约束)。

    起始帧布料尚整齐, 空间邻近 ≈ 网格拓扑邻近: 以 radius_scale × median(最近邻距)
    为半径取粒子对, 记录 rest 距离; 起始已拉伸的异常对 (> max_rest_scale × dnn) 剔除。
    返回 None 表示场景无 cloth 物体。
    """
    instance_idx = phases_dict['instance_idx']
    materials = phases_dict['material']

    idx_i_list, idx_j_list, rest_list = [], [], []
    for obj_id in range(len(instance_idx) - 1):
        if materials[obj_id] != 'cloth':
            continue
        st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
        cloth_pos = pos_start[st:ed]
        if cloth_pos.shape[0] < 4:
            continue
        tree = cKDTree(cloth_pos)
        nn_dist, _ = tree.query(cloth_pos, k=2)
        dnn = np.median(nn_dist[:, 1])
        pairs = np.array(list(tree.query_pairs(radius_scale * dnn)))
        if pairs.size == 0:
            continue
        rest = np.linalg.norm(cloth_pos[pairs[:, 0]] - cloth_pos[pairs[:, 1]], axis=1)
        keep = rest <= max_rest_scale * dnn
        pairs, rest = pairs[keep], rest[keep]
        idx_i_list.append(pairs[:, 0] + st)
        idx_j_list.append(pairs[:, 1] + st)
        rest_list.append(rest)

    if not idx_i_list:
        return None

    idx_i = np.concatenate(idx_i_list)
    idx_j = np.concatenate(idx_j_list)
    rest_len = np.concatenate(rest_list)
    # 每个粒子参与的约束数, 用于 Jacobi 平均化 (未参与约束的粒子置 1 防除零)
    count = np.zeros(pos_start.shape[0])
    np.add.at(count, idx_i, 1.0)
    np.add.at(count, idx_j, 1.0)
    count[count == 0] = 1.0
    return {'idx_i': idx_i, 'idx_j': idx_j, 'rest_len': rest_len, 'count': count}


def apply_cloth_distance_projection(pos_pred, constraints, n_iters=5, stiffness=0.8):
    """
    Jacobi 式 PBD 距离投影: 把拉伸/压缩的布料约束对拉回自然间距。
    约束索引只覆盖 cloth 粒子, 其余粒子不动。只防拉散/压缩, 不阻止弯折。
    """
    if constraints is None:
        return pos_pred
    pos = pos_pred.copy()
    idx_i = constraints['idx_i']
    idx_j = constraints['idx_j']
    rest = constraints['rest_len']
    count = constraints['count']

    for _ in range(n_iters):
        diff = pos[idx_i] - pos[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-9)
        corr = (dist - rest) / dist
        delta = 0.5 * stiffness * corr[:, None] * diff
        disp = np.zeros_like(pos)
        np.add.at(disp, idx_i, -delta)
        np.add.at(disp, idx_j, delta)
        pos += disp / count[:, None]
    return pos


def inference_step(model, pos_t, pos_prev, phases_dict, stat, dt, device,
                   neighbor_radius=0.08, floor_cheat=True, ransac_on_pred=False,
                   cloth_constraints=None, cloth_proj_iters=5, cloth_proj_stiffness=0.8):
    """
    单步推理: 预测下一帧位置

    Returns:
        pos_pred: (n_particles, 3) 预测的下一帧位置
    """
    data = build_pyg_data(pos_t, pos_prev, phases_dict, stat, dt,
                          neighbor_radius, floor_cheat)
    data = data.to(device)

    with torch.no_grad():
        pred_vel_norm = model(data)  # (n_particles, 3) 归一化速度

    pred_vel_norm = pred_vel_norm.cpu().numpy()

    # 反归一化
    pred_vel = denormalize(pred_vel_norm, stat[1])

    # 预测位置
    pos_pred = pos_t + pred_vel * dt

    if ransac_on_pred:
        pos_pred = apply_rigid_projection(pos_t, pos_pred, phases_dict)

    if cloth_constraints is not None:
        pos_pred = apply_cloth_distance_projection(
            pos_pred, cloth_constraints, cloth_proj_iters, cloth_proj_stiffness)

    return pos_pred


def rollout(model, trial_path, phases_dict, stat, start_timestep, rollout_steps,
            training_fpt, device, neighbor_radius=0.08, floor_cheat=True,
            ransac_on_pred=False, cloth_proj=False, cloth_proj_iters=5,
            cloth_proj_stiffness=0.8, cloth_proj_radius_scale=1.5):
    """
    多步 rollout 推理

    Returns:
        pred_positions: list of (n_particles, 3), 长度 rollout_steps
        gt_positions: list of (n_particles, 3), 长度 rollout_steps
        errors: list of dict, 每步的误差
    """
    dt = training_fpt * 0.01

    # 加载初始两帧
    t0 = start_timestep
    t_prev = max(0, t0 - training_fpt)

    pos_prev = load_positions(trial_path, t_prev, phases_dict)
    pos_t = load_positions(trial_path, t0, phases_dict)

    # 布料保距约束 (起始帧构建, 场景无 cloth 时为 None)
    cloth_constraints = None
    if cloth_proj:
        cloth_constraints = build_cloth_constraints(
            pos_t, phases_dict, cloth_proj_radius_scale)
        if cloth_constraints is not None:
            print(f"  布料保距约束: {len(cloth_constraints['rest_len'])} 对, "
                  f"iters={cloth_proj_iters}, stiffness={cloth_proj_stiffness}")

    pred_positions = []
    gt_positions = []
    errors = []

    for step in range(rollout_steps):
        t_next = t0 + (step + 1) * training_fpt

        # 预测下一帧
        pos_pred = inference_step(
            model, pos_t, pos_prev, phases_dict, stat, dt, device,
            neighbor_radius, floor_cheat, ransac_on_pred,
            cloth_constraints, cloth_proj_iters, cloth_proj_stiffness)

        # 加载真实下一帧
        pos_gt = load_positions(trial_path, t_next, phases_dict)

        # 计算误差
        mse = np.mean((pos_pred - pos_gt) ** 2)
        rmse = np.sqrt(mse)
        max_err = np.max(np.linalg.norm(pos_pred - pos_gt, axis=1))

        pred_positions.append(pos_pred)
        gt_positions.append(pos_gt)
        errors.append({
            'step': step + 1,
            'timestep': t_next,
            'mse': float(mse),
            'rmse': float(rmse),
            'max_error': float(max_err),
        })

        # 更新 (autoregressive)
        pos_prev = pos_t
        pos_t = pos_pred

    return pred_positions, gt_positions, errors


# ========== 可视化 ==========

def _compute_global_bounds(pred_positions, gt_positions):
    """计算所有帧的全局坐标范围，确保 GIF 相机不抖动"""
    all_pos = np.concatenate(pred_positions + gt_positions, axis=0)
    margin = 0.1
    bounds = {
        'x': (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin),
        'z': (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin),
        'y': (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin),
    }
    return bounds


def _render_frame(pred_pos, gt_pos, trial_name, step, total_steps, rmse, bounds):
    """渲染单帧为 numpy array (RGB)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6))

    # Ground Truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(gt_pos[:, 0], gt_pos[:, 2], gt_pos[:, 1],
                c='dodgerblue', s=2, alpha=0.7, edgecolors='none')
    ax1.set_title('Ground Truth', fontsize=13)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')

    # Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pred_pos[:, 0], pred_pos[:, 2], pred_pos[:, 1],
                c='tomato', s=2, alpha=0.7, edgecolors='none')
    ax2.set_title('Prediction', fontsize=13)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')

    # 固定坐标范围
    for ax in [ax1, ax2]:
        ax.set_xlim(*bounds['x'])
        ax.set_ylim(*bounds['z'])
        ax.set_zlim(*bounds['y'])
        ax.view_init(elev=25, azim=-60)

    plt.suptitle(f'{trial_name}\nStep {step}/{total_steps}  RMSE={rmse:.4f}',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 转为 numpy array
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return buf


def visualize_gif(pred_positions, gt_positions, errors, trial_name, outf, fps=4):
    """生成全时段 GIF 动画"""
    import imageio

    bounds = _compute_global_bounds(pred_positions, gt_positions)
    total_steps = len(pred_positions)

    frames = []
    for i in range(total_steps):
        rmse = errors[i]['rmse']
        frame = _render_frame(
            pred_positions[i], gt_positions[i],
            trial_name, i + 1, total_steps, rmse, bounds)
        frames.append(frame)

    gif_path = os.path.join(outf, f'{trial_name}.gif')
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    return gif_path


def visualize_error_curve(all_errors, outf):
    """绘制所有 trial 的误差随步数变化曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for trial_name, errors in all_errors.items():
        steps = [e['step'] for e in errors]
        rmses = [e['rmse'] for e in errors]
        mses = [e['mse'] for e in errors]
        ax1.plot(steps, rmses, marker='o', markersize=3, label=trial_name[:30])
        ax2.plot(steps, mses, marker='o', markersize=3, label=trial_name[:30])

    ax1.set_xlabel('Rollout Step')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Rollout Step')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Rollout Step')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE vs Rollout Step')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outf, 'error_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"误差曲线已保存到 {path}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='GNS-PyG Inference')

    parser.add_argument('--checkpoint', type=str,
                        default='./output_pyg/Collide/best_model.pth',
                        help='checkpoint 路径')
    parser.add_argument('--data_root', type=str,
                        default='./data/preprocessed/train',
                        help='数据根目录')
    parser.add_argument('--scenario', type=str, default='Collide',
                        help='场景名称')
    parser.add_argument('--n_trials', type=int, default=5,
                        help='选取的 trial 数量')
    parser.add_argument('--split', type=str, default='valid',
                        choices=['valid', 'train', 'all'],
                        help="which trials to sample from: 'valid' (default, <scenario>/valid.txt), "
                             "'train' (<scenario>/train.txt) or 'all' (every trial dir; legacy behaviour, "
                             "includes training trials)")
    parser.add_argument('--rollout_steps', type=int, default=10,
                        help='连续推理步数 (--full_rollout 时忽略)')
    parser.add_argument('--full_rollout', action='store_true',
                        help='从 start_timestep 一直推理到最后一个可用时间步')
    parser.add_argument('--start_timestep', type=int, default=45,
                        help='开始推理的时间步 (默认 15*fpt=45)')
    parser.add_argument('--training_fpt', type=int, default=3,
                        help='帧间隔')
    parser.add_argument('--neighbor_radius', type=float, default=0.08,
                        help='邻居搜索半径')
    parser.add_argument('--outf', type=str, default='./inference_output',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化 (GIF 动画 + 误差曲线)')
    parser.add_argument('--gif_fps', type=int, default=4,
                        help='GIF 帧率')
    parser.add_argument('--ransac_on_pred', type=int, default=0,
                        help='是否对 rigid 粒子做刚体投影 (0/1, 对齐 eval.py)')
    parser.add_argument('--cloth_proj', type=int, default=0,
                        help='是否对 cloth 粒子做保距投影 (PBD 距离约束, 0/1)')
    parser.add_argument('--cloth_proj_iters', type=int, default=5,
                        help='保距投影每步迭代轮数')
    parser.add_argument('--cloth_proj_stiffness', type=float, default=0.8,
                        help='保距投影刚度 (0~1)')
    parser.add_argument('--cloth_proj_radius_scale', type=float, default=1.5,
                        help='约束半径 = scale x median(最近邻距)')

    # 模型参数 (与训练一致)
    parser.add_argument('--nf_particle', type=int, default=200)
    parser.add_argument('--nf_relation', type=int, default=300)
    parser.add_argument('--nf_effect', type=int, default=200)
    parser.add_argument('--pstep', type=int, default=10)
    parser.add_argument('--use_layernorm', type=str, default='false',
                        help='是否使用 LayerNorm (true/false)')
    parser.add_argument('--deep_propagator', type=str, default='false',
                        help='是否使用深层 Propagator MLP (true/false)')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"RANSAC rigid projection: {bool(args.ransac_on_pred)}")

    # 解析 bool 参数
    use_ln = args.use_layernorm.lower() in ('true', '1', 'yes', 't')
    deep_prop = args.deep_propagator.lower() in ('true', '1', 'yes', 't')

    # 加载模型
    model = create_gns_model(
        nf_particle=args.nf_particle,
        nf_relation=args.nf_relation,
        nf_effect=args.nf_effect,
        pstep=args.pstep,
        residual=True,
        use_layernorm=use_ln,
        deep_propagator=deep_prop,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    ckpt_epoch = checkpoint.get('epoch', '?')
    ckpt_vloss = checkpoint.get('valid_loss', '?')
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt_epoch}, Valid Loss: {ckpt_vloss}")

    # 选取 trial
    scenario_dir = os.path.join(args.data_root, args.scenario)
    if args.split == 'all':
        # Legacy behaviour: sample from every trial directory (includes training trials).
        all_trials = sorted([
            d for d in os.listdir(scenario_dir)
            if os.path.isdir(os.path.join(scenario_dir, d))
        ])
    else:
        # Sample only from the trials listed in <scenario_dir>/<split>.txt
        # (default: valid.txt, i.e. trials never seen during training).
        split_txt = os.path.join(scenario_dir, f'{args.split}.txt')
        if not os.path.isfile(split_txt):
            raise FileNotFoundError(
                f"Split file not found: {split_txt} "
                f"(copy data_splits/<Scenario>/ into the data root, or use --split all)")
        with open(split_txt, 'r') as f:
            split_names = [line.strip() for line in f if line.strip()]
        all_trials = sorted([
            d for d in split_names
            if os.path.isdir(os.path.join(scenario_dir, d))
        ])
    print(f"Total trials in {args.scenario} (split={args.split}): {len(all_trials)}")

    selected_indices = np.random.choice(len(all_trials), args.n_trials, replace=False)
    selected_trials = [all_trials[i] for i in sorted(selected_indices)]
    print(f"Selected {args.n_trials} trials: {selected_trials}\n")

    # 创建输出目录
    os.makedirs(args.outf, exist_ok=True)

    # 推理
    all_errors = {}
    all_results = {}

    for trial_idx, trial_name in enumerate(selected_trials):
        trial_path = os.path.join(scenario_dir, trial_name)
        print(f"{'='*60}")
        print(f"Trial {trial_idx+1}/{args.n_trials}: {trial_name}")

        # 加载 phases_dict 和 stat
        phases_dict = load_phases_dict(trial_path)
        stat = load_stat(trial_path)

        # 计算 rollout 步数
        max_timestep = phases_dict['time_step']
        if args.full_rollout:
            actual_steps = (max_timestep - args.start_timestep) // args.training_fpt - 1
            print(f"  全时段 rollout: {actual_steps} 步")
        else:
            needed_end = args.start_timestep + args.rollout_steps * args.training_fpt
            if needed_end >= max_timestep:
                actual_steps = (max_timestep - args.start_timestep) // args.training_fpt - 1
                print(f"  警告: 时间步不够, rollout_steps 从 {args.rollout_steps} 减少到 {actual_steps}")
            else:
                actual_steps = args.rollout_steps

        if actual_steps <= 0:
            print(f"  跳过: 时间步不足")
            continue

        print(f"  粒子数: {phases_dict['n_particles']}, 物体数: {phases_dict['n_objects']}")
        print(f"  总时间步: {max_timestep}, 推理: step {args.start_timestep} -> {args.start_timestep + actual_steps * args.training_fpt}")

        # Rollout
        pred_positions, gt_positions, errors = rollout(
            model, trial_path, phases_dict, stat,
            args.start_timestep, actual_steps, args.training_fpt,
            device, args.neighbor_radius, True, bool(args.ransac_on_pred),
            bool(args.cloth_proj), args.cloth_proj_iters,
            args.cloth_proj_stiffness, args.cloth_proj_radius_scale)

        all_errors[trial_name] = errors
        all_results[trial_name] = {
            'trial_path': trial_path,
            'n_particles': phases_dict['n_particles'],
            'n_objects': phases_dict['n_objects'],
            'start_timestep': args.start_timestep,
            'rollout_steps': actual_steps,
            'errors': errors,
        }

        # 打印误差
        print(f"\n  {'Step':>6} {'Timestep':>10} {'RMSE':>12} {'MSE':>12} {'MaxErr':>12}")
        print(f"  {'-'*54}")
        for e in errors:
            print(f"  {e['step']:>6} {e['timestep']:>10} {e['rmse']:>12.6f} {e['mse']:>12.8f} {e['max_error']:>12.6f}")

        # 生成 GIF
        if args.visualize:
            print(f"\n  正在渲染 GIF ({len(pred_positions)} 帧)...")
            gif_path = visualize_gif(
                pred_positions, gt_positions, errors,
                trial_name, args.outf, fps=args.gif_fps)
            print(f"  GIF 已保存: {gif_path}")

        print()

    # 汇总
    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")

    # 按步聚合
    max_steps = max(len(e) for e in all_errors.values()) if all_errors else 0
    for step in range(1, max_steps + 1):
        rmses = []
        for errors in all_errors.values():
            if step <= len(errors):
                rmses.append(errors[step-1]['rmse'])
        if rmses:
            print(f"  Step {step:>3}: 平均 RMSE = {np.mean(rmses):.6f}, "
                  f"std = {np.std(rmses):.6f}, "
                  f"min = {np.min(rmses):.6f}, max = {np.max(rmses):.6f}")

    # 保存结果
    results_path = os.path.join(args.outf, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 {results_path}")

    # 误差曲线
    if args.visualize and all_errors:
        visualize_error_curve(all_errors, args.outf)


if __name__ == '__main__':
    main()
