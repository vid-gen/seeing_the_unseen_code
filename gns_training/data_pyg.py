"""
data_pyg.py - PyG 格式的 Physion 数据集 (v4)

v4 关键修复:
- floor 归一化时机：先加 floor 再归一化
- 噪声缩放：使用 noise_std / sqrt(n_velsteps)
"""

import os
import pickle
import copy
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class PhysionPyGDataset(Dataset):
    """PyG 格式的 Physion 数据集 (v4)"""

    def __init__(
        self,
        data_root: str,
        scenarios: list,
        phase: str = 'train',
        neighbor_radius: float = 0.08,
        noise_std: float = 3e-4,
        training_fpt: int = 3,
        floor_cheat: bool = True,
        subsample_limit: int = 3000,
        train_samples_per_trial: int = None,
        stat_path: str = None,
        unroll_steps: int = 1,
        transform=None,
        pre_transform=None,
    ):
        self.data_root = data_root
        # Keep scenario parsing robust against inputs like "A, B".
        self.scenarios = [s.strip() for s in scenarios if str(s).strip()]
        self.phase = phase
        self.neighbor_radius = neighbor_radius
        self.noise_std = noise_std
        self.training_fpt = training_fpt
        self.dt = training_fpt * 0.01
        self.floor_cheat = floor_cheat
        self.subsample_limit = subsample_limit
        self.train_samples_per_trial = train_samples_per_trial
        self.start_timestep = int(15 * training_fpt)
        self.unroll_steps = max(1, unroll_steps)

        self.stat = self._load_stat(stat_path)
        self.all_trials, self.trial_scenarios = self._load_trial_list()
        self.n_trials = len(self.all_trials)

        if self.n_trials == 0:
            raise RuntimeError(
                f"No valid trials found for scenarios={self.scenarios} under {self.data_root}")

        # Scenario -> trial indices mapping, used by custom samplers.
        self.scenario_to_trial_indices = {}
        for trial_idx, scenario in enumerate(self.trial_scenarios):
            self.scenario_to_trial_indices.setdefault(scenario, []).append(trial_idx)

        if phase == 'train':
            if self.train_samples_per_trial is None:
                self.samples_per_trial = max(1, int(13499 / self.n_trials) + 1)
            else:
                self.samples_per_trial = max(1, int(self.train_samples_per_trial))
        else:
            self.samples_per_trial = 1

        super().__init__(data_root, transform, pre_transform)

    def _load_stat(self, stat_path):
        """加载归一化统计 (实际数据中可能全为 0)"""
        if stat_path and os.path.exists(stat_path):
            with h5py.File(stat_path, 'r') as f:
                positions_stat = np.array(f['positions'])
                velocities_stat = np.array(f['velocities'])
            return [positions_stat, velocities_stat]
        else:
            return [np.zeros((3, 3)), np.zeros((3, 3))]

    def _load_trial_list(self):
        all_trials = []
        trial_scenarios = []
        for scenario in self.scenarios:
            txt_path = os.path.join(self.data_root, scenario, f'{self.phase}.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    trial_names = [line.strip() for line in f if line.strip()]
            else:
                scenario_dir = os.path.join(self.data_root, scenario)
                trial_names = [d for d in os.listdir(scenario_dir)
                              if os.path.isdir(os.path.join(scenario_dir, d))]
            for trial_name in trial_names:
                trial_path = os.path.join(self.data_root, scenario, trial_name)
                if os.path.exists(os.path.join(trial_path, 'phases_dict.pkl')):
                    all_trials.append(trial_path)
                    trial_scenarios.append(scenario)
        return all_trials, trial_scenarios

    def len(self):
        return self.n_trials * self.samples_per_trial

    def get(self, idx):
        """获取单个样本 (支持 pushforward 多帧)"""
        trial_idx = idx % self.n_trials
        trial_path = self.all_trials[trial_idx]

        phases_dict = self._load_phases_dict(trial_path)

        K = self.unroll_steps
        # 确保有足够时间步: 需要 t-1, t, t+1, ..., t+K (K+2 帧)
        max_timestep = phases_dict['time_step'] - K * self.training_fpt
        if max_timestep <= self.start_timestep:
            K = max(1, (phases_dict['time_step'] - self.start_timestep) // self.training_fpt - 1)
            max_timestep = phases_dict['time_step'] - K * self.training_fpt
        timestep = np.random.randint(self.start_timestep, max_timestep)

        # 加载三帧 (t-1, t, t+1)
        timestep_prev = max(0, timestep - self.training_fpt)
        timestep_nxt = timestep + self.training_fpt

        pos_prev = self._load_positions(trial_path, timestep_prev, phases_dict)
        pos_t = self._load_positions(trial_path, timestep, phases_dict)
        pos_nxt = self._load_positions(trial_path, timestep_nxt, phases_dict)

        # 计算速度 (避免标签泄漏)
        vel_t = (pos_t - pos_prev) / self.dt
        vel_nxt = (pos_nxt - pos_t) / self.dt

        # 噪声注入 (与源代码一致: train/valid 只要 noise_std>0 都注入)
        if self.noise_std > 0:
            # 源代码: shape (n_velsteps, vel_dim) = (2, 3)
            # noise[0] 不使用, noise[1] 广播到所有粒子 (全局速度偏移)
            n_velsteps = 2
            vel_dim = vel_t.shape[1]  # = 3
            vel_sequence_noise = np.random.normal(
                0, self.noise_std / (n_velsteps ** 0.5),
                size=(n_velsteps, vel_dim))
            vel_t = vel_t + vel_sequence_noise[1]  # (n_particles, 3) + (3,) 广播
            pos_t = pos_prev + vel_t * self.dt

        n_particles = pos_t.shape[0]

        # ===== v4 修复: 先加 floor，再归一化 =====
        # 保存原始坐标 (用于构边)
        pos_t_raw = pos_t.copy()

        # 先添加 floor 节点
        if self.floor_cheat:
            pos_t_raw = np.vstack([pos_t_raw, np.zeros((1, 3))])
            pos_t = np.vstack([pos_t, np.zeros((1, 3))])
            vel_t = np.vstack([vel_t, np.zeros((1, 3))])
            vel_nxt = np.vstack([vel_nxt, np.zeros((1, 3))])

        # 再一起归一化
        pos_t_norm = self._normalize(pos_t, self.stat[0])
        vel_t_norm = self._normalize(vel_t, self.stat[1])
        vel_nxt_norm = self._normalize(vel_nxt, self.stat[1])

        # 标签 (只取粒子，不含 floor)
        label = vel_nxt_norm[:n_particles]

        # 构建 object_id 和 is_rigid
        object_id, is_rigid = self._build_object_id(phases_dict, n_particles)

        # 构建节点属性
        attr = self._build_node_attr(phases_dict, n_particles)

        # 用原始坐标构边 (v3 已修复)
        edge_index, edge_attr = self._build_edges_v4(
            pos_t_raw, pos_t_norm, n_particles)

        # 添加 floor 边
        if self.floor_cheat:
            floor_edge_index, floor_edge_attr = self._build_floor_edges_v4(
                pos_t_raw, pos_t_norm, n_particles)
            if floor_edge_index.shape[1] > 0:
                edge_index = np.hstack([edge_index, floor_edge_index])
                edge_attr = np.vstack([edge_attr, floor_edge_attr])

        # 节点特征: attr + vel_masked
        n_nodes = pos_t_norm.shape[0]
        vel_masked = np.zeros((n_nodes, 6), dtype=np.float32)
        vel_masked[:, 3:6] = vel_t_norm

        node_features = np.concatenate([attr, vel_masked], axis=1)  # (N, 9)

        # state 用于计算 offset
        state = np.concatenate([pos_t_norm, vel_t_norm], axis=1)  # (N, 6)

        # 创建 PyG Data 对象
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.FloatTensor(label),
            n_particles=n_particles,
            state=torch.FloatTensor(state),
            pos_raw=torch.FloatTensor(pos_t_raw),
            object_id=torch.LongTensor(object_id),
            is_rigid=torch.BoolTensor(is_rigid),
        )

        # ===== 刚体信息 (始终存储, rigid projection 需要) =====
        data.instance_idx = torch.LongTensor(phases_dict['instance_idx'])
        materials_encoded = []
        for m in phases_dict['material']:
            materials_encoded.append(1 if m == 'rigid' else 0)
        data.materials_is_rigid = torch.LongTensor(materials_encoded)
        data.n_objects = len(phases_dict['instance_idx']) - 1

        # ===== Pushforward 多帧数据 =====
        if K > 1:
            # 加载 t+2, t+3, ..., t+K 的位置 (t+1 已加载为 pos_nxt)
            # 重新加载原始 pos_t (无噪声) 和 pos_nxt 用于 GT 速度计算
            pos_t_raw_clean = self._load_positions(trial_path, timestep, phases_dict)
            pos_nxt_raw_clean = self._load_positions(trial_path, timestep_nxt, phases_dict)

            # 收集所有未来帧的原始位置: [pos_t_raw, pos_{t+1}_raw, ..., pos_{t+K}_raw]
            future_raw_positions = [pos_t_raw_clean, pos_nxt_raw_clean]
            for k in range(2, K + 1):
                t_future = timestep + k * self.training_fpt
                pos_future = self._load_positions(trial_path, t_future, phases_dict)
                future_raw_positions.append(pos_future)

            # 计算每步的 GT 归一化速度: vel_k = normalize((pos_{t+k} - pos_{t+k-1}) / dt)
            gt_vel_norms = []
            for k in range(K):
                gt_vel = (future_raw_positions[k + 1] - future_raw_positions[k]) / self.dt
                gt_vel_norm = self._normalize(gt_vel, self.stat[1])
                gt_vel_norms.append(gt_vel_norm)

            # ---- 逐步存储 (node-level, 可 batch_size>1 拼接) ----
            for k in range(K):
                setattr(data, f'gt_vel_step_{k}', torch.FloatTensor(gt_vel_norms[k]))
            data.n_unroll_steps = K

        return data

    def _build_object_id(self, phases_dict, n_particles):
        n_nodes = n_particles + (1 if self.floor_cheat else 0)
        object_id = np.zeros(n_nodes, dtype=np.int64)
        is_rigid = np.zeros(n_nodes, dtype=bool)

        instance_idx = phases_dict['instance_idx']
        materials = phases_dict['material']

        for obj_id in range(len(instance_idx) - 1):
            st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
            object_id[st:ed] = obj_id
            if materials[obj_id] == 'rigid':
                is_rigid[st:ed] = True

        if self.floor_cheat:
            object_id[-1] = -1

        return object_id, is_rigid

    def _build_edges_v4(self, pos_raw, pos_norm, n_particles):
        """用原始坐标构建邻居边，edge_attr 用归一化坐标"""
        pos = pos_raw[:n_particles, :3]
        tree = cKDTree(pos)
        neighbors_list = tree.query_ball_point(pos, self.neighbor_radius, p=2)

        senders, receivers = [], []
        for i, neighbors in enumerate(neighbors_list):
            for j in neighbors:
                senders.append(j)
                receivers.append(i)

        if len(senders) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        edge_index = np.array([senders, receivers], dtype=np.int64)

        # edge_attr 用归一化坐标
        pos_r = pos_norm[receivers, :3]
        pos_s = pos_norm[senders, :3]
        relative_pos = pos_r - pos_s
        distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)
        edge_attr = np.concatenate([relative_pos, distance], axis=1).astype(np.float32)

        return edge_index, edge_attr

    def _build_floor_edges_v4(self, pos_raw, pos_norm, n_particles):
        """floor 判断用原始坐标，edge_attr 用归一化坐标"""
        if not self.floor_cheat:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

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

    def _normalize(self, data, stat):
        mean = stat[:, 0]
        std = stat[:, 1]
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std

    def _build_node_attr(self, phases_dict, n_particles):
        n_nodes = n_particles + (1 if self.floor_cheat else 0)
        attr = np.zeros((n_nodes, 3), dtype=np.float32)

        # attr 语义: 刚体粒子 [0,0,0], 软体粒子 (cloth/fluid) [0,1,0], floor [0,0,1]
        # dim1 与源代码 DPI 分支约定一致 (fluid/cloth 共用); rigid 的几何信息仍由 offset 提供
        instance_idx = phases_dict['instance_idx']
        materials = phases_dict['material']
        for obj_id in range(len(instance_idx) - 1):
            if materials[obj_id] != 'rigid':
                st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
                attr[st:ed, 1] = 1

        if self.floor_cheat:
            attr[-1, 2] = 1

        return attr

    def _load_phases_dict(self, trial_path):
        pkl_path = os.path.join(trial_path, 'phases_dict.pkl')
        with open(pkl_path, 'rb') as f:
            phases_dict = pickle.load(f)
        phases_dict['trial_dir'] = trial_path
        self._correct_bad_chair(phases_dict)
        self._remove_large_obstacles(phases_dict)
        self._subsample_particles(phases_dict, self.subsample_limit)
        return phases_dict

    def _load_positions(self, trial_path, timestep, phases_dict):
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
                    # Match upstream data.py exactly to minimize numeric drift.
                    transformed_pts = np.matmul(rot, pts.T).T + np.expand_dims(trans, axis=0)
                    all_positions.append(transformed_pts)
                positions = np.concatenate(all_positions, axis=0)

        return positions.astype(np.float32)

    def _correct_bad_chair(self, phases_dict):
        if len(phases_dict["instance_idx"]) - 1 != phases_dict["n_objects"]:
            obj_points = []
            n_empty_obj = 0
            opt_ids = []
            for opt_id, opts in enumerate(phases_dict["obj_points"]):
                if opts.shape[0] != 0:
                    obj_points.append(opts)
                    opt_ids.append(opt_id)
                else:
                    n_empty_obj += 1
            phases_dict["obj_points"] = obj_points
            phases_dict["before_fix_n_objects"] = phases_dict["n_objects"]
            phases_dict["n_objects"] = len(obj_points)
            phases_dict["bad_lamp"] = True
            phases_dict["ok_obj_id"] = opt_ids
            assert len(phases_dict["instance_idx"]) - 1 == phases_dict["n_objects"]
            return True
        else:
            # Keep the same special-case behavior as upstream data.py.
            if "drop" in phases_dict["trial_dir"] and "train/50" in phases_dict["trial_dir"]:
                n_empty_obj = 0
                opt_ids = []
                for opt_id, opts in enumerate(phases_dict["obj_points"]):
                    if opts.shape[0] != 0:
                        opt_ids.append(opt_id)
                    else:
                        n_empty_obj += 1
                if n_empty_obj > 0:
                    list_items = ["root_des_radius", "root_num", "clusters", "instance", "material", "obj_points"]
                    for item in list_items:
                        phases_dict[item] = [phases_dict[item][a] for a in opt_ids]
                    new_instance_idx = [0]
                    for obj_pts in phases_dict["obj_points"]:
                        new_instance_idx.append(new_instance_idx[-1] + obj_pts.shape[0])
                    phases_dict["instance_idx"] = new_instance_idx
                    phases_dict["n_objects"] = len(phases_dict["obj_points"])
                    phases_dict["ok_obj_id"] = opt_ids
                    assert phases_dict["n_particles"] == new_instance_idx[-1]
                    assert len(phases_dict["instance_idx"]) - 1 == phases_dict["n_objects"]
                    assert len(phases_dict["root_num"]) == phases_dict["n_objects"]
                    return True
                return False
            return False

    def _remove_large_obstacles(self, phases_dict):
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
                phases_dict[item] = [phases_dict[item][a] for a in ok_obj_id]
            phases_dict["n_objects"] = len(ok_obj_id)
            phases_dict["n_particles"] = sum(objs_npts)
            phases_dict["instance_idx"] = new_instance_idx
            phases_dict["ok_obj_id"] = ok_obj_id

    def _subsample_particles(self, phases_dict, limit=3000):
        n_objects = phases_dict["n_objects"]
        obj_names = phases_dict["instance"]
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
                phases_dict["clusters"][obj_id][0][0] = \
                    phases_dict["clusters"][obj_id][0][0][selected_idxs]
            else:
                obj_subsample_idx.append(np.array(range(obj_npts)))
            current_particle_idx = new_instance_idx[-1] + obj_points[obj_id].shape[0]
            new_instance_idx.append(current_particle_idx)
        phases_dict["n_particles"] = new_instance_idx[-1]
        phases_dict["obj_points"] = obj_points
        phases_dict["instance_idx_before_subsample"] = copy.deepcopy(phases_dict["instance_idx"])
        phases_dict["instance_idx"] = new_instance_idx
        phases_dict["obj_subsample_idx"] = obj_subsample_idx
        phases_dict["is_subsample"] = is_subsample


def create_pyg_dataloaders(data_root, scenarios, batch_size=32, num_workers=8, stat_path=None, **kwargs):
    train_dataset = PhysionPyGDataset(data_root, scenarios, 'train', stat_path=stat_path, **kwargs)
    valid_dataset = PhysionPyGDataset(data_root, scenarios, 'valid', stat_path=stat_path, **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, valid_loader
