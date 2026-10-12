# Source: htung0101/Physion-particles @ 5ac002a (upstream file) + local modifications used to preprocess the
# training data of this release; see patches/preprocessing_vs_upstream_5ac002a.patch and README.md.
import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py
import copy


import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import rand_float, rand_int

import ipdb
_st=ipdb.set_trace


# HDF5 数据缓存，避免重复读取文件
_h5_cache = {}
_h5_cache_maxsize = 1000

def load_h5_cached(path, data_names_tuple):
    """缓存 HDF5 文件内容，加速数据加载

    Args:
        path: HDF5 文件路径
        data_names_tuple: 要加载的数据名称元组 (必须是元组以便哈希)

    Returns:
        dict: 数据名称到 numpy 数组的映射 (返回副本以避免缓存污染)
    """
    cache_key = (path, data_names_tuple)
    if cache_key not in _h5_cache:
        # 缓存容量管理
        if len(_h5_cache) >= _h5_cache_maxsize:
            # 删除最老的一半条目
            keys_to_remove = list(_h5_cache.keys())[:len(_h5_cache) // 2]
            for k in keys_to_remove:
                del _h5_cache[k]

        with h5py.File(path, 'r') as hf:
            _h5_cache[cache_key] = {k: np.array(hf.get(k)) for k in data_names_tuple if k in hf.keys()}

    # 返回副本以避免原地修改污染缓存
    return {k: v.copy() for k, v in _h5_cache[cache_key].items()}


def collate_fn(data):
    return data[0]


def collate_fn_batched(data):
    """合并多个样本为批处理图 (用于 PyG 模型)

    Args:
        data: list of tuples from __getitem__

    Returns:
        dict: 包含批处理数据的字典
    """
    positions_list, velocities_list, particle_types_list = [], [], []
    n_particles_list, labels_list, phases_dicts = [], [], []
    instance_idxs_list = []

    for sample in data:
        attr, state, rels, n_particles, n_shapes, instance_idx, label, phases_dict = sample

        # state: [n_nodes, state_dim] where state_dim = 6 (pos + vel)
        # 只取粒子部分，不包含 shape nodes
        positions_list.append(state[:n_particles, :3].numpy())  # 位置
        velocities_list.append(state[:n_particles, 3:6].numpy())  # 速度

        # attr: [n_nodes, attr_dim]
        # 粒子类型：attr[:, 0] = rigid, attr[:, 1] = fluid
        # 使用 argmax 获取类型索引
        particle_types_list.append(attr[:n_particles, :2].argmax(dim=1).numpy())

        n_particles_list.append(n_particles)
        labels_list.append(label.numpy())
        phases_dicts.append(phases_dict)
        instance_idxs_list.append(instance_idx)

    return {
        'positions': torch.tensor(np.vstack(positions_list), dtype=torch.float32),
        'velocities': torch.tensor(np.vstack(velocities_list), dtype=torch.float32),
        'particle_types': torch.tensor(np.concatenate(particle_types_list), dtype=torch.long),
        'n_particles_per_example': torch.tensor(n_particles_list, dtype=torch.int32),
        'labels': torch.tensor(np.vstack(labels_list), dtype=torch.float32),
        'phases_dicts': phases_dicts,
        'instance_idxs': instance_idxs_list
    }


def _radius_graph_scipy(positions, radius, batch_ids, max_num_neighbors=64):
    """使用 scipy cKDTree 构建 radius graph (CPU fallback)

    Args:
        positions: [N, 3] 粒子位置
        radius: 邻居搜索半径
        batch_ids: [N] 每个粒子所属的 batch
        max_num_neighbors: 最大邻居数

    Returns:
        edge_index: [2, E] 边索引
    """
    pos_np = positions.cpu().numpy()
    batch_np = batch_ids.cpu().numpy()

    senders_list, receivers_list = [], []
    n_batches = int(batch_np.max()) + 1

    for b in range(n_batches):
        mask = batch_np == b
        idx = np.where(mask)[0]
        pos_b = pos_np[mask]

        if len(pos_b) == 0:
            continue

        tree = spatial.cKDTree(pos_b)
        neighbors = tree.query_ball_point(pos_b, radius, p=2)

        for i, neigh in enumerate(neighbors):
            # 排除自环
            neigh = [n for n in neigh if n != i]
            # 限制邻居数量
            if len(neigh) > max_num_neighbors:
                neigh = neigh[:max_num_neighbors]

            for j in neigh:
                senders_list.append(idx[j])
                receivers_list.append(idx[i])

    if len(senders_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

    edge_index = torch.tensor([senders_list, receivers_list], dtype=torch.long, device=positions.device)
    return edge_index


def prepare_input_batched(batch_data, args, stat, device):
    """批处理图构建 (支持 GPU 加速和 CPU fallback)

    Args:
        batch_data: collate_fn_batched 返回的字典
        args: 训练参数
        stat: 归一化统计量
        device: 目标设备

    Returns:
        state: 归一化后的节点特征 [total_particles, state_dim]
        edge_index: 边索引 [2, n_edges]
        edge_attr: 边特征 [n_edges, 4] (相对位置 + 距离)
        batch_ids: 每个粒子所属的样本 ID [total_particles]
    """
    # 尝试导入 PyG radius_graph
    try:
        from torch_geometric.nn import radius_graph
        USE_PYG_RADIUS = True
    except (ImportError, RuntimeError):
        USE_PYG_RADIUS = False

    positions = batch_data['positions'].to(device)
    velocities = batch_data['velocities'].to(device)
    particle_types = batch_data['particle_types'].to(device)
    n_particles_per_example = batch_data['n_particles_per_example']

    # 确保 std 不为 0
    mean_p = torch.tensor(stat[0][:3, 0], dtype=torch.float32, device=device)
    std_p = torch.tensor(stat[0][:3, 1], dtype=torch.float32, device=device)
    std_p = torch.where(std_p == 0, torch.ones_like(std_p), std_p)

    mean_v = torch.tensor(stat[1][:, 0], dtype=torch.float32, device=device)
    std_v = torch.tensor(stat[1][:, 1], dtype=torch.float32, device=device)
    std_v = torch.where(std_v == 0, torch.ones_like(std_v), std_v)

    # 归一化
    positions_norm = (positions - mean_p) / std_p
    velocities_norm = (velocities - mean_v) / std_v

    # 构建 batch 索引
    batch_ids = torch.cat([
        torch.full((n.item(),), i, dtype=torch.long, device=device)
        for i, n in enumerate(n_particles_per_example)
    ])

    # 图构建 - 优先使用 PyG radius_graph，否则使用 scipy fallback
    if USE_PYG_RADIUS:
        try:
            edge_index = radius_graph(
                positions,
                r=args.neighbor_radius,
                batch=batch_ids,
                loop=False,
                max_num_neighbors=64
            )
        except (ImportError, RuntimeError):
            # torch-cluster 加载失败时使用 scipy fallback
            edge_index = _radius_graph_scipy(positions, args.neighbor_radius, batch_ids, max_num_neighbors=64)
    else:
        edge_index = _radius_graph_scipy(positions, args.neighbor_radius, batch_ids, max_num_neighbors=64)

    # 边特征: 相对位置 + 相对距离
    senders, receivers = edge_index[0], edge_index[1]
    relative_pos = positions[receivers] - positions[senders]
    relative_dist = torch.norm(relative_pos, dim=-1, keepdim=True)

    # 归一化边特征
    edge_attr = torch.cat([
        relative_pos / args.neighbor_radius,
        relative_dist / args.neighbor_radius
    ], dim=-1)

    # 节点特征: 归一化位置 + 归一化速度
    state = torch.cat([positions_norm, velocities_norm], dim=-1)

    return state, edge_index, edge_attr, batch_ids


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path, mode=None):
    if not isinstance(path, list):
        paths = [path]
        one_item = True
    else:
        paths = path
        one_item = False

    multiple_data = []
    for path in paths:
        hf = h5py.File(path, 'r')
        data = []
        for i in range(len(data_names)):
            d = np.array(hf.get(data_names[i]))
            data.append(d)
        hf.close()
        multiple_data.append(data)

    if one_item or len(multiple_data) == 1:
        return multiple_data[0]

    if mode == "avg":
        nitems = len(multiple_data[0])
        outputs = []
        for itemid in range(nitems):
           outputs.append(np.mean(np.stack([data[itemid] for data in multiple_data], 0), 0))
        return outputs
    else:
        raise ValueError


    return data

def load_data_dominoes(data_names, path, static_data_info, load_data_names=["obj_positions", "obj_rotations"]):
    """
    static_data_info: num_objects, object point cloud


    """
    if not isinstance(path, list):
        paths = [path]
        one_item = True
    else:
        paths = path
        one_item = False

    n_objects = static_data_info["n_objects"]
    obj_points = static_data_info["obj_points"]


    flex_engine = False
    if "Drape" in static_data_info["trial_dir"]:
        flex_engine = True
        load_data_names = ['particle_positions', "particle_velocities"]

    if "is_subsample" in static_data_info:
        is_subsample = static_data_info["is_subsample"]
        obj_subsample_idx = static_data_info["obj_subsample_idx"]
        instance_idx_old = static_data_info["instance_idx_before_subsample"]

    else:
        is_subsample = False
    #import ipdb; ipdb.set_trace()
    multiple_data = []
    for path in paths:
        # 使用缓存读取 HDF5 文件
        data_raw = load_h5_cached(path, tuple(load_data_names))
        data = []

        for data_name in data_names:
            if data_name == "positions":

                if flex_engine:

                    particle_positions = data_raw["particle_positions"]


                    particle_positions *= 0.05/0.035
                    if is_subsample:
                        n_particles = static_data_info["n_particles"]
                        positions = []

                        for obj_id in range(n_objects):
                            st, ed = instance_idx_old[obj_id], instance_idx_old[obj_id + 1]
                            pos = particle_positions[st:ed, :]
                            positions.append(pos[obj_subsample_idx[obj_id]])


                        particle_positions = np.concatenate(positions, axis=0)

                        # add scale correction
                        assert(n_particles == particle_positions.shape[0])

                    data.append(particle_positions)
                else:
                    transformed_obj_pts = []
                    # object point cloud and rotation/positions to compute particles
                    obj_rotations = data_raw["obj_rotations"]
                    obj_positions = data_raw["obj_positions"]

                    if "ok_obj_id" in static_data_info:

                        if "bad_lamp" in static_data_info and obj_positions.shape[0] != static_data_info["before_fix_n_objects"]:
                            print("good trial with bad lamp", static_data_info["trial_dir"])
                            import ipdb; ipdb.set_trace()

                        # print(obj_rotations.shape)
                        # print(obj_positions.shape)
                        # print(len(obj_points))
                        # print("ok_obj_id", static_data_info["ok_obj_id"])

                        #if (len(obj_points) < obj_rotations.shape[0]):
                        #    import ipdb; ipdb.set_trace()
                        for idx, obj_id in enumerate(static_data_info["ok_obj_id"]):
                            rot = R.from_quat(obj_rotations[obj_id]).as_matrix()
                            trans = obj_positions[obj_id]
                            transformed_pts = np.matmul(rot, obj_points[idx].T).T + np.expand_dims(trans, axis=0)
                            transformed_obj_pts.append(transformed_pts)

                    else:
                        #print("#######", n_objects, obj_rotations.shape, obj_positions.shape)
                        if not n_objects == obj_rotations.shape[0]:
                            import ipdb; ipdb.set_trace()
                        assert(n_objects == obj_rotations.shape[0])
                        for obj_id in range(n_objects):
                            rot = R.from_quat(obj_rotations[obj_id]).as_matrix()
                            trans = obj_positions[obj_id]
                            transformed_pts = np.matmul(rot, obj_points[obj_id].T).T + np.expand_dims(trans, axis=0)
                            transformed_obj_pts.append(transformed_pts)
                    positions = np.concatenate(transformed_obj_pts, axis=0)
                    # if is_subsample:
                    #     positions_tmp = []
                    #     for obj_id in range(n_objects):
                    #         st, ed = instance_idx_old[obj_id], instance_idx_old[obj_id + 1]

                    #         pos = positions[st:ed, :]

                    #         positions_tmp.append(pos[obj_subsample_idx[obj_id], :])
                    #     positions = np.concatenate(positions_tmp, axis=0)
                    #     print("2", positions.shape)
                    #import ipdb; ipdb.set_trace()
                    data.append(positions)

            elif data_name == "velocities":
                # should compute later on
                data.append(None)

            else:
                raise (ValueError, f"{data_name} not supported")
        multiple_data.append(data)

    if one_item:
        return multiple_data[0]

    if mode == "avg":
        nitems = len(multiple_data[0])
        outputs = []
        for itemid in range(nitems):
           outputs.append(np.mean(np.stack([data[itemid] for data in multiple_data], 0), 0))
        return outputs
    else:
        raise ValueError


    return data

def recalculate_velocities(list_of_data, dt, data_names):
    """
    input:
        list_of_data: list of data starting from the oldest
    output:
        return the position, velocities for (len(list_of_data) - 1) data
    """
    positions_over_T = []
    velocities_over_T = []

    for data in list_of_data:
        positions = data[data_names.index("positions")]
        velocities = data[data_names.index("velocities")]
        positions_over_T.append(positions)
        velocities_over_T.append(velocities)

    output_list_of_data = []
    for t in range(len(list_of_data)):

        current_data = []
        for item in data_names:
            if item == "positions":
                current_data.append(positions_over_T[t])
            elif item == "velocities":
                if t == 0:
                    current_data.append(velocities_over_T[t])
                else:
                    current_data.append((positions_over_T[t] - positions_over_T[t - 1]) / dt)
            else:
                raise (ValueError, f"not supporting augmentation for {item}")
        output_list_of_data.append(current_data)

    return output_list_of_data

def remove_large_obstacles(phases_dict):
    if phases_dict["n_particles"] > 3000:
        #critical_objects = [b'cube', b'sphere', b'cone', b'torus', b'cylinder']
        #print(np.array(phases_dict["instance_idx"][1:]) - np.array(phases_dict["instance_idx"][:-1]))

        critical_objects = [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']
        import copy
        old_phases_dict = copy.deepcopy(phases_dict)

        n_objects = phases_dict["n_objects"]

        objs_npts = []
        ok_obj_id = []
        is_removed = False
        #print(n_objects)
        for obj_id in range(n_objects):
            #print("<<<<<<", len(phases_dict["instance_idx"]), obj_id)
            ##print(phases_dict["instance_idx"])
            #print(phases_dict["instance_idx"][obj_id])
            #print(phases_dict["instance_idx"][obj_id + 1])
            obj_npts = phases_dict["instance_idx"][obj_id + 1] - phases_dict["instance_idx"][obj_id]
            #print(obj_id, obj_npts, phases_dict["instance"][obj_id])
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
            phases_dict["n_particles"] = np.sum(objs_npts)
            phases_dict["instance_idx"] = new_instance_idx
            phases_dict["ok_obj_id"] = ok_obj_id

            #print("before", old_phases_dict["instance"])
            #print("after", phases_dict["instance"])
            return True

            #print("after", np.array(phases_dict["instance_idx"][1:]) - np.array(phases_dict["instance_idx"][:-1]))
            #print("obj points shape", [item.shape[0] for item in phases_dict["obj_points"]])
            #print("n_particles", phases_dict["n_particles"])
            #import ipdb; ipdb.set_trace()
        return False
    return False

def subsample_particles_on_large_objects(phases_dict, limit=3000):
    n_objects = phases_dict["n_objects"]
    obj_names = phases_dict["instance"]
    obj_subsample_idx = []

    obj_points = phases_dict["obj_points"]
    #print([obj_pts.shape[0] for obj_pts in obj_points])
    new_instance_idx = [0]
    is_subsample = False
    for obj_id in range(n_objects):
        obj_npts = phases_dict["instance_idx"][obj_id + 1] - phases_dict["instance_idx"][obj_id]
        #print(obj_id, obj_npts, phases_dict["instance"][obj_id])
        if obj_npts > limit and obj_names[obj_id] != b'cloth_square':
            selected_idxs = np.random.choice(obj_npts, limit, replace=False)
            obj_points[obj_id] = obj_points[obj_id][selected_idxs]
            obj_subsample_idx.append(selected_idxs)
            # phases_dict["clusters"][0][0][0] -> phases_dict["clusters"][obj_id][nodeid_in_a_root_set][0]
            is_subsample = True
            phases_dict["clusters"][obj_id][0][0] = phases_dict["clusters"][obj_id][0][0][selected_idxs]
        else:
            obj_subsample_idx.append(np.array(range(obj_npts)))
        current_particle_idx = new_instance_idx[-1] + obj_points[obj_id].shape[0]
        new_instance_idx.append(current_particle_idx)

    # if is_subsample:
    #     print("after")
    #     print(phases_dict["instance"])

    #     print([obj_pts.shape[0] for obj_pts in obj_points])
    #     print(new_instance_idx)


    phases_dict["n_particles"] = new_instance_idx[-1]
    phases_dict["obj_points"] = obj_points
    phases_dict["instance_idx_before_subsample"] = copy.deepcopy(phases_dict["instance_idx"])
    phases_dict["instance_idx"] = new_instance_idx
    phases_dict["obj_subsample_idx"] = obj_subsample_idx
    phases_dict["is_subsample"] = is_subsample
    return is_subsample

def correct_bad_chair(phases_dict):
    """
    bad chair b'648972_chair_poliform_harmony' is not completely removed in current data
    try to fix it here
    """
    if len(phases_dict["instance_idx"]) - 1 != phases_dict["n_objects"]:
        # remove the empty object
        obj_points = []
        n_empty_obj = 0
        opt_ids = []
        for opt_id, opts in enumerate(phases_dict["obj_points"]):
            if not opts.shape[0] == 0:
                obj_points.append(opts)
                opt_ids.append(opt_id)
            else:
                n_empty_obj += 1
        phases_dict["obj_points"] = obj_points
        phases_dict["before_fix_n_objects"] = phases_dict["n_objects"]
        phases_dict["n_objects"] = len(obj_points)
        phases_dict["bad_lamp"] = True
        phases_dict["ok_obj_id"] = opt_ids
        assert(len(phases_dict["instance_idx"]) - 1 == phases_dict["n_objects"])
        return True
    else:
        # there is empty mesh in drop

        if "drop" in phases_dict["trial_dir"] and "train/50" in phases_dict["trial_dir"]:

            n_empty_obj = 0
            opt_ids = []
            for opt_id, opts in enumerate(phases_dict["obj_points"]):
                if not opts.shape[0] == 0:
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

                assert(phases_dict["n_particles"] == new_instance_idx[-1])
                assert(len(phases_dict["instance_idx"]) - 1 == phases_dict["n_objects"])
                assert(len(phases_dict["root_num"]) == phases_dict["n_objects"])
                return True
            else:
                return False


        return False

def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            stat_dim = stat[i].shape[0]

            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


def rotateByQuat(p, quat):
    R = np.zeros((3, 3))
    a, b, c, d = quat[3], quat[0], quat[1], quat[2]
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * b * c - 2 * a * d
    R[0, 2] = 2 * b * d + 2 * a * c
    R[1, 0] = 2 * b * c + 2 * a * d
    R[1, 1] = a**2 - b**2 + c**2 - d**2
    R[1, 2] = 2 * c * d - 2 * a * b
    R[2, 0] = 2 * b * d - 2 * a * c
    R[2, 1] = 2 * c * d + 2 * a * b
    R[2, 2] = a**2 - b**2 - c**2 + d**2

    return np.dot(R, p)


def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    ax.set_aspect('equal')

    plt.show()


def find_relations_neighbor(positions, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions


    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)


    # import scipy
    # mat = scipy.spatial.distance_matrix(pos[anchor_idx], pos[query_idx])
    # print("min distance", np.min(mat))

    # import trimesh
    # axis = trimesh.creation.axis(axis_length=1)
    # npts = pos.shape[0]
    # colors = np.ones((npts, 4))
    # colors[query_idx] = np.array([[1, 0, 0, 1]])
    # pcd = trimesh.PointCloud(pos, colors=colors)
    # (trimesh.Scene(pcd) + axis).show()

    # import ipdb; ipdb.set_trace()

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender, np.ones(count_neighbors)], axis=1))

    return relations


def make_hierarchy(env, attr, positions, velocities, idx, st, ed, phases_dict, count_nodes, clusters, verbose=0, var=False):
    order = 2
    n_root_level = len(phases_dict["root_num"][idx])
    attr, relations, relations_types, node_r_idx, node_s_idx, pstep = [attr], [], [], [], [], []

    relations_rev, relations_rev_types, node_r_idx_rev, node_s_idx_rev, pstep_rev = [], [], [], [], []

    pos = positions.data.cpu().numpy() if var else positions
    vel = velocities.data.cpu().numpy() if var else velocities

    for i in range(n_root_level):
        root_num = phases_dict["root_num"][idx][i]
        #root_sib_radius = phases_dict["root_sib_radius"][idx][i]
        root_des_radius = phases_dict["root_des_radius"][idx][i]
        root_pstep = phases_dict["root_pstep"][idx][i]

        if verbose:
            print('root info', root_num, root_sib_radius, root_des_radius, root_pstep)


        ### clustring the nodes
        # st_time = time.time()
        # kmeans = MiniBatchKMeans(n_clusters=root_num, random_state=0).fit(pos[st:ed, 3:6])
        # print('Time on kmeans', time.time() - st_time)
        # clusters = kmeans.labels_
        # centers = kmeans.cluster_centers_


        ### relations between roots and desendants
        rels, rels_rev = [], []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(st, ed))
        node_r_idx_rev.append(node_s_idx[-1])
        node_s_idx_rev.append(node_r_idx[-1])
        pstep.append(1); pstep_rev.append(1)

        if verbose:
            centers = np.zeros((root_num, 3))
            # compute the mean of each sub-parts
            for j in range(root_num):
                des = np.nonzero(clusters[i][0]==j)[0] #indices inside the group
                center = np.mean(pos[st:ed][des, -3:], 0, keepdims=True)
                centers[j] = center[0]
                visualize_neighbors(pos[st:ed], center, 0, des)

        for j in range(root_num):
            desendants = np.nonzero(clusters[i][0]==j)[0]
            roots = np.ones(desendants.shape[0]) * j
            if verbose:
                print(roots, desendants)
            rels += [np.stack([roots, desendants, np.zeros(desendants.shape[0])], axis=1)]
            rels_rev += [np.stack([desendants, roots, np.zeros(desendants.shape[0])], axis=1)]
            if verbose:
                print(np.max(np.sqrt(np.sum(np.square(pos[st + desendants, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))
        relations_rev.append(np.concatenate(rels_rev, 0))
        relations_types.append("leaf-root")
        relations_rev_types.append("root-leaf")


        ### relations between roots and roots
        # point_tree = spatial.cKDTree(centers)
        # neighbors = point_tree.query_ball_point(centers, root_sib_radius, p=order)

        '''
        for j in range(len(neighbors)):
            visualize_neighbors(centers, centers, j, neighbors[j])
        '''

        rels = []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(count_nodes, count_nodes + root_num))
        pstep.append(root_pstep)

        # adding all possible pairs of root nodes
        roots = np.repeat(np.arange(root_num), root_num)
        siblings = np.tile(np.arange(root_num), root_num)
        if verbose:
            print(roots, siblings)
        rels += [np.stack([roots, siblings, np.zeros(root_num * root_num)], axis=1)]
        if verbose:
            print(np.max(np.sqrt(np.sum(np.square(centers[siblings, :3] - centers[j]), 1))))


        relations.append(np.concatenate(rels, 0))
        relations_types.append("root-root")


        ### add to attributes/positions/velocities
        positions = [positions]
        velocities = [velocities]
        attributes = []
        for j in range(root_num):
            ids = np.nonzero(clusters[i][0]==j)[0]
            if var:
                positions += [torch.mean(positions[0][st:ed, :][ids], 0, keepdim=True)]
                velocities += [torch.mean(velocities[0][st:ed, :][ids], 0, keepdim=True)]
            else:
                positions += [np.mean(positions[0][st:ed, :][ids], 0, keepdims=True)]
                velocities += [np.mean(velocities[0][st:ed, :][ids], 0, keepdims=True)]

            attributes += [np.mean(attr[0][st:ed, :][ids], 0, keepdims=True)]

        attributes = np.concatenate(attributes, 0)

        if not attributes[0, -1] == 0:
            import ipdb; ipdb.set_trace()
        assert(attributes[0, -1] == 0), "last dimension should save for parent node"
        attributes[0, -1] = 1


        if verbose:
            print('Attr sum', np.sum(attributes, 0))

        attr += [attributes]
        if var:
            positions = torch.cat(positions, 0)
            velocities = torch.cat(velocities, 0)
        else:
            positions = np.concatenate(positions, 0)
            velocities = np.concatenate(velocities, 0)

        # add #[root_num] of root nodes
        st = count_nodes
        ed = count_nodes + root_num
        count_nodes += root_num

        if verbose:
            print(st, ed, count_nodes, positions.shape, velocities.shape)

    attr = np.concatenate(attr, 0)
    if verbose:
        print("attr", attr.shape)

    relations += relations_rev[::-1]
    relations_types += relations_rev_types[::-1]

    node_r_idx += node_r_idx_rev[::-1]
    node_s_idx += node_s_idx_rev[::-1]
    pstep += pstep_rev[::-1]

    return attr, positions, velocities, count_nodes, relations, relations_types, node_r_idx, node_s_idx, pstep


def prepare_input(data, stat, args, phases_dict, verbose=0, var=False):

    # Arrangement:
    # particles, shapes, roots
    if args.env == "TDWdominoes":
        # attributes:
        # [1, 0, 0]: water particle
        # [0, 1, 0]: rigid particle
        # [0, 0, 1]: whole floor node
        # [x, x, 1]: parent node
        positions, velocities = data
        clusters = phases_dict["clusters"]
        n_shapes = 0

        if args.floor_cheat:
            # add the floor node
            n_shapes = 1
            positions = np.concatenate([positions, np.zeros((1,3))], axis=0)
            velocities = np.concatenate([velocities, np.zeros((1,3))], axis=0)

    else:
        raise ValueError


    count_nodes = positions.size(0) if var else positions.shape[0]
    n_particles = count_nodes - n_shapes


    if verbose:
        print("positions", positions.shape)
        print("velocities", velocities.shape)

        print("n_particles", n_particles)
        print("n_shapes", n_shapes)

    ### instance idx
    #   instance_idx (n_instance + 1): start idx of instance
    instance_idx = phases_dict["instance_idx"]
    if verbose:
        print("Instance_idx:", instance_idx)

    ### object attributes
    #   dim 10: [rigid, fluid, root_0, root_1, gripper_0, gripper_1, mass_inv,
    #            clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep]
    attr = np.zeros((count_nodes, args.attr_dim))

    ### construct relations
    Rr_idxs = []        # relation receiver idx list
    Rs_idxs = []        # relation sender idx list
    Ras = []            # relation attributes list
    values = []         # relation value list (should be 1)
    node_r_idxs = []    # list of corresponding receiver node idx
    node_s_idxs = []    # list of corresponding sender node idx
    psteps = []         # propagation steps

    ##### add env specific graph components
    rels = []
    rels_types = []
    if args.env == "TDWdominoes" and args.floor_cheat:

        pos = positions.data.cpu().numpy() if var else positions
        dis = pos[:n_particles, 1] - 0

        nodes = np.nonzero(dis < 0.1)[0]
        attr[-1, 2] = 1 # [0, 0, 1] is floor
        floor = np.ones(nodes.shape[0], dtype=int) * (n_particles + 0) #0 for idx starting from zero
        rels += [np.stack([nodes, floor, np.ones(nodes.shape[0])], axis=1)]

    if verbose and len(rels) > 0:
        print(np.concatenate(rels, 0).shape)

    if args.env == "TDWdominoes":
        nobjs = len(instance_idx) - 1
        phases_dict["root_pstep"] = [[args.pstep]]*nobjs

    ##### add relations between leaf particles

    if args.model_name in ["GNS", "GNSRigid", "GNSRigidH"]:
        # for model without hierarchy, just search particles that are close

        queries = np.arange(n_particles)
        anchors = np.arange(n_particles)
        pos = positions
        pos = pos[:, -3:]
        rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
    else:
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]

            if verbose:
                print('instance #%d' % i, st, ed)

            if args.env == "TDWdominoes":
                if phases_dict['material'][i] == 'rigid':
                    attr[st:ed, 0] = 1
                    queries = np.arange(st, ed)
                    anchors = np.concatenate((np.arange(st), np.arange(ed, n_particles)))
                elif phases_dict['material'][i] == 'fluid' or phases_dict['material'][i] == 'cloth' :
                    attr[st:ed, 1] = 1
                    queries = np.arange(st, ed)
                    anchors = np.arange(n_particles)
                else:
                    raise AssertionError("Unsupported materials")

            else:
                raise AssertionError("Unsupported materials")

            # st_time = time.time()
            pos = positions
            pos = pos[:, -3:]
            rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
            # print("Time on neighbor search", time.time() - st_time)

    if verbose:
        print("Attr shape (after add env specific graph components):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))

    if len(rels) > 0:
        rels = np.concatenate(rels, 0)
        if rels.shape[0] > 0:
            if verbose:
                print("Relations neighbor", rels.shape)
            n_rels = rels.shape[0]
            rel_indices = torch.arange(n_rels, dtype=torch.long)
            Rr_idxs.append(torch.stack([
                torch.from_numpy(rels[:, 0].astype(np.int64)),
                rel_indices
            ]))
            Rs_idxs.append(torch.stack([
                torch.from_numpy(rels[:, 1].astype(np.int64)),
                rel_indices
            ]))
            Ras.append(torch.zeros(n_rels, args.relation_dim, dtype=torch.float32))
            values.append(torch.ones(n_rels, dtype=torch.float32))
            node_r_idxs.append(np.arange(n_particles))
            node_s_idxs.append(np.arange(n_particles + n_shapes))
            psteps.append(args.pstep)
            rels_types.append("leaf-leaf")

    if verbose:
        print('clusters', clusters)

    # add heirarchical relations per instance
    cnt_clusters = 0
    # clusters: [[[ array(#num_nodes_in_instance) ]]*n_root_level   ]*num_clusters

    if args.model_name not in ["GNS", "GNSRigid"]: #GNS has no hierarchy
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            n_root_level = len(phases_dict["root_num"][i])

            if n_root_level > 0:
                attr, positions, velocities, count_nodes, \
                rels, rels_type, node_r_idx, node_s_idx, pstep = \
                        make_hierarchy(args.env, attr, positions, velocities, i, st, ed,
                                       phases_dict, count_nodes, clusters[cnt_clusters], verbose, var)

                for j in range(len(rels)):
                    if verbose:
                        print("Relation instance", j, rels[j].shape)
                    n_rels_j = rels[j].shape[0]
                    rel_indices_j = torch.arange(n_rels_j, dtype=torch.long)
                    Rr_idxs.append(torch.stack([
                        torch.from_numpy(rels[j][:, 0].astype(np.int64)),
                        rel_indices_j
                    ]))
                    Rs_idxs.append(torch.stack([
                        torch.from_numpy(rels[j][:, 1].astype(np.int64)),
                        rel_indices_j
                    ]))
                    Ra_tensor = torch.zeros(n_rels_j, args.relation_dim, dtype=torch.float32)
                    Ra_tensor[:, 0] = 1
                    Ras.append(Ra_tensor)
                    values.append(torch.ones(n_rels_j, dtype=torch.float32))
                    node_r_idxs.append(node_r_idx[j])
                    node_s_idxs.append(node_s_idx[j])
                    psteps.append(pstep[j])
                    rels_types.append(rels_type[j])

                cnt_clusters += 1

    if verbose:

        print("Attr shape (after hierarchy building):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particles], axis=0))
        print("Shape attr:", np.sum(attr[n_particles:n_particles+n_shapes], axis=0))
        print("Roots attr:", np.sum(attr[n_particles+n_shapes:], axis=0))

    ### normalize data
    data = [positions, velocities]
    data_a = normalize(data, stat, var)
    positions, velocities = data[0], data[1]

    if verbose:
        print("Particle positions stats")
        print(positions.shape)
        print(np.min(positions[:n_particles], 0))
        print(np.max(positions[:n_particles], 0))
        print(np.mean(positions[:n_particles], 0))
        print(np.std(positions[:n_particles], 0))

        show_vel_dim = 3
        print("Velocities stats")
        print(velocities.shape)
        print(np.mean(velocities[:n_particles, :show_vel_dim], 0))
        print(np.std(velocities[:n_particles, :show_vel_dim], 0))


    if var:
        state = torch.cat([positions, velocities], 1)
    else:
        state = torch.FloatTensor(np.concatenate([positions, velocities], axis=1))

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

        for i in range(len(Ras)):
            print(i, np.min(node_r_idxs[i]), np.max(node_r_idxs[i]), np.min(node_s_idxs[i]), np.max(node_s_idxs[i]))

    attr = torch.FloatTensor(attr)
    relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps, rels_types]

    return attr, state, relations, n_particles, n_shapes, instance_idx


class PhysicsFleXDataset(Dataset):

    def __init__(self, args, phase, phases_dict, verbose):
        self.args = args
        self.phase = phase
        self.phases_dict = phases_dict
        self.verbose = verbose


        self.augment_coord = self.args.augment_worldcoord

        assert(isinstance(self.args.dataf, list))
        self.data_dir = [os.path.join(dataf, phase) for dataf in self.args.dataf]

        if self.args.statf:
            self.stat_path = os.path.join(self.args.data_root, self.args.statf)
        else:
            self.stat_path = None

        if args.env == "TDWdominoes":
            self.data_names = ['positions', 'velocities']
        else:
            raise ValueError
        self.training_fpt = self.args.training_fpt
        self.dt = self.training_fpt * self.args.dt
        self.start_timestep = int(15 * self.training_fpt)

        if self.args.n_rollout == None:
            self.all_trials = []
            self.n_rollout = 0


            for ddir in self.data_dir:
                file = open(ddir +  ".txt", "r")
                ddir_root = "/".join(ddir.split("/")[:-1])
                trial_names = [line.strip("\n") for line in file if line != "\n"]
                n_trials = len(trial_names)

                self.all_trials += [os.path.join(ddir_root, trial_name) for trial_name in trial_names]
                self.n_rollout += n_trials

            if phase == "train":
                self.mean_time_step = int(13499/self.n_rollout) + 1
            else:
                self.mean_time_step = 1
        else:
            ratio = self.args.train_valid_ratio
            if phase == 'train':
                self.n_rollout = int(self.args.n_rollout * ratio)
            elif phase == 'valid':
                self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
            else:
                raise AssertionError("Unknown phase")

    def __len__(self):
        if self.args.env == "TDWdominoes":
            # each rollout can have different length, sample length in get_item
            return self.n_rollout * self.mean_time_step
        else:
            return self.n_rollout * (self.args.time_step - 1)

    def load_data(self, name):
        # load the global statistics of "position" and "velocities"

        if self.stat_path is not None:
            self.stat = load_data(self.data_names[:2], self.stat_path)
            for i in range(len(self.stat)):
                self.stat[i] = self.stat[i][-self.args.position_dim:, :]
        else:
            positions_stat = np.zeros((3,3))
            velocities_stat= np.zeros((3,3))

            self.stat = [positions_stat, velocities_stat]

    def augment_worldcoord(self, list_of_data):
        """
        input:
            list_of_data: list of data to transform
        output:
            return the position, velocities for (len(list_of_data)) data
        """
        # sample rotation and translation
        import math
        import copy

        azimuth = np.random.uniform(0,360)
        elevation = 0
        # rotation along y axis
        rot = np.array([[np.cos(math.radians(azimuth)), 0 , -np.sin(math.radians(azimuth))],
                  [0 , 1 ,0 ],
                  [np.sin(math.radians(azimuth)), 0, np.cos(math.radians(azimuth))]])
        T = np.array([np.random.uniform(-1.0, 1.0), 0, np.random.uniform(-1.0, 1.0)])
        RT = np.eye(4)
        RT[:3,:3] = rot
        MatT = np.eye(4)
        MatT[:3, 3] = T
        RT = np.matmul(RT, MatT) # translate then rotate
        R = copy.deepcopy(RT)
        R[:3, 3] = 0

        positions_over_T = []
        velocities_over_T = []

        for data in list_of_data:

            positions = data[self.data_names.index("positions")]
            velocities = data[self.data_names.index("velocities")]

            # import trimesh
            # axis = trimesh.creation.axis(axis_length=1)
            # pcd = trimesh.PointCloud(positions)
            # (trimesh.Scene(pcd) + axis).show()
            ## plot the two
            transformed_positions = np.matmul(RT, np.concatenate([positions, np.ones((positions.shape[0], 1))], axis=1).T).T[:,:3]
            transformed_velocities = np.matmul(R, np.concatenate([velocities, np.ones((velocities.shape[0], 1))], axis=1).T).T[:,:3]

            # import trimesh
            # axis = trimesh.creation.axis(axis_length=1)
            # pcd = trimesh.PointCloud(transformed_positions)
            # (trimesh.Scene(pcd) + axis).show()

            positions_over_T.append(transformed_positions)
            velocities_over_T.append(transformed_velocities)

        output_list_of_data = []
        for t in range(len(list_of_data)):
            current_data = []
            for item in self.data_names:
                if item == "positions":
                    current_data.append(positions_over_T[t])
                elif item == "velocities":
                    current_data.append(velocities_over_T[t])
                else:
                    raise (ValueError, f"not supporting augmentation for {item}")
            output_list_of_data.append(current_data)

        return output_list_of_data




    def __getitem__(self, idx):

        if self.args.env == "TDWdominoes":
            idx = idx % self.n_rollout
            trial_dir = self.all_trials[idx]
            #print(self.args.outf.split("/")[-1], trial_dir)

            data_dir = "/".join(trial_dir.split("/")[:-1])

            trial_fullname = trial_dir.split("/")[-1]

            pkl_path = os.path.join(trial_dir, 'phases_dict.pkl')
            with open(pkl_path, "rb") as f:
                phases_dict = pickle.load(f)
            phases_dict["trial_dir"] = trial_dir

            correct_bad_chair(phases_dict)
            # remove obstacles that are too big
            remove_large_obstacles(phases_dict)

            #if self.args.subsample_particles:
            is_subsample = subsample_particles_on_large_objects(phases_dict, limit=3000)

            # because we want to sample also idx_timestep + 1
            time_step = phases_dict["time_step"] - self.training_fpt
            idx_timestep = np.random.randint(self.start_timestep, time_step)

            #idx_timestep = 100

        else:
            raise ValueError

        data_path = os.path.join(data_dir, trial_fullname, str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(data_dir, trial_fullname, str(int(idx_timestep + self.training_fpt)) + '.h5')
        current_oldest = 0
        data = load_data_dominoes(self.data_names, data_path, phases_dict)
        data_nxt = load_data_dominoes(self.data_names, data_nxt_path, phases_dict)

        data_prev_path = os.path.join(data_dir, trial_fullname, str(max(0, int(idx_timestep - self.training_fpt))) + '.h5')
        data_prev = load_data_dominoes(self.data_names, data_prev_path, phases_dict)
        _, data, data_nxt = recalculate_velocities([data_prev, data, data_nxt], self.dt, self.data_names)


        # adding history
        data_hiss = []
        path = os.path.join(data_dir, trial_fullname, str(max(0, int(idx_timestep - (1) * self.training_fpt))) + '.h5')
        data_cur = load_data_dominoes(self.data_names, path, phases_dict)
        for i in range(self.args.n_his):
            path = os.path.join(data_dir, trial_fullname, str(max(0, int(idx_timestep - (i + 1 + 1) * self.training_fpt))) + '.h5')
            if self.args.env == "TDWdominoes":
               data_prev = load_data_dominoes(self.data_names, path, phases_dict)
               _, data_his = recalculate_velocities([data_prev, data_cur], self.dt, self.data_names)
               data_cur = data_prev
            else:
               data_his = load_data(self.data_names, path)
            data_hiss.append(data_his)
            current_oldest = - (i + 1) * self.training_fpt


        # calculate the velocities if training_timestep is not 1
        # augment
        if self.augment_coord or self.training_fpt > 1 or self.args.noise_std > 0:
            assert(self.args.env == "TDWdominoes") # haven't checked for other tasks
            # load one more in the back so we can compute the velocities

            path = os.path.join(data_dir, trial_fullname, str(max(0, int(idx_timestep - current_oldest - self.training_fpt))) + '.h5')
            data_append = load_data_dominoes(self.data_names, path, phases_dict)
            data_seq = [data_append] + data_hiss + [data, data_nxt]

        if self.training_fpt > 1:
            #recalculate the velocities
            data_seq = recalculate_velocities(data_seq, self.dt, self.data_names)

        if self.args.noise_std > 0:
            n_velsteps = len(data_seq) - 1 # remove the nxt step and the first one
            vel_dim = data_seq[1][1].shape[1]
            vel_sequence_noise = np.random.normal(0, self.args.noise_std/ (n_velsteps**0.5), size=(n_velsteps, vel_dim))

            for t in range(1, n_velsteps):
                # add noise to velocities
                data_seq[t][1] = data_seq[t][1] + vel_sequence_noise[t]
                # update position
                data_seq[t][0] = data_seq[t-1][0] + data_seq[t][1] * self.dt

        if self.augment_coord:
            data_seq = self.augment_worldcoord(data_seq[1:])
        else:
            data_seq = data_seq[1:]
        data = data_seq[-2]
        data_nxt = data_seq[-1]
        data_hiss = data_seq[:-2]


        vel_his = []
        for i in range(self.args.n_his):

            data_his = data_hiss[i]
            vel_his.append(data_his[1])

        data[1] = np.concatenate([data[1]] + vel_his, 1)

        attr, state, relations, n_particles, n_shapes, instance_idx = \
                prepare_input(data, self.stat, self.args, phases_dict, self.verbose)

        ### normalized velocities
        data_nxt = normalize(data_nxt, self.stat)
        label = torch.FloatTensor(data_nxt[1][:n_particles])


        return attr, state, relations, n_particles, n_shapes, instance_idx, label, phases_dict

