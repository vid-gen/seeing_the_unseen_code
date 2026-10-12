"""
地面 Mesh 生成模块

从 phases_dict.pkl 和 coord_transform.json 读取地面粒子信息，
生成一个 NxN 网格平面 mesh 并保存为 PLY 文件 (ground_mesh_posed.ply)。

地面 mesh 保存在 scene_3dgs/ 目录中，与 object_*_mesh_posed.ply 并列。
mesh_wireframe_vis.py 和 mesh_to_tracking.py 可以直接加载使用。

用法:
  # 作为独立脚本
  python generate_ground_mesh.py --scene_dir /path/to/scene

  # 作为模块导入
  from generate_ground_mesh import generate_ground_mesh
  generate_ground_mesh(scene_dir='/path/to/scene')
"""

import os
import json
import pickle
import argparse
import numpy as np
import trimesh


def _find_ground_in_phases_dict(phases_dict):
    """在 phases_dict 中找到地面物体 (名称为 'cube')。

    Returns:
        ground_idx: 地面物体的索引
        ground_local: (M, 3) 地面粒子局部坐标
    """
    names = []
    for name in phases_dict.get('instance', []):
        if isinstance(name, bytes):
            name = name.decode()
        names.append(name)

    # 地面在 build_physion_input.py 中以 'cube' 命名
    ground_idx = None
    for i, name in enumerate(names):
        if name == 'cube':
            ground_idx = i
            break

    if ground_idx is None:
        return None, None

    ground_local = phases_dict['obj_points'][ground_idx]
    return ground_idx, ground_local


def _generate_grid_mesh(x_min, x_max, z_min, z_max, y, grid_n=20):
    """生成 NxN 网格平面 mesh。

    Args:
        x_min, x_max, z_min, z_max: XZ 平面范围
        y: Y 高度
        grid_n: 网格划分数 (NxN)

    Returns:
        vertices: (V, 3) float64
        faces: (F, 3) int64
    """
    xs = np.linspace(x_min, x_max, grid_n + 1)
    zs = np.linspace(z_min, z_max, grid_n + 1)
    xx, zz = np.meshgrid(xs, zs)
    yy = np.full_like(xx, y)

    # 顶点: (grid_n+1) x (grid_n+1)
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # 面片: 每个网格格子 2 个三角面片
    faces = []
    n_cols = grid_n + 1
    for row in range(grid_n):
        for col in range(grid_n):
            # 当前格子的 4 个顶点索引
            v00 = row * n_cols + col
            v10 = row * n_cols + (col + 1)
            v01 = (row + 1) * n_cols + col
            v11 = (row + 1) * n_cols + (col + 1)
            # 两个三角面片 (逆时针缠绕，法线朝 +Y)
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])

    vertices = vertices.astype(np.float64)
    faces = np.array(faces, dtype=np.int64)

    return vertices, faces


def generate_ground_mesh(scene_dir, output_path=None, grid_n=20):
    """从 phases_dict 和 coord_transform 生成地面 mesh PLY。

    Args:
        scene_dir: 场景目录 (包含 trial/ 和 scene_3dgs/ 子目录)
        output_path: 输出 PLY 路径，默认 scene_3dgs/ground_mesh_posed.ply
        grid_n: 网格划分数 (NxN)，默认 20

    Returns:
        output_path: 生成的 PLY 文件路径，失败返回 None
    """
    trial_dir = os.path.join(scene_dir, 'trial')
    scene_3dgs_dir = os.path.join(scene_dir, 'scene_3dgs')

    phases_dict_path = os.path.join(trial_dir, 'phases_dict.pkl')
    coord_transform_path = os.path.join(trial_dir, 'coord_transform.json')

    if not os.path.exists(phases_dict_path):
        print(f"  警告: phases_dict.pkl 不存在: {phases_dict_path}")
        return None
    if not os.path.exists(coord_transform_path):
        print(f"  警告: coord_transform.json 不存在: {coord_transform_path}")
        return None

    # 1. 加载 phases_dict，找地面
    with open(phases_dict_path, 'rb') as f:
        pd = pickle.load(f)

    ground_idx, ground_local = _find_ground_in_phases_dict(pd)
    if ground_idx is None:
        print("  警告: phases_dict 中未找到地面物体 ('cube')，跳过")
        return None

    # 2. 获取地面粒子的 XZ 范围和 Y 高度 (局部坐标)
    x_min_local = ground_local[:, 0].min()
    x_max_local = ground_local[:, 0].max()
    z_min_local = ground_local[:, 2].min()
    z_max_local = ground_local[:, 2].max()
    y_local = ground_local[0, 1]  # 所有粒子 Y 相同

    # 3. 从 rollout 获取地面 centroid，还原世界坐标
    #    rollout 中地面粒子 = ground_local + centroid
    #    但 rollout 存储的是所有粒子的世界坐标
    #    我们直接从 instance_idx 提取地面粒子的世界坐标范围
    rollout_path = os.path.join(scene_dir, 'rollout', 'rollout_positions.npz')
    if os.path.exists(rollout_path):
        raw = np.load(rollout_path)
        step0 = raw['step_0']
        inst_idx = pd['instance_idx']
        ground_st = inst_idx[ground_idx]
        ground_ed = inst_idx[ground_idx + 1]
        ground_world = step0[ground_st:ground_ed]

        x_min_world = ground_world[:, 0].min()
        x_max_world = ground_world[:, 0].max()
        z_min_world = ground_world[:, 2].min()
        z_max_world = ground_world[:, 2].max()
        y_world = ground_world[0, 1]
    else:
        print("  警告: rollout_positions.npz 不存在, 使用局部坐标估算")
        x_min_world = x_min_local
        x_max_world = x_max_local
        z_min_world = z_min_local
        z_max_world = z_max_local
        y_world = y_local

    # 4. 反变换到 scene 空间 (与 object_*_mesh_posed.ply 同空间)
    with open(coord_transform_path, 'r') as f:
        ct = json.load(f)
    offset = np.array([ct['center_x'], ct['ground_y'], ct['center_z']])
    scale = ct['scale'] * ct.get('scene_scale_factor', 1.0)
    gc = ct.get('ground_clearance', 0.0)

    # 正变换 (scene → physion): p_physion = (p_scene - offset) * scale; p_physion[:, 1] += gc
    # 反变换 (physion → scene): p_physion[:, 1] -= gc; p_scene = p_physion / scale + offset

    def physion_to_scene(pts):
        pts = pts.copy()
        pts[:, 1] -= gc
        return pts / scale + offset

    corners_physion = np.array([
        [x_min_world, y_world, z_min_world],
        [x_max_world, y_world, z_min_world],
        [x_max_world, y_world, z_max_world],
        [x_min_world, y_world, z_max_world],
    ])
    corners_scene = physion_to_scene(corners_physion)

    x_min_s = corners_scene[:, 0].min()
    x_max_s = corners_scene[:, 0].max()
    z_min_s = corners_scene[:, 2].min()
    z_max_s = corners_scene[:, 2].max()
    y_s = corners_scene[0, 1]

    # 5. 生成 NxN 网格 mesh (scene 空间)
    vertices, faces = _generate_grid_mesh(
        x_min_s, x_max_s, z_min_s, z_max_s, y_s, grid_n=grid_n)

    print(f"  地面 mesh: {len(vertices)} 顶点, {len(faces)} 面片 "
          f"({grid_n}x{grid_n} 网格)")
    print(f"    Physion XZ: [{x_min_world:.3f}, {x_max_world:.3f}] x "
          f"[{z_min_world:.3f}, {z_max_world:.3f}], Y={y_world:.4f}")
    print(f"    Scene   XZ: [{x_min_s:.3f}, {x_max_s:.3f}] x "
          f"[{z_min_s:.3f}, {z_max_s:.3f}], Y={y_s:.4f}")

    # 6. 保存为 PLY
    if output_path is None:
        output_path = os.path.join(scene_3dgs_dir, 'ground_mesh_posed.ply')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path)
    print(f"  保存地面 mesh: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='生成地面 Mesh PLY')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='场景目录 (包含 trial/, scene_3dgs/, rollout/)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出 PLY 路径 (默认: scene_3dgs/ground_mesh_posed.ply)')
    parser.add_argument('--grid_n', type=int, default=20,
                        help='网格划分数 NxN (默认: 20)')
    args = parser.parse_args()

    generate_ground_mesh(
        scene_dir=args.scene_dir,
        output_path=args.output_path,
        grid_n=args.grid_n,
    )


if __name__ == '__main__':
    main()
