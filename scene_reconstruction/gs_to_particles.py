"""
gs_to_particles.py - 将 3DGS 输出转换为 Physion 格式的粒子点云

功能:
- 从 auto_scene_pipeline.py 输出的 3DGS PLY 文件提取高斯中心
- 过滤低不透明度高斯
- 体素下采样为均匀粒子
- 坐标系对齐到 Physion 仿真坐标系 (Y-up, 地面在 Y≈0)
"""

import os
import json
import argparse
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


def load_gaussian_ply(ply_path):
    """
    从 3DGS PLY 文件加载高斯属性

    Returns:
        dict with keys: xyz, opacity, scale, rotation, f_dc
    """
    if PlyData is None:
        raise ImportError("需要安装 plyfile: pip install plyfile")

    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    n = len(vertex)

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)

    # opacity: PLY 中存储的是 inverse_sigmoid(opacity), 需要 sigmoid 还原
    if 'opacity' in vertex.data.dtype.names:
        raw_opacity = np.array(vertex['opacity'], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))  # sigmoid
    else:
        opacity = np.ones(n, dtype=np.float32)

    # scale: PLY 中存储的是 log(scale), 需要 exp 还原
    if 'scale_0' in vertex.data.dtype.names:
        scale = np.stack([
            np.exp(np.array(vertex['scale_0'], dtype=np.float32)),
            np.exp(np.array(vertex['scale_1'], dtype=np.float32)),
            np.exp(np.array(vertex['scale_2'], dtype=np.float32)),
        ], axis=1)
    else:
        scale = None

    # SH DC coefficients (颜色)
    if 'f_dc_0' in vertex.data.dtype.names:
        f_dc = np.stack([
            np.array(vertex['f_dc_0'], dtype=np.float32),
            np.array(vertex['f_dc_1'], dtype=np.float32),
            np.array(vertex['f_dc_2'], dtype=np.float32),
        ], axis=1)
    else:
        f_dc = None

    return {
        'xyz': xyz,
        'opacity': opacity,
        'scale': scale,
        'f_dc': f_dc,
        'n_gaussians': n,
    }


def filter_by_opacity(gs_data, threshold=0.5):
    """过滤低不透明度的高斯"""
    mask = gs_data['opacity'] > threshold
    filtered = {
        'xyz': gs_data['xyz'][mask],
        'opacity': gs_data['opacity'][mask],
        'n_gaussians': int(mask.sum()),
    }
    if gs_data['scale'] is not None:
        filtered['scale'] = gs_data['scale'][mask]
    if gs_data.get('f_dc') is not None:
        filtered['f_dc'] = gs_data['f_dc'][mask]
    return filtered


def voxel_downsample(points, voxel_size=0.05):
    """
    体素网格下采样, 使粒子均匀分布

    对每个体素格子, 取其中所有点的均值作为代表点
    """
    if len(points) == 0:
        return points

    # 计算体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # 对每个唯一体素取均值
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    n_voxels = len(unique_voxels)
    downsampled = np.zeros((n_voxels, 3), dtype=np.float32)
    counts = np.zeros(n_voxels, dtype=np.int32)

    for i in range(len(points)):
        downsampled[inverse[i]] += points[i]
        counts[inverse[i]] += 1

    downsampled /= counts[:, np.newaxis]
    return downsampled


def _voxelize_and_fill(points, voxel_size):
    """
    单遍体素化 + binary_fill_holes 内部填充

    Returns:
        solid_centers: (M, 3) 实心体素中心坐标
        n_surface: 表面体素数
        n_interior: 内部填充体素数
    """
    from scipy.ndimage import binary_fill_holes

    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    min_idx = voxel_indices.min(axis=0)
    voxel_indices -= min_idx

    max_idx = voxel_indices.max(axis=0)
    grid_shape = tuple(max_idx + 3)  # +1 for 0-index, +2 for padding
    voxel_indices += 1  # 偏移 1 格, 留出边界

    unique_vi = np.unique(voxel_indices, axis=0)
    n_surface = len(unique_vi)

    grid = np.zeros(grid_shape, dtype=bool)
    grid[unique_vi[:, 0], unique_vi[:, 1], unique_vi[:, 2]] = True

    filled = binary_fill_holes(grid)

    solid_indices = np.argwhere(filled)
    solid_indices = solid_indices - 1 + min_idx
    solid_centers = (solid_indices.astype(np.float32) + 0.5) * voxel_size

    n_interior = len(solid_centers) - n_surface
    return solid_centers, n_surface, n_interior


def convex_hull_fill(points, voxel_size=0.05):
    """
    凸包内部填充 (兜底方案)
    当 binary_fill_holes 失败时使用

    算法:
    1. 计算物体凸包
    2. 在凸包 bbox 内生成 spacing=voxel_size 的规则网格
    3. 只保留凸包内部的网格点
    4. 合并原始表面点 + 内部点, 去重
    """
    from scipy.spatial import ConvexHull, Delaunay

    if len(points) < 4:
        return points

    try:
        hull = ConvexHull(points)
        delaunay = Delaunay(points[hull.vertices])
    except Exception as e:
        print(f"    凸包计算失败 ({e}), 保持原样")
        return points

    # 在凸包 bbox 内生成规则网格
    min_pt = points.min(axis=0) - voxel_size
    max_pt = points.max(axis=0) + voxel_size
    grid_x = np.arange(min_pt[0], max_pt[0], voxel_size)
    grid_y = np.arange(min_pt[1], max_pt[1], voxel_size)
    grid_z = np.arange(min_pt[2], max_pt[2], voxel_size)
    grid = np.stack(np.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), axis=-1)
    grid = grid.reshape(-1, 3).astype(np.float32)

    # 只保留凸包内部的点
    inside = delaunay.find_simplex(grid) >= 0
    interior_pts = grid[inside]

    n_interior = len(interior_pts)
    print(f"    凸包填充: 凸包顶点={len(hull.vertices)}, 内部网格点={n_interior}")

    # 合并原始表面点 + 内部点, 去重
    merged = np.concatenate([points, interior_pts], axis=0)
    return voxel_downsample(merged, voxel_size)


def densify_thin_object(coarse_particles, fine_spacing=0.025):
    """
    Grid-based 薄物体增密 (经验证方案, 来自 test_dense_car_rollout.py)

    在粗粒子的 bounding box 内生成 fine_spacing 间距的规则网格,
    通过 kd-tree 距离过滤保持物体形状。

    此方法已在汽车场景验证: 51→252 粒子, 位移 0.211→3.366 (16x 改善)

    Args:
        coarse_particles: (N, 3) 粗间距粒子 (spacing=0.05)
        fine_spacing: 目标细间距 (默认 0.025)

    Returns:
        (M, 3) 增密后的粒子
    """
    from scipy.spatial import cKDTree

    min_xyz = coarse_particles.min(axis=0).copy()
    max_xyz = coarse_particles.max(axis=0).copy()
    ranges = max_xyz - min_xyz

    # 对非常薄的维度稍微扩展 (确保至少 2 层)
    for d in range(3):
        if ranges[d] < fine_spacing * 2:
            min_xyz[d] -= fine_spacing
            max_xyz[d] += fine_spacing

    # 生成规则网格
    grids = []
    for d in range(3):
        n = max(2, int((max_xyz[d] - min_xyz[d]) / fine_spacing) + 1)
        grids.append(np.linspace(float(min_xyz[d]), float(max_xyz[d]), n))

    xx, yy, zz = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    # 只保留距离粗粒子 ≤ 2.5×fine_spacing 的网格点 (保持形状)
    tree = cKDTree(coarse_particles)
    dists, _ = tree.query(grid_points)
    mask = dists <= fine_spacing * 2.5
    dense_points = grid_points[mask]

    # 保持质心不偏移
    old_center = coarse_particles.mean(axis=0)
    new_center = dense_points.mean(axis=0)
    dense_points = dense_points - new_center + old_center

    return dense_points.astype(np.float32)


def voxel_fill_interior(points, voxel_size=0.05, fine_factor=2, use_convex_hull=True):
    """
    对表面点云做体素化 + 内部填充, 生成实心粒子

    使用"先细后粗"两遍策略:
    Pass 1: 用 fine_spacing (voxel_size/fine_factor) 体素化+填充
            → 更密的表面壳 → binary_fill_holes 封闭成功率更高
    Pass 2: 下采样到 voxel_size → 匹配训练数据间距

    特殊处理:
    - 薄物体 (min_extent < 3*voxel_size): 使用 grid-based 增密
      先下采样到 voxel_size 获取粗粒子, 再用 densify_thin_object() 在
      粗粒子附近生成 fine_spacing 规则网格 (经 test_dense_car_rollout.py 验证)
    - 空壳物体 (fill_ratio < 0.3 且足够大): 回退到凸包填充

    这模拟了 Physion 中 binvox -aw 的体素化效果 (实心填充)
    """
    if len(points) == 0:
        return points

    fine_spacing = voxel_size / fine_factor

    # ---- Pass 1: 细网格体素化 + 内部填充 ----
    fine_particles, n_fine_surface, n_fine_interior = _voxelize_and_fill(points, fine_spacing)

    # ---- 计算物体几何特征 (基于细网格粒子, 更准确) ----
    extent = fine_particles.max(axis=0) - fine_particles.min(axis=0)
    max_extent = float(extent.max())
    min_extent = float(extent[extent > 0].min()) if (extent > 0).any() else 0.0
    thin_layers = min_extent / voxel_size  # 最薄维度在 voxel_size 下的层数

    # ---- 薄物体检测 ----
    # 条件: 最薄维度 < 3 层 且 物体不是很小 (max_extent > 4*voxel_size)
    is_thin = thin_layers < 3.0 and max_extent > 4 * voxel_size

    if is_thin:
        # 薄物体: 使用 grid-based 增密 (经验证方案)
        # Step 1: 先下采样到粗间距获取形状模板
        coarse_particles = voxel_downsample(fine_particles, voxel_size)
        n_coarse = len(coarse_particles)
        # Step 2: 在粗粒子附近生成 fine_spacing 规则网格 + kd-tree 距离过滤
        final_particles = densify_thin_object(coarse_particles, fine_spacing)
        print(f"    两遍填充(fine={fine_spacing:.3f}): 细网格表面={n_fine_surface}, "
              f"内部={n_fine_interior}, 总={len(fine_particles)}")
        print(f"    → 薄物体检测: min_extent={min_extent:.3f} ({thin_layers:.1f}层), "
              f"grid增密: {n_coarse}@{voxel_size} → {len(final_particles)}@{fine_spacing}, "
              f"最终 {len(final_particles)} 粒子")
    else:
        # ---- Pass 2: 正常物体, 下采样到目标 spacing ----
        final_particles = voxel_downsample(fine_particles, voxel_size)

        # 计算填充率
        volume = float(max(extent[0], voxel_size) * max(extent[1], voxel_size) * max(extent[2], voxel_size))
        expected_solid = volume / (voxel_size ** 3)
        fill_ratio = len(final_particles) / max(expected_solid, 1)

        print(f"    两遍填充(fine={fine_spacing:.3f}): 细网格表面={n_fine_surface}, "
              f"内部={n_fine_interior}, 下采样后={len(final_particles)}, "
              f"fill_ratio={fill_ratio:.2f}, max_extent={max_extent:.3f}")

        # 空壳物体: 回退到凸包填充
        if use_convex_hull and fill_ratio < 0.3 and max_extent > 0.20:
            print(f"    fill_ratio={fill_ratio:.2f} 过低且 max_extent={max_extent:.3f}>0.20, "
                  f"回退到凸包填充")
            hull_particles = convex_hull_fill(fine_particles, voxel_size)
            if len(hull_particles) > len(final_particles):
                final_particles = hull_particles
                fill_ratio_new = len(final_particles) / max(expected_solid, 1)
                print(f"    凸包填充后: {len(final_particles)} 粒子, fill_ratio={fill_ratio_new:.2f}")

    return final_particles


def upsample_particles(points, min_particles, jitter_scale=0.01):
    """
    当粒子数不足 min_particles 时，通过复制+抖动上采样。

    保持整体 spacing 特征不变（抖动幅度 = spacing 的 20%），
    确保邻居数接近训练数据分布。

    Args:
        points: (N, 3) 当前粒子坐标
        min_particles: 最低粒子数要求
        jitter_scale: 抖动幅度 (默认 0.01, 即 spacing=0.05 的 20%)

    Returns:
        (min_particles, 3) 上采样后的粒子
    """
    if len(points) >= min_particles:
        return points

    need = min_particles - len(points)
    # 随机有放回采样
    indices = np.random.choice(len(points), need, replace=True)
    sampled = points[indices].copy()
    # 添加均匀抖动
    jitter = np.random.uniform(-jitter_scale, jitter_scale, sampled.shape)
    sampled += jitter

    return np.concatenate([points, sampled], axis=0)


def load_pose(pose_path):
    """加载物体 pose JSON 文件"""
    with open(pose_path, 'r') as f:
        pose = json.load(f)
    rotation = np.array(pose['rotation'][0], dtype=np.float32)    # (4,) 四元数
    translation = np.array(pose['translation'][0], dtype=np.float32)  # (3,)
    scale_factor = float(pose['scale'][0][0])                      # 各向同性缩放
    return rotation, translation, scale_factor


def quaternion_to_rotation_matrix(quat):
    """
    四元数转旋转矩阵 (PyTorch3D 格式: w, x, y, z)

    注意: pose JSON 中四元数格式为 PyTorch3D 的 (w, x, y, z)，实部在前
    PyTorch3D quaternion_to_matrix() 源码: r, i, j, k = unbind(q, -1) 其中 r=w
    """
    # PyTorch3D 格式: (w, x, y, z) — 实部 w 在第一位
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def transform_local_to_scene(points, rotation, translation, scale_factor):
    """
    将物体局部坐标变换到场景坐标系

    PyTorch3D 使用行向量 (row-vector) 惯例:
      compose_transform(scale, R, t).transform_points(p) = (p * s) @ R + t
    注意: 是 @ R 不是 @ R.T
    """
    R = quaternion_to_rotation_matrix(rotation)
    # PyTorch3D 行向量惯例: points @ R (不是 R.T)
    points_scene = (points * scale_factor) @ R + translation
    return points_scene


def detect_ground_y(all_points):
    """
    检测地面 Y 坐标

    使用所有物体点云的 Y 最小值作为地面高度。
    物体最低点即为地面（物体应坐落在地面上）。
    注意: 输入应已经过 opacity 过滤和体素下采样，噪声点已被清除。
    """
    if len(all_points) == 0:
        return 0.0
    return float(all_points[:, 1].min())


def align_to_physion_coords(particles_list, ground_y=None, target_spacing=0.05):
    """
    将场景坐标系对齐到 Physion 仿真坐标系

    Physion 坐标系:
    - Y 轴朝上 (scene_posed.ply 已确认 Y-up)
    - 地面在 Y ≈ 0
    - 粒子间距 ≈ 0.05 (binvox spacing)
    - 邻居搜索半径 = 0.08

    处理:
    1. 平移地面到 Y=0
    2. 居中 X/Z 避免坐标值过大
    3. 不做大幅缩放 (场景坐标尺寸已接近 Physion 尺度)
       只做微调: 使体素下采样后的粒子间距 = target_spacing

    Returns:
        aligned_list: 对齐后的粒子列表
        scale: 缩放因子
        coord_transform: dict 坐标变换参数，用于将 scene_posed 空间的任意点
                         变换到 Physion 空间: p_physion = (p_scene - offset) * scale
    """
    if not particles_list:
        return particles_list, 1.0, {"ground_y": 0.0, "center_x": 0.0, "center_z": 0.0, "scale": 1.0}

    # 合并所有粒子计算统计量
    all_points = np.concatenate(particles_list, axis=0)

    # 检测地面 Y 坐标
    if ground_y is None:
        ground_y = detect_ground_y(all_points)
    else:
        fallback_y = detect_ground_y(all_points)
        print(f"  floor_y (MoGe): {ground_y:.4f}, fallback (min obj): {fallback_y:.4f}, "
              f"diff: {abs(ground_y - fallback_y):.4f}")

    # 计算场景 X/Z 中心
    center_x = float((all_points[:, 0].min() + all_points[:, 0].max()) / 2)
    center_z = float((all_points[:, 2].min() + all_points[:, 2].max()) / 2)

    # 缩放策略: 场景坐标中的体素间距直接用 target_spacing
    # 不基于原始高斯密度计算缩放 (那会导致 92x 等极端值)
    # scene_posed 的坐标单位和 Physion 类似, scale ≈ 1.0
    scale = 1.0

    print(f"  坐标对齐: ground_y={ground_y:.4f}, center_xz=[{center_x:.4f},{center_z:.4f}], scale={scale:.4f}")

    # 应用变换: 居中 X/Z, 平移地面到 Y=0
    aligned_list = []
    for pts in particles_list:
        pts_aligned = pts.copy()
        pts_aligned[:, 0] -= center_x
        pts_aligned[:, 1] -= ground_y
        pts_aligned[:, 2] -= center_z
        pts_aligned *= scale
        aligned_list.append(pts_aligned.astype(np.float32))

    # 记录坐标变换参数: p_physion = (p_scene - [center_x, ground_y, center_z]) * scale
    coord_transform = {
        "ground_y": float(ground_y),
        "center_x": float(center_x),
        "center_z": float(center_z),
        "scale": float(scale),
        "description": "p_physion = (p_scene - [center_x, ground_y, center_z]) * scale"
    }

    return aligned_list, scale, coord_transform


def gs_to_particles_single(ply_path, pose_path=None, spacing=0.05,
                           opacity_threshold=0.5, max_particles=3000,
                           min_particles=20, fill_interior=True,
                           fine_factor=2):
    """
    将单个物体的 3DGS PLY 转换为粒子点云

    Args:
        ply_path: 3DGS PLY 文件路径
        pose_path: 物体 pose JSON 文件路径 (可选, 用于场景坐标变换)
        spacing: 目标粒子间距
        opacity_threshold: 不透明度过滤阈值
        max_particles: 最大粒子数
        min_particles: 最小粒子数
        fill_interior: 是否填充物体内部 (模拟 binvox 实心体素化)

    Returns:
        particles: (N, 3) 粒子位置
        info: dict 包含转换信息
    """
    # 加载高斯
    gs_data = load_gaussian_ply(ply_path)
    n_original = gs_data['n_gaussians']

    # 过滤低不透明度
    gs_data = filter_by_opacity(gs_data, opacity_threshold)
    n_filtered = gs_data['n_gaussians']

    # 提取高斯中心作为初始粒子
    particles = gs_data['xyz'].copy()

    # 如果有 pose, 变换到场景坐标系
    if pose_path is not None and os.path.exists(pose_path):
        rotation, translation, scale_factor = load_pose(pose_path)
        particles = transform_local_to_scene(particles, rotation, translation, scale_factor)

    # 体素化 + 可选内部填充
    if fill_interior:
        particles = voxel_fill_interior(particles, voxel_size=spacing, fine_factor=fine_factor)
    else:
        particles = voxel_downsample(particles, voxel_size=spacing)
    n_voxelized = len(particles)

    # 控制粒子数量
    if len(particles) > max_particles:
        indices = np.random.choice(len(particles), max_particles, replace=False)
        particles = particles[indices]

    info = {
        'n_original_gaussians': n_original,
        'n_after_opacity_filter': n_filtered,
        'n_after_voxelization': n_voxelized,
        'n_final_particles': len(particles),
        'fill_interior': fill_interior,
        'ply_path': ply_path,
    }

    if len(particles) < min_particles:
        print(f"  警告: 物体粒子数过少 ({len(particles)}), 低于最小阈值 {min_particles}")

    return particles.astype(np.float32), info


def _load_object_names(scene_dir, n_objects):
    """从 detection_result.json 加载物体名称"""
    object_names = [f'object_{i}' for i in range(n_objects)]
    det_path = os.path.join(scene_dir, 'detection_result.json')
    if os.path.exists(det_path):
        with open(det_path, 'r') as f:
            det_result = json.load(f)
        det_list = None
        if isinstance(det_result, list):
            det_list = det_result
        elif isinstance(det_result, dict) and 'objects' in det_result:
            det_list = det_result['objects']
        if det_list:
            for i, det in enumerate(det_list):
                if i < n_objects and 'label' in det:
                    object_names[i] = det['label']
                elif i < n_objects and 'name' in det:
                    object_names[i] = det['name']
    return object_names


def _load_scene_posed(scene_dir, opacity_threshold=0.5):
    """
    从 scene_posed.ply 加载场景坐标, 按物体高斯数分割

    scene_posed.ply 由 PyTorch3D make_scene() 生成，高斯按
    [object_0, object_1, ...] 顺序拼接

    Returns:
        scene_points: list of (N_i, 3) 每物体在场景坐标中的点云
        object_infos: list of dict
    """
    scene_path = os.path.join(scene_dir, 'scene_posed.ply')
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"找不到 {scene_path}")

    # 加载 scene_posed.ply
    print(f"加载 scene_posed.ply...")
    scene_data = load_gaussian_ply(scene_path)
    print(f"  总高斯数: {scene_data['n_gaussians']}")

    # 获取每个物体的高斯数
    object_counts = []
    obj_idx = 0
    while True:
        ply_path = os.path.join(scene_dir, f'object_{obj_idx}.ply')
        if not os.path.exists(ply_path):
            break
        obj_data = load_gaussian_ply(ply_path)
        object_counts.append(obj_data['n_gaussians'])
        print(f"  object_{obj_idx}: {obj_data['n_gaussians']} 高斯")
        obj_idx += 1

    if obj_idx == 0:
        raise FileNotFoundError(f"在 {scene_dir} 中未找到 object_*.ply 文件")

    # 验证高斯数匹配
    total = sum(object_counts)
    if total != scene_data['n_gaussians']:
        print(f"  警告: 高斯数不匹配! sum={total} vs scene={scene_data['n_gaussians']}")
        print(f"  回退到逐物体模式")
        return None, None

    print(f"  高斯数匹配: {total} = {' + '.join(str(c) for c in object_counts)}")

    # 按高斯数分割
    scene_points = []
    object_infos = []
    offset = 0
    for i, count in enumerate(object_counts):
        chunk_xyz = scene_data['xyz'][offset:offset + count]
        chunk_opacity = scene_data['opacity'][offset:offset + count]

        # 过滤低不透明度
        mask = chunk_opacity > opacity_threshold
        filtered_xyz = chunk_xyz[mask]

        info = {
            'n_original_gaussians': count,
            'n_after_opacity_filter': int(mask.sum()),
            'source': 'scene_posed',
        }
        scene_points.append(filtered_xyz)
        object_infos.append(info)

        centroid = filtered_xyz.mean(axis=0)
        print(f"  object_{i}: {count} → {int(mask.sum())} (opacity>{opacity_threshold}), "
              f"质心=[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        offset += count

    return scene_points, object_infos


def gs_scene_to_particles(scene_dir, spacing=0.05, opacity_threshold=0.5,
                          max_particles=3000, min_particles=150,
                          discard_threshold=20, use_scene_posed=True,
                          fill_interior=True, fine_factor=2,
                          use_convex_hull=True,
                          floor_y_scene=None):
    """
    将整个场景的 3DGS 输出转换为 Physion 格式粒子

    Args:
        scene_dir: auto_scene_pipeline.py 的输出目录
        spacing: 目标粒子间距 (Physion 坐标空间, 默认 0.05)
        opacity_threshold: 不透明度过滤阈值
        max_particles: 每物体最大粒子数
        min_particles: 每物体最低粒子数 (低于此值上采样, 默认 150)
        discard_threshold: 低于此值直接丢弃物体 (默认 20)
        use_scene_posed: 使用 scene_posed.ply (推荐, 默认True)
                         scene_posed.ply 由 PyTorch3D 正确变换, 物体相对位置准确
        fill_interior: 是否填充物体内部 (模拟 binvox 实心体素化)

    Returns:
        particles_list: list of (N_i, 3), 每物体粒子 (Physion 坐标)
        obj_points: list of (N_i, 3), 每物体局部坐标
        centroids: list of (3,), 每物体质心
        object_names: list of str, 物体名称
        scene_info: dict, 场景信息
    """
    particles_list = []
    object_names = []
    object_infos = []
    scene_raw_points = None  # 各物体在场景坐标中的点 (对齐前)

    # ============================================================
    # 方式 A: 使用 scene_posed.ply (推荐)
    # ============================================================
    if use_scene_posed:
        scene_raw_points, object_infos_posed = _load_scene_posed(
            scene_dir, opacity_threshold=opacity_threshold)

        if scene_raw_points is not None:
            object_infos = object_infos_posed
            n_objects = len(scene_raw_points)
            object_names = _load_object_names(scene_dir, n_objects)
            print(f"\n使用 scene_posed.ply 模式: {n_objects} 个物体")
        else:
            print("scene_posed 模式失败, 回退到逐物体模式")
            use_scene_posed = False

    # ============================================================
    # 方式 B: 逐物体加载 + 手动变换 (回退)
    # ============================================================
    if not use_scene_posed or scene_raw_points is None:
        scene_raw_points = []
        obj_idx = 0
        while True:
            ply_path = os.path.join(scene_dir, f'object_{obj_idx}.ply')
            pose_path = os.path.join(scene_dir, f'object_{obj_idx}_pose.json')
            if not os.path.exists(ply_path):
                break

            print(f"处理物体 {obj_idx}: {ply_path}")
            gs_data = load_gaussian_ply(ply_path)
            n_original = gs_data['n_gaussians']

            gs_data = filter_by_opacity(gs_data, opacity_threshold)
            n_filtered = gs_data['n_gaussians']
            points = gs_data['xyz'].copy()

            # 手动变换到场景坐标 (已修复四元数格式和旋转方向)
            if os.path.exists(pose_path):
                rotation, translation, scale_factor = load_pose(pose_path)
                points = transform_local_to_scene(points, rotation, translation, scale_factor)

            scene_raw_points.append(points)
            object_infos.append({
                'n_original_gaussians': n_original,
                'n_after_opacity_filter': n_filtered,
                'source': 'manual_transform',
            })
            object_names.append(f'object_{obj_idx}')
            print(f"  高斯数: {n_original} → {n_filtered}")
            obj_idx += 1

        if obj_idx == 0:
            raise FileNotFoundError(f"在 {scene_dir} 中未找到 object_*.ply 文件")
        object_names = _load_object_names(scene_dir, len(scene_raw_points))

    # ============================================================
    # 坐标系对齐到 Physion (平移地面到 Y=0, 居中 XZ)
    # 使用粗下采样计算对齐参数, 然后将同样的变换应用到原始高密度点
    # ============================================================
    print(f"\n计算坐标对齐参数 (粗下采样 voxel={spacing})...")
    downsampled_for_align = []
    for i, pts in enumerate(scene_raw_points):
        ds = voxel_downsample(pts, voxel_size=spacing)
        print(f"  物体 {i} ({object_names[i]}): {len(pts)} 高斯 → {len(ds)} 粗采样点")
        downsampled_for_align.append(ds)

    print("\n坐标系对齐到 Physion 仿真坐标系...")
    _, coord_scale, coord_transform = align_to_physion_coords(
        downsampled_for_align, ground_y=floor_y_scene, target_spacing=spacing)

    # 将同样的对齐变换应用到原始高密度点 (不先下采样, 保留密度信息给 fill)
    print("\n对齐原始高密度高斯点...")
    aligned_raw_points = []
    for i, pts in enumerate(scene_raw_points):
        pts_aligned = pts.copy()
        pts_aligned[:, 0] -= coord_transform['center_x']
        pts_aligned[:, 1] -= coord_transform['ground_y']
        pts_aligned[:, 2] -= coord_transform['center_z']
        pts_aligned *= coord_transform['scale']
        aligned_raw_points.append(pts_aligned.astype(np.float32))

    # ============================================================
    # 两遍体素化+内部填充 (在 Physion 坐标空间)
    # 使用原始高密度高斯点, 保证细网格能捕获足够的表面细节
    # ============================================================
    print(f"\n体素化+内部填充 (spacing={spacing}, fine_factor={fine_factor}, Physion 坐标空间)...")
    discarded_indices = []
    for i, pts in enumerate(aligned_raw_points):
        print(f"  物体 {i} ({object_names[i]}): {len(pts)} 高斯 →", end=' ')
        if fill_interior:
            voxelized = voxel_fill_interior(pts, voxel_size=spacing, fine_factor=fine_factor,
                                              use_convex_hull=use_convex_hull)
        else:
            voxelized = voxel_downsample(pts, voxel_size=spacing)

        # 丢弃极小物体
        if len(voxelized) < discard_threshold:
            print(f"{len(voxelized)} 粒子 — 丢弃 (低于阈值 {discard_threshold})")
            discarded_indices.append(i)
            continue

        # 下采样: 粒子过多时截断到 max_particles
        if len(voxelized) > max_particles:
            indices = np.random.choice(len(voxelized), max_particles, replace=False)
            voxelized = voxelized[indices]

        object_infos[i]['n_after_voxelization'] = len(voxelized)
        object_infos[i]['n_final_particles'] = len(voxelized)
        object_infos[i]['fill_interior'] = fill_interior
        particles_list.append(voxelized.astype(np.float32))
        print(f"{len(voxelized)} 粒子")

    # 清理已丢弃物体的名称和信息
    if discarded_indices:
        print(f"\n  丢弃了 {len(discarded_indices)} 个极小物体: "
              f"{[object_names[i] for i in discarded_indices]}")
        object_names = [n for i, n in enumerate(object_names) if i not in discarded_indices]
        object_infos = [info for i, info in enumerate(object_infos) if i not in discarded_indices]

    # ============================================================
    # 计算物体局部参考点云 (obj_points)
    # ============================================================
    obj_points = []
    centroids = []
    for pts in particles_list:
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        obj_pts = pts - centroid
        obj_points.append(obj_pts)

    total_particles = sum(len(p) for p in particles_list)

    scene_info = {
        'n_objects': len(particles_list),
        'n_particles': total_particles,
        'object_names': object_names,
        'object_infos': object_infos,
        'coord_scale': float(coord_scale),
        'coord_transform': coord_transform,
        'centroids': [c.tolist() for c in centroids],
        'particles_per_object': [len(p) for p in particles_list],
        'mode': 'scene_posed' if use_scene_posed else 'manual_transform',
        'discarded_indices': discarded_indices,
    }

    print(f"\n场景汇总: {scene_info['n_objects']} 个物体, 共 {total_particles} 个粒子")
    for i, name in enumerate(object_names):
        print(f"  物体 {i} ({name}): {len(particles_list[i])} 粒子, 质心 {centroids[i]}")

    return particles_list, obj_points, centroids, object_names, scene_info


def main():
    parser = argparse.ArgumentParser(description='3DGS → Physion 粒子转换')
    parser.add_argument('--scene-dir', type=str, required=True,
                        help='auto_scene_pipeline.py 的输出目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: scene_dir/particles)')
    parser.add_argument('--spacing', type=float, default=0.05,
                        help='目标粒子间距 (Physion 默认 0.05)')
    parser.add_argument('--opacity-threshold', type=float, default=0.5,
                        help='不透明度过滤阈值')
    parser.add_argument('--max-particles', type=int, default=3000,
                        help='每物体最大粒子数')
    parser.add_argument('--min-particles', type=int, default=150,
                        help='每物体最低粒子数 (低于此值上采样)')
    parser.add_argument('--discard-threshold', type=int, default=20,
                        help='低于此值直接丢弃物体')
    parser.add_argument('--no-scene-posed', action='store_true',
                        help='不使用 scene_posed.ply, 改用逐物体+手动变换')
    parser.add_argument('--no-fill', action='store_true',
                        help='不填充物体内部')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.scene_dir, 'particles')
    os.makedirs(args.output_dir, exist_ok=True)

    particles_list, obj_points, centroids, names, info = gs_scene_to_particles(
        scene_dir=args.scene_dir,
        spacing=args.spacing,
        opacity_threshold=args.opacity_threshold,
        max_particles=args.max_particles,
        min_particles=args.min_particles,
        discard_threshold=args.discard_threshold,
        use_scene_posed=not args.no_scene_posed,
        fill_interior=not args.no_fill,
    )

    # 保存粒子点云 (numpy)
    for i, (pts, obj_pts, name) in enumerate(zip(particles_list, obj_points, names)):
        np.save(os.path.join(args.output_dir, f'particles_{i}.npy'), pts)
        np.save(os.path.join(args.output_dir, f'obj_points_{i}.npy'), obj_pts)

    # 保存场景信息
    with open(os.path.join(args.output_dir, 'scene_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # 保存合并粒子 PLY (可视化用)
    all_pts = np.concatenate(particles_list, axis=0)
    save_particles_ply(os.path.join(args.output_dir, 'all_particles.ply'),
                       all_pts, particles_list)

    print(f"\n输出已保存到: {args.output_dir}")


def save_particles_ply(path, all_points, particles_list=None):
    """保存粒子点云为 PLY 文件 (用于可视化)"""
    n = len(all_points)

    # 为每个物体分配不同颜色
    colors = np.zeros((n, 3), dtype=np.uint8)
    palette = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 0, 0], [0, 128, 0], [0, 0, 128],
    ]

    if particles_list is not None:
        offset = 0
        for i, pts in enumerate(particles_list):
            color = palette[i % len(palette)]
            colors[offset:offset + len(pts)] = color
            offset += len(pts)
    else:
        colors[:] = [128, 128, 128]

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{all_points[i, 0]:.6f} {all_points[i, 1]:.6f} {all_points[i, 2]:.6f} "
                    f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


if __name__ == '__main__':
    main()
