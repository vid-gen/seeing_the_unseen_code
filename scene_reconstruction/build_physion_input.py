"""
build_physion_input.py - 将粒子点云构建为 Physion 训练数据格式

功能:
- 接收 gs_to_particles.py 输出的粒子数据
- 生成 phases_dict.pkl (场景元数据)
- 生成帧 .h5 文件 (物体位置和旋转)
- 生成 stat.h5 (归一化统计量)
- 支持设置初始速度

Physion 数据格式:
  trial_dir/
  ├── phases_dict.pkl   # 场景元数据
  ├── 0.h5              # 第0帧 (前一帧)
  ├── 3.h5              # 第1帧 (当前帧, 间隔 = training_fpt)
  └── stat.h5           # 归一化统计量 (全零)
"""

import os
import argparse
import json
import pickle
import numpy as np
import h5py


def generate_ground_particles(particles_list, spacing=0.05, y_height=0.005, margin=0.3):
    """
    生成地面粒子平面，匹配 Physion 训练数据格式。

    训练数据中地面是一个独立的 'cube' 物体 (material='rigid')，
    以 spacing=0.05 的规则网格分布在 Y=0.005 平面上。

    Args:
        particles_list: list of (N_i, 3) ndarray - 现有物体粒子（Physion 坐标）
        spacing: 粒子间距（默认 0.05，匹配训练数据）
        y_height: 地面 Y 高度（默认 0.005，匹配训练数据）
        margin: XZ 方向超出物体范围的额外覆盖（默认 0.3）

    Returns:
        ground_local: (M, 3) float64 地面局部坐标（Y=y_height, XZ 居中）
        ground_centroid: (3,) 地面质心 [center_x, 0.0, center_z]
    """
    # 合并所有物体粒子，计算 XZ 范围
    all_pts = np.concatenate(particles_list, axis=0)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    # XZ 各方向扩展 margin
    x_min -= margin
    x_max += margin
    z_min -= margin
    z_max += margin

    # 生成规则网格
    xs = np.arange(x_min, x_max + spacing * 0.5, spacing)
    zs = np.arange(z_min, z_max + spacing * 0.5, spacing)
    xx, zz = np.meshgrid(xs, zs)
    xx = xx.flatten()
    zz = zz.flatten()
    yy = np.full_like(xx, y_height)

    # 世界坐标
    world_pts = np.stack([xx, yy, zz], axis=1)  # (M, 3)

    # 质心: [center_x, 0.0, center_z] — Y=0 匹配训练数据
    center_x = (x_min + x_max) / 2.0
    center_z = (z_min + z_max) / 2.0
    ground_centroid = np.array([center_x, 0.0, center_z], dtype=np.float64)

    # 局部坐标 = 世界坐标 - 质心
    # Y: y_height - 0.0 = y_height (局部 Y = 0.005)
    # XZ: 相对于中心居中
    ground_local = world_pts - ground_centroid

    print(f"  地面粒子: {len(ground_local)} 个 "
          f"({len(xs)}x{len(zs)} 网格, spacing={spacing})")
    print(f"    XZ 范围: [{x_min:.3f}, {x_max:.3f}] x [{z_min:.3f}, {z_max:.3f}]")
    print(f"    质心: [{center_x:.3f}, 0.000, {center_z:.3f}]")

    return ground_local.astype(np.float64), ground_centroid


def create_phases_dict(obj_points_list, object_names=None, dt=0.01):
    """
    创建 Physion 格式的 phases_dict

    Args:
        obj_points_list: list of (N_i, 3) ndarray, 每物体局部坐标粒子
        object_names: list of str, 物体名称
        dt: 时间步长

    Returns:
        phases_dict: dict, 与 Physion 训练数据完全相同的格式
    """
    n_objects = len(obj_points_list)

    if object_names is None:
        object_names = [f'object_{i}' for i in range(n_objects)]

    # 粒子索引边界
    instance_idx = [0]
    total = 0
    for pts in obj_points_list:
        total += len(pts)
        instance_idx.append(total)

    # 聚类信息 (全零, 每物体一个 cluster)
    clusters = []
    for pts in obj_points_list:
        clusters.append([[np.zeros(len(pts), dtype=np.int32)]])

    # root_num 和 root_des_radius
    root_num = [[1]] * n_objects
    root_des_radius = [[0.05]] * n_objects

    phases_dict = {
        'n_objects': n_objects,
        'n_particles': np.int64(total),
        'instance': [name.encode('utf-8') if isinstance(name, str) else name
                     for name in object_names],
        'material': ['rigid'] * n_objects,
        'instance_idx': instance_idx,
        'obj_points': [pts.astype(np.float64) for pts in obj_points_list],
        'clusters': clusters,
        'root_num': root_num,
        'root_des_radius': root_des_radius,
        'dt': dt,
        'time_step': 300,  # 足够大, 不影响推理
        'yellow_id': 0,
        'red_id': min(1, n_objects - 1),
    }

    return phases_dict


def create_frame_h5(h5_path, centroids, rotations=None):
    """
    创建单帧 .h5 文件

    Args:
        h5_path: 输出路径
        centroids: (n_objects, 3) 物体质心位置
        rotations: (n_objects, 4) 四元数 (x,y,z,w), 默认单位旋转
    """
    n_objects = len(centroids)
    centroids = np.array(centroids, dtype=np.float32)

    if rotations is None:
        # 单位四元数 (x, y, z, w) = (0, 0, 0, 1) - scipy 格式
        rotations = np.zeros((n_objects, 4), dtype=np.float32)
        rotations[:, 3] = 1.0  # w = 1
    else:
        rotations = np.array(rotations, dtype=np.float32)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('obj_positions', data=centroids)
        f.create_dataset('obj_rotations', data=rotations)


def create_stat_h5(h5_path):
    """创建 stat.h5 (全零, 与训练数据一致)"""
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('positions', data=np.zeros((3, 3), dtype=np.float64))
        f.create_dataset('velocities', data=np.zeros((3, 3), dtype=np.float64))


def remap_velocity_spec(velocity_spec, discarded_indices):
    """
    物体丢弃后重映射 velocity_spec 中的索引。

    当 gs_to_particles 丢弃了某些极小物体后，剩余物体的索引会变化。
    此函数将原始 velocity_spec 中的索引映射到丢弃后的新索引。

    Args:
        velocity_spec: 原始速度规格字符串，如 "0:0.5,0,0;3:0,-2,0"
        discarded_indices: 被丢弃物体的原始索引列表，如 [2]

    Returns:
        重映射后的 velocity_spec 字符串

    Example:
        remap_velocity_spec("0:0.5,0,0;3:0,-2,0", [2])
        → "0:0.5,0,0;2:0,-2,0"
        (原索引3前面丢弃了索引2，新索引变为2)
    """
    if not velocity_spec or not discarded_indices:
        return velocity_spec

    discarded_set = set(discarded_indices)
    remapped_parts = []

    for part in velocity_spec.split(';'):
        part = part.strip()
        if not part:
            continue

        tokens = part.split(':')
        orig_idx = int(tokens[0])

        # 跳过被丢弃物体的速度条目
        if orig_idx in discarded_set:
            print(f"  velocity_spec: 物体 {orig_idx} 已被丢弃，跳过其速度设置")
            continue

        # 计算新索引：原索引减去它之前被丢弃的物体数
        new_idx = orig_idx - sum(1 for d in discarded_indices if d < orig_idx)

        # 重建条目
        tokens[0] = str(new_idx)
        remapped_parts.append(':'.join(tokens))

    remapped = ';'.join(remapped_parts) if remapped_parts else None
    if remapped != velocity_spec:
        print(f"  velocity_spec 重映射: {velocity_spec} → {remapped}")
    return remapped


def parse_velocity_spec(vel_spec, n_objects):
    """
    解析速度规格字符串

    格式: "obj_idx:vx,vy,vz" 或 "obj_idx:vx,vy,vz:scale"
    多个物体用分号分隔: "0:0,-1,0;1:1,0,0"

    Returns:
        velocities: (n_objects, 3) 速度向量
    """
    velocities = np.zeros((n_objects, 3), dtype=np.float32)

    if vel_spec is None or vel_spec.strip() == '':
        return velocities

    for part in vel_spec.split(';'):
        part = part.strip()
        if not part:
            continue

        tokens = part.split(':')
        obj_idx = int(tokens[0])
        vel_components = [float(x) for x in tokens[1].split(',')]
        vel = np.array(vel_components, dtype=np.float32)

        if len(tokens) > 2:
            scale = float(tokens[2])
            vel *= scale

        if obj_idx < n_objects:
            velocities[obj_idx] = vel
        else:
            print(f"  警告: 物体索引 {obj_idx} 超出范围 (共 {n_objects} 个物体)")

    return velocities


def angular_velocity_to_quaternion(omega, dt):
    """
    角速度 (wx,wy,wz) + 时间 → 旋转四元数 (scipy 格式 x,y,z,w)

    Args:
        omega: (3,) 角速度向量 (rad/s)，右手定则
        dt: 时间间隔 (s)

    Returns:
        (4,) 四元数 [x, y, z, w]
    """
    omega = np.array(omega, dtype=np.float64)
    angle = np.linalg.norm(omega) * dt  # 旋转角度 (rad)
    if angle < 1e-10:
        return np.array([0, 0, 0, 1], dtype=np.float32)  # 单位四元数
    axis = omega / np.linalg.norm(omega)  # 旋转轴(单位向量)
    half = angle / 2
    quat = np.array([
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
        np.cos(half)
    ], dtype=np.float32)
    return quat


def parse_angular_velocity_spec(angular_spec, n_objects):
    """
    解析角速度规格字符串

    格式与 velocity_spec 一致: "obj_idx:wx,wy,wz"
    多个物体用分号分隔: "0:0,0,-7;1:0,7,0"

    Args:
        angular_spec: 角速度规格字符串
        n_objects: 物体数量

    Returns:
        angular_velocities: (n_objects, 3) 角速度向量 (rad/s)
    """
    angular_velocities = np.zeros((n_objects, 3), dtype=np.float32)

    if not angular_spec or angular_spec.strip() == '':
        return angular_velocities

    for pair in angular_spec.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        tokens = pair.split(':')
        obj_idx = int(tokens[0].strip())
        wx, wy, wz = [float(v) for v in tokens[1].split(',')]
        if obj_idx < n_objects:
            angular_velocities[obj_idx] = [wx, wy, wz]
        else:
            print(f"  警告: 角速度物体索引 {obj_idx} 超出范围 (共 {n_objects} 个物体)")

    return angular_velocities


def compute_domino_rotation(velocity_spec, angular_velocity_spec, centroids, n_objects):
    """
    根据速度/角速度推断主运动方向，计算绕 Y 轴旋转使运动方向对齐 +X。

    Returns:
        rotation_matrix: (3, 3) 旋转矩阵，None 表示无需旋转
        theta: float 旋转角度 (rad)，0 表示无需旋转
    """
    velocities = parse_velocity_spec(velocity_spec, n_objects)
    angular_vels = parse_angular_velocity_spec(angular_velocity_spec, n_objects)

    direction = None

    # 优先从线速度推断方向
    for i in range(n_objects):
        v = velocities[i]
        vxz = np.array([v[0], v[2]])
        if np.linalg.norm(vxz) > 0.1:
            direction = vxz
            print(f"  多米诺方向检测: 物体 {i} 线速度 XZ=({v[0]:.2f}, {v[2]:.2f})")
            break

    # 从角速度推断倾倒方向
    if direction is None:
        for i in range(n_objects):
            omega = angular_vels[i]
            if np.linalg.norm(omega) < 0.1:
                continue
            # omega_z < 0 → 倒向 +X, omega_z > 0 → 倒向 -X
            # omega_x < 0 → 倒向 +Z, omega_x > 0 → 倒向 -Z
            dx = -omega[2]  # Z 轴角速度决定 X 方向
            dz = omega[0]   # X 轴角速度决定 Z 方向
            if abs(dx) > 0.01 or abs(dz) > 0.01:
                direction = np.array([dx, dz])
                print(f"  多米诺方向检测: 物体 {i} 角速度 omega=({omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f}) → 倾倒方向 XZ=({dx:.2f}, {dz:.2f})")
                break

    if direction is None:
        print("  多米诺方向检测: 未检测到水平运动方向，跳过旋转")
        return None, 0.0

    direction = direction / np.linalg.norm(direction)
    theta = np.arctan2(direction[1], direction[0])  # direction 与 +X 的夹角

    if abs(theta) < 0.1:  # ~5.7°, 已接近 +X
        print(f"  多米诺方向检测: 方向已接近 +X (theta={np.degrees(theta):.1f}°)，跳过旋转")
        return None, 0.0

    # 绕 Y 轴旋转 -theta，使运动方向对齐 +X
    c, s = np.cos(-theta), np.sin(-theta)
    R_y = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ], dtype=np.float64)

    print(f"  多米诺方向旋转: theta={np.degrees(theta):.1f}°, 旋转 {np.degrees(-theta):.1f}° 使运动方向对齐 +X")

    return R_y, theta


def apply_domino_rotation(particles_list, obj_points, centroids, R):
    """
    对所有粒子坐标和质心施加旋转矩阵。就地修改。
    """
    for i in range(len(particles_list)):
        particles_list[i] = (particles_list[i] @ R.T).astype(np.float32)
        obj_points[i] = (obj_points[i] @ R.T).astype(np.float32)
        centroids[i] = (np.array(centroids[i]) @ R.T).astype(np.float32)
        if isinstance(centroids[i], np.ndarray):
            centroids[i] = centroids[i].tolist()


def rotate_velocity_spec(velocity_spec, R):
    """将 velocity_spec 字符串中的速度向量旋转。"""
    if not velocity_spec or velocity_spec.strip() == '':
        return velocity_spec

    parts = []
    for pair in velocity_spec.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        tokens = pair.split(':')
        obj_idx = tokens[0].strip()
        vel = np.array([float(x) for x in tokens[1].split(',')], dtype=np.float64)
        vel_rotated = vel @ R.T
        parts.append(f"{obj_idx}:{vel_rotated[0]:.6f},{vel_rotated[1]:.6f},{vel_rotated[2]:.6f}")

    return ";".join(parts)


def rotate_angular_velocity_spec(angular_spec, R):
    """将 angular_velocity_spec 字符串中的角速度向量旋转。"""
    if not angular_spec or angular_spec.strip() == '':
        return angular_spec

    parts = []
    for pair in angular_spec.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        tokens = pair.split(':')
        obj_idx = tokens[0].strip()
        omega = np.array([float(x) for x in tokens[1].split(',')], dtype=np.float64)
        omega_rotated = omega @ R.T
        parts.append(f"{obj_idx}:{omega_rotated[0]:.6f},{omega_rotated[1]:.6f},{omega_rotated[2]:.6f}")

    return ";".join(parts)


def build_trial(obj_points_list, centroids, object_names=None,
                output_dir='trial_output', velocity_spec=None,
                angular_velocity_spec=None,
                training_fpt=3, dt=0.01, n_real_objects=None):
    """
    构建完整的 Physion trial 目录

    Args:
        obj_points_list: list of (N_i, 3) ndarray, 每物体局部坐标粒子
        centroids: list/array of (3,), 每物体质心
        object_names: list of str, 物体名称
        output_dir: 输出目录
        velocity_spec: 速度规格字符串 (见 parse_velocity_spec)
        angular_velocity_spec: 角速度规格字符串 (见 parse_angular_velocity_spec)，
                               格式: "obj_idx:wx,wy,wz"，单位 rad/s
        training_fpt: 帧间隔 (与训练一致, 默认 3)
        dt: 物理时间步长
        n_real_objects: 真实物体数量（不含地面等附加物体），
                        velocity_spec 只对前 n_real_objects 个物体生效，
                        其余物体（如地面）速度为零。默认 None 表示全部。

    Returns:
        trial_path: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)

    n_objects = len(obj_points_list)
    centroids = np.array(centroids, dtype=np.float32)

    # 1. 生成 phases_dict.pkl
    phases_dict = create_phases_dict(obj_points_list, object_names, dt)
    pkl_path = os.path.join(output_dir, 'phases_dict.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(phases_dict, f)
    print(f"  保存 phases_dict.pkl: {n_objects} 物体, {phases_dict['n_particles']} 粒子")

    # 2. 解析初始速度（只对真实物体，地面等附加物体速度为零）
    n_vel_objects = n_real_objects if n_real_objects is not None else n_objects
    velocities = parse_velocity_spec(velocity_spec, n_vel_objects)
    if n_vel_objects < n_objects:
        pad = np.zeros((n_objects - n_vel_objects, 3), dtype=np.float32)
        velocities = np.vstack([velocities, pad])
        print(f"  速度分配: 前 {n_vel_objects} 个真实物体接受 velocity_spec，"
              f"后 {n_objects - n_vel_objects} 个附加物体（地面等）速度为零")
    has_velocity = np.any(np.abs(velocities) > 1e-6)

    if has_velocity:
        print(f"  初始速度:")
        for i in range(n_objects):
            if np.any(np.abs(velocities[i]) > 1e-6):
                print(f"    物体 {i}: vel = [{velocities[i, 0]:.3f}, {velocities[i, 1]:.3f}, {velocities[i, 2]:.3f}]")
    else:
        print(f"  零速度模式 (自由落体)")

    # 2b. 解析角速度（只对真实物体，地面等附加物体无角速度）
    angular_vels = parse_angular_velocity_spec(angular_velocity_spec, n_vel_objects)
    if n_vel_objects < n_objects:
        pad = np.zeros((n_objects - n_vel_objects, 3), dtype=np.float32)
        angular_vels = np.vstack([angular_vels, pad])
    has_angular = np.any(np.abs(angular_vels) > 1e-6)

    if has_angular:
        print(f"  初始角速度:")
        for i in range(n_objects):
            if np.any(np.abs(angular_vels[i]) > 1e-6):
                print(f"    物体 {i}: omega = [{angular_vels[i, 0]:.3f}, {angular_vels[i, 1]:.3f}, {angular_vels[i, 2]:.3f}] rad/s")

    # 3. 生成帧数据
    # 推理代码中: t_prev = max(0, t0 - training_fpt), t0 = start_timestep
    # 默认 start_timestep = training_fpt (这样 t_prev=0, t0=training_fpt)
    # 需要生成第 0 帧和第 training_fpt 帧

    # 当前帧 (t0): 物体在当前位置
    centroids_t0 = centroids.copy()

    # 前一帧 (t_prev): 如果有速度, 反推前一帧位置
    # vel = (pos_t0 - pos_prev) / (training_fpt * dt)
    # pos_prev = pos_t0 - vel * training_fpt * dt
    delta_t = training_fpt * dt  # 帧间实际时间间隔
    centroids_prev = centroids_t0 - velocities * delta_t

    # 角速度: 帧 0 使用反向旋转（从帧 3 的单位旋转反推），帧 3 保持单位旋转
    # GNS 计算 (R3 @ pts + t3 - R0 @ pts - t0) / delta_t 时，
    # 不同粒子会有不同速度 → 模型感知到旋转
    rotations_prev = None
    if has_angular:
        rotations_prev = np.zeros((n_objects, 4), dtype=np.float32)
        rotations_prev[:, 3] = 1.0  # 默认单位四元数
        for i in range(n_objects):
            if np.any(np.abs(angular_vels[i]) > 1e-6):
                # 帧 0 = 帧 3 的前一帧 → 反向旋转
                rotations_prev[i] = angular_velocity_to_quaternion(
                    -angular_vels[i], delta_t)

    # 保存第 0 帧（含角速度反向旋转）
    create_frame_h5(os.path.join(output_dir, '0.h5'), centroids_prev, rotations_prev)
    print(f"  保存 0.h5: 前一帧位置" +
          (" + 反向旋转" if has_angular else ""))

    # 保存第 training_fpt 帧（单位旋转）
    create_frame_h5(os.path.join(output_dir, f'{training_fpt}.h5'), centroids_t0)
    print(f"  保存 {training_fpt}.h5: 当前帧位置")

    # 为了让推理脚本可以加载更多帧 (虽然是自回归), 生成足够的帧
    # 这些帧只用于初始化, 后续帧由模型预测
    # 但为了安全, 多生成一些帧 (位置不变, 作为 "GT" 占位)
    for t in range(training_fpt * 2, training_fpt * 100 + 1, training_fpt):
        # 用简单的匀速运动外推 (仅作占位, 不影响自回归推理)
        steps_from_t0 = (t - training_fpt) / delta_t
        centroids_t = centroids_t0 + velocities * steps_from_t0 * delta_t

        # 角速度: 累积旋转
        rotations_t = None
        if has_angular:
            elapsed = steps_from_t0 * delta_t  # 从 t0 起的时间 (秒)
            rotations_t = np.zeros((n_objects, 4), dtype=np.float32)
            rotations_t[:, 3] = 1.0
            for i in range(n_objects):
                if np.any(np.abs(angular_vels[i]) > 1e-6):
                    rotations_t[i] = angular_velocity_to_quaternion(
                        angular_vels[i], elapsed)

        create_frame_h5(os.path.join(output_dir, f'{t}.h5'), centroids_t, rotations_t)

    # 4. 生成 stat.h5
    create_stat_h5(os.path.join(output_dir, 'stat.h5'))
    print(f"  保存 stat.h5: 全零统计量")

    # 5. 保存构建信息
    build_info = {
        'n_objects': n_objects,
        'n_particles': int(phases_dict['n_particles']),
        'object_names': object_names or [f'object_{i}' for i in range(n_objects)],
        'centroids': centroids.tolist(),
        'velocity_spec': velocity_spec,
        'velocities': velocities.tolist(),
        'angular_velocity_spec': angular_velocity_spec,
        'angular_velocities': angular_vels.tolist(),
        'has_angular_velocity': bool(has_angular),
        'training_fpt': training_fpt,
        'dt': dt,
        'particles_per_object': [len(p) for p in obj_points_list],
    }
    with open(os.path.join(output_dir, 'build_info.json'), 'w') as f:
        json.dump(build_info, f, indent=2)

    print(f"\nTrial 数据已保存到: {output_dir}")
    return output_dir


def build_from_gs_output(scene_dir, output_dir, velocity_spec=None,
                         angular_velocity_spec=None,
                         spacing=0.05, opacity_threshold=0.5,
                         max_particles=3000, min_particles=150,
                         discard_threshold=20, training_fpt=3,
                         fill_interior=True, fine_factor=2,
                         use_convex_hull=True,
                         add_ground=True, ground_margin=0.3,
                         ground_clearance=0.020,
                         floor_y_scene=None):
    """
    从 auto_scene_pipeline.py 的输出直接构建 Physion trial

    这是一个便捷函数, 串联 gs_to_particles + build_trial
    """
    from gs_to_particles import gs_scene_to_particles

    print("=" * 60)
    print("Step 1: 3DGS → 粒子转换")
    print("=" * 60)

    particles_list, obj_points, centroids, names, scene_info = gs_scene_to_particles(
        scene_dir=scene_dir,
        spacing=spacing,
        opacity_threshold=opacity_threshold,
        max_particles=max_particles,
        min_particles=min_particles,
        discard_threshold=discard_threshold,
        fill_interior=fill_interior,
        fine_factor=fine_factor,
        use_convex_hull=use_convex_hull,
        floor_y_scene=floor_y_scene,
    )

    # 重映射 velocity_spec 索引（如果有物体被丢弃）
    discarded_indices = scene_info.get('discarded_indices', [])
    if discarded_indices and velocity_spec:
        velocity_spec = remap_velocity_spec(velocity_spec, discarded_indices)
    if discarded_indices and angular_velocity_spec:
        angular_velocity_spec = remap_velocity_spec(angular_velocity_spec, discarded_indices)

    # 记录真实物体数量（地面粒子添加前）
    n_real_objects = len(particles_list)

    # 添加地面粒子物体（匹配训练数据格式）
    if add_ground:
        print("\n  添加地面粒子...")

        # 1. 将所有物体粒子上移 ground_clearance，为地面腾出空间
        for i in range(len(particles_list)):
            particles_list[i][:, 1] += ground_clearance
            centroids[i][1] += ground_clearance
            # obj_points 局部坐标不需要改，因为质心也上移了

        # 2. 记录 ground_clearance 到 coord_transform（给 camera_restore.py 投影补偿用）
        # 不再修改 ground_y，改为显式记录 ground_clearance，在投影时统一补偿
        if 'coord_transform' in scene_info:
            ct = scene_info['coord_transform']
            ct['ground_clearance'] = float(ground_clearance)

        # 3. 生成地面粒子
        ground_local, ground_centroid = generate_ground_particles(
            particles_list, spacing=spacing, margin=ground_margin)

        # 4. 追加地面到列表末尾（不影响用户 velocity_spec 索引）
        # 地面的世界坐标粒子 = ground_local + ground_centroid
        ground_world = ground_local + ground_centroid
        particles_list.append(ground_world)
        obj_points.append(ground_local)
        centroids.append(ground_centroid)
        names.append('cube')  # 匹配训练数据中地面的 instance 名称

        print(f"  物体上移 {ground_clearance:.3f}, 地面追加为第 {len(names)-1} 个物体")

    print("\n" + "=" * 60)
    print("Step 2: 构建 Physion 数据格式")
    print("=" * 60)

    trial_path = build_trial(
        obj_points_list=obj_points,
        centroids=centroids,
        object_names=names,
        output_dir=output_dir,
        velocity_spec=velocity_spec,
        angular_velocity_spec=angular_velocity_spec,
        training_fpt=training_fpt,
        n_real_objects=n_real_objects,
    )

    # 额外保存粒子可视化
    from gs_to_particles import save_particles_ply
    all_pts = np.concatenate(particles_list, axis=0)
    save_particles_ply(os.path.join(output_dir, 'particles_vis.ply'),
                       all_pts, particles_list)
    print(f"  保存粒子可视化: particles_vis.ply")

    # 保存坐标变换参数 (用于摄像头还原)
    if 'coord_transform' in scene_info:
        coord_transform_path = os.path.join(output_dir, 'coord_transform.json')
        with open(coord_transform_path, 'w') as f:
            json.dump(scene_info['coord_transform'], f, indent=2)
        print(f"  保存坐标变换: coord_transform.json")

    return trial_path, scene_info


def build_from_mesh_output(scene_dir, output_dir, velocity_spec=None,
                            angular_velocity_spec=None,
                            spacing=0.05,
                            discard_threshold=10, training_fpt=3,
                            add_ground=True, ground_margin=0.3,
                            ground_clearance=0.020,
                            max_particles_per_obj=600,
                            min_particles_per_obj=20,
                            target_particles=350,
                            floor_y_scene=None,
                            ground_plane=None,
                            thicken=False,
                            thicken_min_layers=3,
                            align_z=False):
    """
    从 mesh 输出直接构建 Physion trial

    与 build_from_gs_output() 对应的 mesh 版本。
    使用 mesh_to_particles.py (binvox 方式) 进行粒子转换,
    与 Physion 训练数据预处理完全一致。
    基于粒子数量自适应缩放场景。
    """
    from mesh_to_particles import mesh_scene_to_particles, save_particles_ply

    print("=" * 60)
    print("Step 1: Mesh → 粒子转换 (binvox 方式)")
    print("=" * 60)

    particles_list, obj_points, centroids, names, scene_info = mesh_scene_to_particles(
        scene_dir=scene_dir,
        spacing=spacing,
        discard_threshold=discard_threshold,
        max_particles_per_obj=max_particles_per_obj,
        min_particles_per_obj=min_particles_per_obj,
        target_particles=target_particles,
        floor_y_scene=floor_y_scene,
        ground_plane=ground_plane,
        thicken=thicken,
        thicken_min_layers=thicken_min_layers,
    )

    # 重映射 velocity_spec 索引（如果有物体被丢弃）
    discarded_indices = scene_info.get('discarded_indices', [])
    if discarded_indices and velocity_spec:
        velocity_spec = remap_velocity_spec(velocity_spec, discarded_indices)
    if discarded_indices and angular_velocity_spec:
        angular_velocity_spec = remap_velocity_spec(angular_velocity_spec, discarded_indices)

    # 多米诺方向旋转：使主运动方向对齐 +X（GNS 模型偏好 +X 方向）
    domino_R, domino_theta = compute_domino_rotation(
        velocity_spec, angular_velocity_spec, centroids, len(particles_list))

    if domino_R is not None:
        print(f"\n  应用多米诺方向旋转...")
        apply_domino_rotation(particles_list, obj_points, centroids, domino_R)
        if velocity_spec:
            velocity_spec = rotate_velocity_spec(velocity_spec, domino_R)
            print(f"  旋转后 velocity_spec: {velocity_spec}")
        if angular_velocity_spec:
            angular_velocity_spec = rotate_angular_velocity_spec(angular_velocity_spec, domino_R)
            print(f"  旋转后 angular_velocity_spec: {angular_velocity_spec}")

        if 'coord_transform' in scene_info:
            scene_info['coord_transform']['domino_rotation_theta'] = float(domino_theta)
            scene_info['coord_transform']['domino_rotation_matrix'] = domino_R.tolist()

    # Z 轴对齐：将所有物体的 Z 中心对齐到均值
    if align_z and len(particles_list) > 1:
        z_centers = [c[2] for c in centroids]
        z_mean = np.mean(z_centers)
        print(f"\n  Z 轴对齐: 各物体 Z 中心 = {[f'{z:.3f}' for z in z_centers]}")
        print(f"  目标 Z = {z_mean:.3f}")
        for i in range(len(particles_list)):
            dz = z_mean - z_centers[i]
            if abs(dz) > 1e-4:
                particles_list[i][:, 2] += dz
                centroids[i][2] += dz
                print(f"    Object {i} ({names[i]}): Z 偏移 {dz:+.3f}")

    # 记录真实物体数量（地面粒子添加前）
    n_real_objects = len(particles_list)

    # 添加地面粒子物体（匹配训练数据格式）
    if add_ground:
        print("\n  添加地面粒子...")

        for i in range(len(particles_list)):
            particles_list[i][:, 1] += ground_clearance
            centroids[i][1] += ground_clearance

        if 'coord_transform' in scene_info:
            ct = scene_info['coord_transform']
            ct['ground_clearance'] = float(ground_clearance)

        ground_local, ground_centroid = generate_ground_particles(
            particles_list, spacing=spacing, margin=ground_margin)

        ground_world = ground_local + ground_centroid
        particles_list.append(ground_world)
        obj_points.append(ground_local)
        centroids.append(ground_centroid)
        names.append('cube')

        print(f"  物体上移 {ground_clearance:.3f}, 地面追加为第 {len(names)-1} 个物体")

    print("\n" + "=" * 60)
    print("Step 2: 构建 Physion 数据格式")
    print("=" * 60)

    trial_path = build_trial(
        obj_points_list=obj_points,
        centroids=centroids,
        object_names=names,
        output_dir=output_dir,
        velocity_spec=velocity_spec,
        angular_velocity_spec=angular_velocity_spec,
        training_fpt=training_fpt,
        n_real_objects=n_real_objects,
    )

    # 保存粒子可视化
    all_pts = np.concatenate(particles_list, axis=0)
    save_particles_ply(os.path.join(output_dir, 'particles_vis.ply'),
                       all_pts, particles_list)
    print(f"  保存粒子可视化: particles_vis.ply")

    # 保存坐标变换参数
    if 'coord_transform' in scene_info:
        ct = scene_info['coord_transform']
        # 将场景缩放因子写入 coord_transform (相机还原需要反变换)
        if 'scene_scale_factor' in scene_info:
            ct['scene_scale_factor'] = scene_info['scene_scale_factor']
        coord_transform_path = os.path.join(output_dir, 'coord_transform.json')
        with open(coord_transform_path, 'w') as f:
            json.dump(ct, f, indent=2)
        print(f"  保存坐标变换: coord_transform.json")

    return trial_path, scene_info


def main():
    parser = argparse.ArgumentParser(description='构建 Physion 格式输入数据')

    # 输入来源 (二选一)
    parser.add_argument('--scene-dir', type=str, default=None,
                        help='auto_scene_pipeline.py 输出目录 (直接从 3DGS 转换)')
    parser.add_argument('--particles-dir', type=str, default=None,
                        help='gs_to_particles.py 输出目录 (已有粒子数据)')

    # 输出
    parser.add_argument('--output-dir', type=str, default='trial_output',
                        help='输出 trial 目录')

    # 速度设置
    parser.add_argument('--velocity', type=str, default=None,
                        help='初始速度: "obj_idx:vx,vy,vz" 多个用分号分隔')

    # 粒子化参数 (仅 --scene-dir 模式)
    parser.add_argument('--spacing', type=float, default=0.05,
                        help='粒子间距')
    parser.add_argument('--opacity-threshold', type=float, default=0.5,
                        help='不透明度阈值')
    parser.add_argument('--max-particles', type=int, default=3000,
                        help='每物体最大粒子数')

    # 训练参数
    parser.add_argument('--training-fpt', type=int, default=3,
                        help='帧间隔 (与训练一致)')

    # 地面粒子参数
    parser.add_argument('--no-ground', dest='add_ground',
                        action='store_false', default=True,
                        help='不添加地面粒子（默认添加）')
    parser.add_argument('--ground-margin', type=float, default=0.3,
                        help='地面 XZ 方向超出物体范围的额外覆盖')

    args = parser.parse_args()

    if args.scene_dir is not None:
        # 模式1: 直接从 3DGS 场景构建
        build_from_gs_output(
            scene_dir=args.scene_dir,
            output_dir=args.output_dir,
            velocity_spec=args.velocity,
            spacing=args.spacing,
            opacity_threshold=args.opacity_threshold,
            max_particles=args.max_particles,
            training_fpt=args.training_fpt,
            add_ground=args.add_ground,
            ground_margin=args.ground_margin,
        )
    elif args.particles_dir is not None:
        # 模式2: 从已有粒子数据构建
        print("从已有粒子数据构建 Physion trial...")
        scene_info_path = os.path.join(args.particles_dir, 'scene_info.json')
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)

        obj_points_list = []
        centroids = []
        n_objects = scene_info['n_objects']

        for i in range(n_objects):
            obj_pts = np.load(os.path.join(args.particles_dir, f'obj_points_{i}.npy'))
            pts = np.load(os.path.join(args.particles_dir, f'particles_{i}.npy'))
            obj_points_list.append(obj_pts)
            centroids.append(pts.mean(axis=0))

        build_trial(
            obj_points_list=obj_points_list,
            centroids=centroids,
            object_names=scene_info.get('object_names'),
            output_dir=args.output_dir,
            velocity_spec=args.velocity,
            training_fpt=args.training_fpt,
        )
    else:
        parser.error("必须指定 --scene-dir 或 --particles-dir")


if __name__ == '__main__':
    main()
