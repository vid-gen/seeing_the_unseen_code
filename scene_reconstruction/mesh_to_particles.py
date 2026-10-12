"""
mesh_to_particles.py - 将 SAM 3D Objects 输出的 Mesh 转换为 Physion 格式粒子

核心思路:
  使用与 Physion 训练数据预处理完全一致的方式:
  binvox -aw -dc (depth carving) 体素化 → 粒子坐标计算 → 边界外过滤
  参考: Physion-particles/utils_geom.py:mesh_to_particles()

依赖:
  - trimesh (mesh 加载和导出 OBJ)
  - binvox (外部可执行文件, 路径: BINVOX_PATH)
  - binvox_rw (本地模块, 读取 .binvox 文件)
  - numpy
"""

import os
import json
import glob
import copy
import shutil
import argparse
import subprocess
import tempfile
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

import binvox_rw

# binvox 可执行文件路径
# Set the BINVOX_PATH environment variable, or put `binvox` on $PATH.
BINVOX_PATH = os.environ.get('BINVOX_PATH') or shutil.which('binvox') or 'binvox'


def _as_mesh(scene_or_mesh):
    """将 trimesh.Scene 转换为 trimesh.Trimesh (复制自 Physion utils_geom.py)"""
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = trimesh.Trimesh()
        else:
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def load_mesh_ply(mesh_path):
    """
    加载 mesh PLY 文件 (使用 process=True, 与 Physion 一致)

    Returns:
        trimesh.Trimesh 对象
    """
    if trimesh is None:
        raise ImportError("需要安装 trimesh: pip install trimesh")
    mesh = _as_mesh(trimesh.load_mesh(mesh_path, process=True))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{mesh_path} 不是有效的 trimesh 对象 (type={type(mesh)})")
    return mesh


def repair_mesh(mesh):
    """
    修复 mesh 使其尽可能 watertight

    尝试多种修复策略，按优先级递减:
    1. trimesh 内置修复 (轻量)
    2. pymeshfix (中量)
    3. SAM 3D Objects 自带的 postprocess_mesh (重量，多视角光栅化+mincut)

    Returns:
        修复后的 trimesh.Trimesh
    """
    # 策略 1: trimesh 内置修复
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    if mesh.is_watertight:
        print(f"    trimesh 修复成功 → watertight")
        return mesh

    # 策略 2: pymeshfix
    try:
        import pymeshfix
        meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        meshfix.repair()
        mesh_fixed = trimesh.Trimesh(meshfix.v, meshfix.f, process=False)
        if mesh_fixed.is_watertight:
            print(f"    pymeshfix 修复成功 → watertight")
            return mesh_fixed
    except ImportError:
        pass
    except Exception as e:
        print(f"    pymeshfix 修复失败: {e}")

    # 策略 3: SAM 3D Objects 的 postprocess_mesh
    try:
        from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh
        verts, faces = postprocess_mesh(
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            simplify=False,
            fill_holes=True,
            fill_holes_num_views=500,
        )
        mesh_fixed = trimesh.Trimesh(verts, faces, process=False)
        if mesh_fixed.is_watertight:
            print(f"    postprocess_mesh 修复成功 → watertight")
            return mesh_fixed
        else:
            print(f"    postprocess_mesh 修复后仍非 watertight, 使用修复后结果继续")
            return mesh_fixed
    except ImportError:
        print(f"    postprocess_mesh 不可用")
    except Exception as e:
        print(f"    postprocess_mesh 修复失败: {e}")

    print(f"    所有修复策略均未达到 watertight, 使用当前状态继续")
    return mesh


def mesh_to_solid_particles(mesh, spacing=0.05):
    """
    Physion 方式: binvox -aw -dc 体素化 + 深度雕刻填充
    与 Physion-particles/utils_geom.py:mesh_to_particles() 完全一致

    Args:
        mesh: trimesh.Trimesh 对象
        spacing: 粒子间距 (默认 0.05, 匹配 Physion 训练数据)

    Returns:
        particles: (N, 3) np.float32 粒子坐标
        info: dict 包含转换信息
    """
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.faces)

    # 保存原始 mesh 边界 (用于后续过滤)
    mesh_ori = copy.deepcopy(mesh)

    # ---- 计算体素化参数 (完全复制 Physion utils_geom.py:90-151) ----
    edges = mesh.bounding_box.extents
    meshLower = mesh.bounds[0, :].copy()
    meshUpper = mesh.bounds[1, :].copy()

    spacingEps = spacing * (1.0 - 1e-4)
    spacingEps_p = 0  # Physion 源码中此值为 0

    # 各轴体素数 (至少 1)
    dx = 1 if spacing > edges[0] else int(edges[0] / spacingEps)
    dy = 1 if spacing > edges[1] else int(edges[1] / spacingEps)
    dz = 1 if spacing > edges[2] else int(edges[2] / spacingEps)

    maxDim = max(dx, dy, dz)

    # 边界框扩展 ±2 voxels
    meshLower_spaced = meshLower - 2.0 * spacing - spacingEps_p
    meshUpper_spaced = meshUpper + 2.0 * spacing + spacingEps_p
    maxDim_spaced = maxDim + 4

    # binvox 渲染分辨率 = 目标粗网格大小 (与 Physion 训练预处理一致)
    binvox_dim = maxDim_spaced

    # 体素网格限制 512
    voxelsize_limit = 512
    if maxDim_spaced > voxelsize_limit:
        for dim in range(3):
            if edges[dim] < (voxelsize_limit - 4) * spacing:
                continue
            else:
                amount_to_cut = edges[dim] - (voxelsize_limit - 4) * spacing
                meshLower_spaced[dim] += amount_to_cut * 0.5
                meshUpper_spaced[dim] -= amount_to_cut * 0.5
        maxDim_spaced = voxelsize_limit

    # 体素中心对称偏移
    meshOffset = np.zeros(3)
    meshOffset[0] = 0.5 * (spacing - (edges[0] - (dx - 1) * spacing))
    meshOffset[1] = 0.5 * (spacing - (edges[1] - (dy - 1) * spacing))
    meshOffset[2] = 0.5 * (spacing - (edges[2] - (dz - 1) * spacing))
    meshLower_spaced -= meshOffset

    meshUpper_spaced = meshLower_spaced + maxDim_spaced * spacing + 2 * spacingEps_p

    print(f"    binvox 参数: dim={binvox_dim} (粗网格={maxDim_spaced}), dx={dx}, dy={dy}, dz={dz}")
    print(f"    bbox: [{meshLower_spaced[0]:.4f}, {meshLower_spaced[1]:.4f}, {meshLower_spaced[2]:.4f}] "
          f"→ [{meshUpper_spaced[0]:.4f}, {meshUpper_spaced[1]:.4f}, {meshUpper_spaced[2]:.4f}]")

    # ---- 保存为临时 OBJ 文件 ----
    tmp_dir = tempfile.mkdtemp(prefix='binvox_')
    tmp_obj = os.path.join(tmp_dir, 'mesh.obj')
    mesh.export(tmp_obj, file_type='obj')

    binvox_file = os.path.join(tmp_dir, 'mesh.binvox')

    try:
        # ---- 调用 binvox ----
        # 通过 xvfb-run 提供虚拟显示 (binvox 使用 OpenGL 深度雕刻)
        # 不使用 -pb (pbuffer), 改用窗口模式, Xvfb 提供虚拟窗口
        # LD_PRELOAD Mesa libGL 避免 Singularity 注入的 NVIDIA libGL 与 Xvfb 冲突
        binvox_cmd = (
            f'{BINVOX_PATH} -aw -dc '
            f'-d {binvox_dim} '
            f'-bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} '
            f'{meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} '
            f'-t binvox {tmp_obj}'
        )
        full_cmd = f'xvfb-run -a -s "-screen 0 1024x768x24" {binvox_cmd}'

        env = os.environ.copy()
        env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGL.so.1'
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'

        result = subprocess.run(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            env=env,
        )

        if not os.path.exists(binvox_file):
            raise RuntimeError(
                f"binvox 未生成输出文件。\n"
                f"命令: {full_cmd}\n"
                f"返回码: {result.returncode}\n"
                f"stderr: {result.stderr.decode('utf-8', errors='replace')[-500:]}")

        # ---- 读取 binvox 输出 ----
        with open(binvox_file, 'rb') as f:
            m1 = binvox_rw.read_as_3d_array(f)

        # ---- 过滤用的 mesh 原始边界 ----
        bbox = mesh_ori.bounds
        lower_bound = bbox[0, :]
        upper_bound = bbox[1, :]

        # ---- 体素 → 粒子坐标 (与 Physion utils_geom.py:180-182 完全一致) ----
        x, y, z = np.nonzero(m1.data)
        adjusted_spacing = (maxDim_spaced * spacing + 2 * spacingEps_p) / maxDim_spaced
        if len(x) > 0:
            points = (np.expand_dims(meshLower_spaced, 0)
                      + np.stack([(x + 0.5) * adjusted_spacing,
                                  (y + 0.5) * adjusted_spacing,
                                  (z + 0.5) * adjusted_spacing], axis=1))
        else:
            points = np.zeros((0, 3))

        n_before_filter = len(points)

        # ---- 过滤边界外粒子 (复制 Physion utils_geom.py:186-196) ----
        if len(points) > 0:
            idx = (points[:, 0] - upper_bound[0] <= 0) * (points[:, 0] - lower_bound[0] >= 0)
            idy = (points[:, 1] - upper_bound[1] <= 0) * (points[:, 1] - lower_bound[1] >= 0)
            idz = (points[:, 2] - upper_bound[2] <= 0) * (points[:, 2] - lower_bound[2] >= 0)
            points = points[idx * idy * idz]

        n_after_filter = len(points)
        print(f"    binvox 结果: {n_before_filter} 体素粒子, 过滤后 {n_after_filter} 粒子")

    finally:
        # ---- 清理临时文件 ----
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    info = {
        'n_mesh_vertices': n_vertices,
        'n_mesh_faces': n_faces,
        'binvox_dim': binvox_dim,
        'n_voxels_before_filter': n_before_filter,
        'n_voxels_after_filter': n_after_filter,
        'n_final_particles': len(points),
        'source': 'binvox',
    }

    return points.astype(np.float32), info


def save_mesh_ply(mesh_result, path):
    """
    将 MeshExtractResult (PyTorch tensor) 保存为标准 mesh PLY

    Args:
        mesh_result: MeshExtractResult 对象 (vertices, faces, vertex_attrs)
        path: 输出路径
    """
    import torch
    verts = mesh_result.vertices.float().cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()

    # vertex_attrs 可能包含颜色
    colors = None
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy()
        if attrs.shape[1] >= 3:
            colors = attrs[:, :3]
            colors = (colors * 255).clip(0, 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=colors, process=False)
    mesh.export(path)


def transform_mesh_to_scene(mesh_result, output_dict):
    """
    将物体局部 mesh 变换到场景坐标系

    直接应用 pose (scale, rotation, translation)，无需额外的 Z-up→Y-up 旋转，
    因为 pose 的旋转矩阵 R 已经包含了坐标系转换 (R[1,:] ≈ [0,0,1])。
    PyTorch3D 行向量惯例: (p * s) @ R + t

    Args:
        mesh_result: MeshExtractResult 对象
        output_dict: SAM 3D inference 输出字典 (含 rotation, translation, scale)

    Returns:
        verts_scene: (V, 3) np.float32 场景坐标中的顶点
        faces: (F, 3) np.int64 面片索引
    """
    import torch

    verts = mesh_result.vertices.float().cpu().numpy()  # (V, 3)
    faces = mesh_result.faces.cpu().numpy()

    # 应用 pose (不做 Z-up→Y-up: pose 旋转矩阵已包含坐标系转换)
    scale = output_dict["scale"]        # (1, 3) tensor
    rotation = output_dict["rotation"]  # (1, 4) tensor, 四元数 (w,x,y,z)
    translation = output_dict["translation"]  # (1, 3) tensor

    s = scale[0].cpu().numpy().astype(np.float32)
    q = rotation[0].cpu().numpy().astype(np.float32)
    t = translation[0].cpu().numpy().astype(np.float32)

    R = _quaternion_to_rotation_matrix(q)

    # PyTorch3D 行向量惯例: (p * s) @ R + t
    verts_scene = (verts * s) @ R + t

    return verts_scene.astype(np.float32), faces


def _quaternion_to_rotation_matrix(quat):
    """
    四元数转旋转矩阵 (PyTorch3D 格式: w, x, y, z)

    与 gs_to_particles.py 中的 quaternion_to_rotation_matrix() 相同
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def _load_object_names(scene_dir, n_objects):
    """从 detection_result.json 加载物体名称 (复用 gs_to_particles 逻辑)"""
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


def _compute_scene_scale_factor(mesh_list, max_object_extent=1.5):
    """
    [DEPRECATED] 已被 _compute_scale_by_particle_count() 替代。
    保留供外部脚本可能引用。

    计算场景缩放因子，确保最大物体的 AABB 最大轴不超过 max_object_extent。

    策略: 找到最大物体，将其缩放到 max_object_extent，所有物体等比缩放。
    这样保证最大物体的粒子数在 Physion 训练分布内 (400-2000 粒子/物体)，
    同时保持所有物体的相对大小不变。

    Physion 训练数据中物体典型大小: 0.2-1.0 单位
    1.5 单位物体在 spacing=0.05 下约 1000-3000 粒子，处于训练分布中上段。

    如果最大物体已在范围内，不缩放 (scale=1.0)。

    Args:
        mesh_list: list of trimesh.Trimesh，所有物体的 mesh (场景坐标)
        max_object_extent: 最大物体允许的 AABB 最大轴上限 (默认 1.5)

    Returns:
        scale_factor: float, 缩放因子 (<=1.0 表示缩小, 1.0 表示不缩放)
        max_extent_found: float, 原始最大物体的 AABB 最大轴长度
    """
    max_extent_found = 0.0
    for mesh in mesh_list:
        extents = mesh.bounding_box.extents  # [dx, dy, dz]
        obj_max = float(extents.max())
        if obj_max > max_extent_found:
            max_extent_found = obj_max

    if max_extent_found <= max_object_extent:
        return 1.0, max_extent_found

    scale_factor = max_object_extent / max_extent_found
    return scale_factor, max_extent_found


def _compute_scale_by_particle_count(particle_counts,
                                      max_particles=600,
                                      min_particles=20,
                                      target_particles=350):
    """
    根据最大物体粒子数计算场景缩放因子。

    规则:
      - 最大物体 > max_particles → 缩小 (目标 max_particles)
      - 最大物体 < min_particles → 放大 (目标 target_particles)
      - 否则不缩放

    粒子数 ∝ 体积 ∝ scale³，因此 scale = (target / current)^(1/3)

    Args:
        particle_counts: list[int], 各物体粒子数
        max_particles: 最大物体粒子数上限 (超过则缩小)
        min_particles: 最大物体粒子数下限 (低于则放大)
        target_particles: 放大时目标粒子数

    Returns:
        scale_factor: float (>1.0 放大, <1.0 缩小, 1.0 不缩放)
        reason: str (描述缩放原因)
    """
    max_count = max(particle_counts)
    if max_count == 0:
        return 1.0, f"所有物体 0 粒子, 无法缩放 (可能是极薄物体)"
    if max_count > max_particles:
        scale = (max_particles / max_count) ** (1.0 / 3.0)
        return scale, f"缩小 (最大物体 {max_count} 粒子 > 上限 {max_particles})"
    elif max_count < min_particles:
        scale = (target_particles / max_count) ** (1.0 / 3.0)
        return scale, f"放大 (最大物体 {max_count} 粒子 < 下限 {min_particles})"
    else:
        return 1.0, f"无需缩放 (最大物体 {max_count} 粒子, 范围 {min_particles}~{max_particles})"


def mesh_scene_to_particles(scene_dir, spacing=0.05,
                             discard_threshold=10,
                             max_particles_per_obj=600,
                             min_particles_per_obj=20,
                             target_particles=350,
                             floor_y_scene=None,
                             ground_plane=None,
                             thicken=False,
                             thicken_min_layers=3):
    """
    将整个场景的 mesh 文件转换为 Physion 格式粒子

    使用 binvox -dc (depth carving) 体素化, 与 Physion 训练数据预处理完全一致。
    基于粒子数量自适应缩放场景:
      - 最大物体 > max_particles_per_obj → 等比缩小所有物体
      - 最大物体 < min_particles_per_obj → 等比放大所有物体到 target_particles
      - 缩放后粒子数 < discard_threshold 的物体丢弃

    输入: scene_dir 中的 object_{i}_mesh_posed.ply 文件
    输出: 与 gs_scene_to_particles() 完全相同的格式

    Args:
        scene_dir: auto_scene_pipeline.py 的输出目录
        spacing: 目标粒子间距 (默认 0.05)
        discard_threshold: 低于此粒子数直接丢弃物体 (默认 10)
        max_particles_per_obj: 最大物体粒子数上限 (默认 600)
        min_particles_per_obj: 最大物体粒子数下限, 低于则放大 (默认 20)
        target_particles: 放大时目标粒子数 (默认 350)
        floor_y_scene: MoGe 估计的地板 Y 坐标 (scene_posed 空间, 可选, 向后兼容)
        ground_plane: RANSAC 拟合的地面平面参数 (scene_posed 空间, 可选)
                      {"normal": [nx, ny, nz], "d": float}

    Returns:
        particles_list: list of (N_i, 3), 每物体粒子 (Physion 坐标)
        obj_points: list of (N_i, 3), 每物体局部坐标
        centroids: list of (3,), 每物体质心
        object_names: list of str, 物体名称
        scene_info: dict, 场景信息
    """
    if trimesh is None:
        raise ImportError("需要安装 trimesh: pip install trimesh")

    # 检查 binvox 可执行文件
    if not os.path.isfile(BINVOX_PATH):
        raise FileNotFoundError(
            f"binvox 可执行文件不存在: {BINVOX_PATH}\n"
            f"请确认 binvox 已正确安装。")

    # 查找 mesh 文件
    mesh_files = sorted(glob.glob(os.path.join(scene_dir, "object_*_mesh_posed.ply")))
    n_objects = len(mesh_files)

    if n_objects == 0:
        raise FileNotFoundError(
            f"在 {scene_dir} 中未找到 object_*_mesh_posed.ply 文件。"
            f"请确认 auto_scene_pipeline.py 已保存 mesh 输出。")

    object_names = _load_object_names(scene_dir, n_objects)
    print(f"\n使用 Mesh + binvox 模式: {n_objects} 个物体 (Physion 训练方式)")

    # 预加载所有 mesh
    meshes_original = []
    for mesh_path in mesh_files:
        mesh = load_mesh_ply(mesh_path)
        meshes_original.append(mesh)

    # --- 粒子数自适应缩放 (迭代式) ---
    scene_scale_factor = 1.0
    max_iterations = 3
    trial_results = None

    for iteration in range(max_iterations):
        # 准备当前缩放的 mesh
        meshes = [copy.deepcopy(m) for m in meshes_original]
        if scene_scale_factor != 1.0:
            for mesh in meshes:
                mesh.vertices *= scene_scale_factor

        # 体素化所有物体
        trial_counts = []
        trial_results = []
        for i, (mesh, mesh_path) in enumerate(zip(meshes, mesh_files)):
            name = object_names[i] if i < len(object_names) else f'object_{i}'
            print(f"  物体 {i} ({name}): {os.path.basename(mesh_path)}")
            print(f"    mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            # --- 薄物体加厚 (不影响原有逻辑) ---
            if thicken:
                from mesh_thicken import thicken_if_needed
                mesh, _thicken_info = thicken_if_needed(mesh, spacing=spacing, min_layers=thicken_min_layers)
            # --- 薄物体加厚结束 ---
            particles, info = mesh_to_solid_particles(mesh, spacing=spacing)
            trial_counts.append(len(particles))
            trial_results.append((particles, info))
            print(f"    → {len(particles)} 粒子 (binvox_dim={info.get('binvox_dim', '?')})")

        # 判断是否需要缩放
        scale, reason = _compute_scale_by_particle_count(
            trial_counts,
            max_particles=max_particles_per_obj,
            min_particles=min_particles_per_obj,
            target_particles=target_particles,
        )

        print(f"\n  [迭代 {iteration + 1}/{max_iterations}] "
              f"各物体粒子数: {trial_counts}, 最大={max(trial_counts)}")
        print(f"    {reason}")

        if scale == 1.0:
            # 无需缩放，使用当前结果
            break

        # 需要缩放: 更新累计缩放因子，下一轮重新体素化
        scene_scale_factor *= scale
        print(f"    本轮缩放 {scale:.4f}x, 累计缩放因子: {scene_scale_factor:.4f}")

    print(f"\n  最终缩放因子: {scene_scale_factor:.4f}")

    # 使用最终 trial_results 构建粒子列表
    raw_particles_list = []
    object_infos = []
    discarded_indices = []

    for i, (particles, info) in enumerate(trial_results):
        if len(particles) < discard_threshold:
            print(f"  物体 {i} ({object_names[i]}): "
                  f"{len(particles)} 粒子 — 丢弃 (< {discard_threshold})")
            discarded_indices.append(i)
            object_infos.append(info)
            continue

        raw_particles_list.append(particles)
        object_infos.append(info)

    if not raw_particles_list:
        raise RuntimeError("所有物体粒子数不足，全部被丢弃")

    # 清理已丢弃物体
    if discarded_indices:
        print(f"\n  丢弃了 {len(discarded_indices)} 个极小物体: "
              f"{[object_names[i] for i in discarded_indices]}")
        object_names = [n for i, n in enumerate(object_names) if i not in discarded_indices]
        object_infos = [info for i, info in enumerate(object_infos) if i not in discarded_indices]

    # 坐标对齐到 Physion (地面→Y=0, 居中 XZ)
    # 复用 gs_to_particles.py 的 align_to_physion_coords()
    from gs_to_particles import align_to_physion_coords

    # 确定 ground_y: 优先用 RANSAC 平面在场景中心处的 Y 值
    external_ground_y = None
    if ground_plane is not None:
        # 物体已对齐到平面，用场景中心的平面 Y 作为全局参考
        all_pts = np.concatenate(raw_particles_list, axis=0)
        center_x = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        center_z = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
        normal = np.array(ground_plane["normal"])
        d_scaled = ground_plane["d"] * scene_scale_factor
        ny = normal[1]
        if abs(ny) > 1e-6:
            external_ground_y = -(normal[0] * center_x + normal[2] * center_z + d_scaled) / ny
            print(f"\n  地面参考 Y (平面在场景中心): {external_ground_y:.4f}")
    elif floor_y_scene is not None:
        # 回退: 使用旧的全局 floor_y
        external_ground_y = floor_y_scene * scene_scale_factor
        print(f"\n  使用 MoGe 地板 Y (旧方法): {floor_y_scene:.4f} * scale {scene_scale_factor:.4f} = {external_ground_y:.4f}")

    print(f"\n坐标系对齐到 Physion 仿真坐标系...")
    aligned_list, coord_scale, coord_transform = align_to_physion_coords(
        raw_particles_list, ground_y=external_ground_y, target_spacing=spacing)

    # 计算质心和局部参考
    centroids = []
    obj_points = []
    for pts in aligned_list:
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        obj_points.append(pts - centroid)

    total_particles = sum(len(p) for p in aligned_list)

    scene_info = {
        'n_objects': len(aligned_list),
        'n_particles': total_particles,
        'object_names': object_names,
        'object_infos': object_infos,
        'coord_scale': float(coord_scale),
        'coord_transform': coord_transform,
        'centroids': [c.tolist() for c in centroids],
        'particles_per_object': [len(p) for p in aligned_list],
        'mode': 'mesh',
        'discarded_indices': discarded_indices,
        'scene_scale_factor': float(scene_scale_factor),
        'max_particles_per_obj': max_particles_per_obj,
        'min_particles_per_obj': min_particles_per_obj,
        'target_particles': target_particles,
    }

    print(f"\n场景汇总: {scene_info['n_objects']} 个物体, 共 {total_particles} 个粒子")
    for i, name in enumerate(object_names):
        print(f"  物体 {i} ({name}): {len(aligned_list[i])} 粒子, 质心 {centroids[i]}")

    return aligned_list, obj_points, centroids, object_names, scene_info


# ============================================================
# 粒子 PLY 可视化保存 (与 gs_to_particles.save_particles_ply 相同)
# ============================================================

def save_particles_ply(path, all_points, particles_list=None):
    """保存粒子点云为 PLY 文件 (用于可视化)"""
    n = len(all_points)

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


def main():
    parser = argparse.ArgumentParser(description='Mesh → Physion 粒子转换 (binvox 方式)')
    parser.add_argument('--scene-dir', type=str, required=True,
                        help='auto_scene_pipeline.py 的输出目录 (含 object_*_mesh_posed.ply)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: scene_dir/particles_mesh)')
    parser.add_argument('--spacing', type=float, default=0.05,
                        help='目标粒子间距 (Physion 默认 0.05)')
    parser.add_argument('--discard-threshold', type=int, default=100,
                        help='低于此值直接丢弃物体')
    parser.add_argument('--max-particles-per-obj', type=int, default=800,
                        help='单物体最大粒子数上限')
    parser.add_argument('--min-particles-per-obj', type=int, default=300,
                        help='最大物体粒子数下限，低于则放大')
    parser.add_argument('--target-particles', type=int, default=650,
                        help='放大时目标粒子数')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.scene_dir, 'particles_mesh')
    os.makedirs(args.output_dir, exist_ok=True)

    particles_list, obj_points, centroids, names, info = mesh_scene_to_particles(
        scene_dir=args.scene_dir,
        spacing=args.spacing,
        discard_threshold=args.discard_threshold,
        max_particles_per_obj=args.max_particles_per_obj,
        min_particles_per_obj=args.min_particles_per_obj,
        target_particles=args.target_particles,
    )

    # 保存粒子
    for i, (pts, obj_pts, name) in enumerate(zip(particles_list, obj_points, names)):
        np.save(os.path.join(args.output_dir, f'particles_{i}.npy'), pts)
        np.save(os.path.join(args.output_dir, f'obj_points_{i}.npy'), obj_pts)

    # 保存场景信息
    with open(os.path.join(args.output_dir, 'scene_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # 保存合并粒子 PLY
    all_pts = np.concatenate(particles_list, axis=0)
    save_particles_ply(os.path.join(args.output_dir, 'all_particles.ply'),
                       all_pts, particles_list)

    print(f"\n输出已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
