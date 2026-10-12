"""
mesh_thicken.py - 薄物体 Mesh 自动加厚模块

对手机等极薄物体，在 binvox 体素化之前沿最薄方向膨胀 mesh，
使其至少产生 min_layers 层粒子，避免单层粒子穿透地面。

核心思路:
  1. PCA 检测最薄主轴方向
  2. 如果该方向跨度 < min_layers * spacing，判定为薄物体
  3. 双层偏移 + 侧面封口，构建加厚后的 mesh
"""

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


def detect_thin_axis(mesh, spacing=0.05, min_layers=3):
    """
    检测 mesh 是否为薄物体

    通过 PCA 找到方差最小的主轴方向，判断该方向跨度是否不足 min_layers 层粒子。

    Args:
        mesh: trimesh.Trimesh 对象
        spacing: 粒子间距 (默认 0.05)
        min_layers: 最少需要的粒子层数 (默认 3)

    Returns:
        dict:
            is_thin: bool, 是否为薄物体
            thin_axis: (3,) ndarray, 最薄方向的单位向量
            thin_span: float, 最薄方向的跨度
            target_thickness: float, 目标厚度 (min_layers * spacing)
            eigenvalues: (3,) ndarray, PCA 三个特征值（从小到大）
    """
    vertices = mesh.vertices  # (N, 3)
    min_thickness = min_layers * spacing

    # PCA: 计算协方差矩阵的特征值分解
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = np.cov(centered, rowvar=False)  # (3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh 返回升序特征值，所以 eigenvalues[0] 最小，eigenvectors[:, 0] 是最薄方向

    thin_axis = eigenvectors[:, 0]  # 最小特征值对应的特征向量
    thin_axis = thin_axis / np.linalg.norm(thin_axis)  # 确保单位向量

    # 计算最薄方向的跨度
    projections = centered @ thin_axis
    thin_span = projections.max() - projections.min()

    is_thin = thin_span < min_thickness

    return {
        'is_thin': is_thin,
        'thin_axis': thin_axis,
        'thin_span': float(thin_span),
        'target_thickness': float(min_thickness),
        'eigenvalues': eigenvalues,
    }


def thicken_mesh(mesh, thin_info):
    """
    对薄物体 mesh 沿 thin_axis 方向加厚

    方法: 双层偏移 + 侧面封口
      - 顶层: 原始顶点沿 thin_axis 正方向偏移
      - 底层: 原始顶点沿 thin_axis 负方向偏移，面片翻转
      - 侧面: 边界边用四边形连接上下两层

    Args:
        mesh: trimesh.Trimesh 对象
        thin_info: detect_thin_axis() 的返回值

    Returns:
        trimesh.Trimesh: 加厚后的 mesh
    """
    if trimesh is None:
        raise ImportError("需要安装 trimesh: pip install trimesh")

    thin_axis = thin_info['thin_axis']
    thin_span = thin_info['thin_span']
    target_thickness = thin_info['target_thickness']

    vertices = mesh.vertices.copy()  # (N, 3)
    faces = mesh.faces.copy()  # (F, 3)
    n_verts = len(vertices)

    # 计算偏移量: 使总厚度达到 target_thickness
    if thin_span < 1e-8:
        # 完全平面 (如手机，thin_span ≈ 0)
        half_offset = target_thickness / 2.0
    else:
        half_offset = (target_thickness - thin_span) / 2.0

    # 顶层顶点: 沿 thin_axis 正方向偏移
    top_vertices = vertices + thin_axis * half_offset
    # 底层顶点: 沿 thin_axis 负方向偏移
    bottom_vertices = vertices - thin_axis * half_offset

    # 顶层面片: 保持原始法线方向
    top_faces = faces.copy()
    # 底层面片: 索引偏移 n_verts，翻转顶点顺序（法线朝外）
    bottom_faces = faces[:, ::-1] + n_verts

    # 侧面封口: 找到边界边，用四边形连接上下两层
    side_faces = _build_side_faces(mesh, n_verts)

    # 合并
    all_vertices = np.vstack([top_vertices, bottom_vertices])
    all_faces = np.vstack([top_faces, bottom_faces] +
                          ([side_faces] if len(side_faces) > 0 else []))

    thickened = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
    return thickened


def _build_side_faces(mesh, n_verts):
    """
    构建侧面封口面片

    找到 mesh 的边界边 (只属于一个面的边)，
    用两个三角形连接上下两层对应的顶点。

    Args:
        mesh: 原始 trimesh.Trimesh
        n_verts: 顶层顶点数 (= 底层偏移量)

    Returns:
        (M, 3) ndarray: 侧面三角形面片索引
    """
    # 提取边界边
    edges = mesh.edges_sorted  # (E, 2), 每条边的两个顶点索引
    # 统计每条边出现的次数（在面中）
    edge_tuples = [tuple(e) for e in edges]
    from collections import Counter
    edge_counts = Counter(edge_tuples)
    boundary_edges = [e for e, count in edge_counts.items() if count == 1]

    if not boundary_edges:
        # 无边界边 (水密 mesh)，用 trimesh 的 outline 尝试
        try:
            boundary_edges_arr = mesh.outline().entities[0].points
            # outline 返回的是有序顶点索引，转为边
            boundary_edges = []
            for i in range(len(boundary_edges_arr)):
                v0 = boundary_edges_arr[i]
                v1 = boundary_edges_arr[(i + 1) % len(boundary_edges_arr)]
                boundary_edges.append((v0, v1))
        except Exception:
            return np.zeros((0, 3), dtype=np.int64)

    if not boundary_edges:
        return np.zeros((0, 3), dtype=np.int64)

    # 每条边界边生成两个三角形 (四边形分为两个三角形)
    # 顶层: v0_top, v1_top  底层: v0_bot, v1_bot
    side_faces = []
    for v0, v1 in boundary_edges:
        v0_top, v1_top = v0, v1
        v0_bot, v1_bot = v0 + n_verts, v1 + n_verts
        # 三角形 1: v0_top, v1_top, v1_bot
        side_faces.append([v0_top, v1_top, v1_bot])
        # 三角形 2: v0_top, v1_bot, v0_bot
        side_faces.append([v0_top, v1_bot, v0_bot])

    return np.array(side_faces, dtype=np.int64)


def thicken_if_needed(mesh, spacing=0.05, min_layers=3):
    """
    便捷入口: 检测薄物体并自动加厚

    Args:
        mesh: trimesh.Trimesh 对象
        spacing: 粒子间距 (默认 0.05)
        min_layers: 最少粒子层数 (默认 3)

    Returns:
        mesh: 原始或加厚后的 trimesh.Trimesh
        thin_info: detect_thin_axis() 的返回值
    """
    thin_info = detect_thin_axis(mesh, spacing=spacing, min_layers=min_layers)

    if thin_info['is_thin']:
        print(f"    [加厚] 检测到薄物体: "
              f"最薄方向跨度 {thin_info['thin_span']:.4f} < 目标厚度 {thin_info['target_thickness']:.4f}")
        print(f"    [加厚] 薄轴方向: [{thin_info['thin_axis'][0]:.3f}, "
              f"{thin_info['thin_axis'][1]:.3f}, {thin_info['thin_axis'][2]:.3f}]")
        thickened = thicken_mesh(mesh, thin_info)
        print(f"    [加厚] 完成: {len(mesh.vertices)} → {len(thickened.vertices)} 顶点, "
              f"{len(mesh.faces)} → {len(thickened.faces)} 面片")
        return thickened, thin_info
    else:
        print(f"    [加厚] 无需加厚: 最薄方向跨度 {thin_info['thin_span']:.4f} "
              f">= 目标厚度 {thin_info['target_thickness']:.4f}")
        return mesh, thin_info
