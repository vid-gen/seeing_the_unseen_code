"""
Mesh 线框投影可视化模块

将 3D mesh 线框按刚体变换投影到 2D 相机视角，叠加在原始图像上。
用于替代粒子散点在物体发生大角度旋转时的可视化不足。

独立于现有的 render_camera_frame / create_camera_view_gif，
通过 --wireframe 参数可选启用。

输出: camera_view_wireframe.gif + 关键帧 PNG
"""

import os
import glob
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# 与 visualize_rollout.py 保持一致的颜色列表
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


def calc_rigid_transform(source, target):
    """
    SVD 计算最优刚体变换 R, t 使得 target ≈ R @ source + t

    Args:
        source: (N, 3) 源点集
        target: (N, 3) 目标点集

    Returns:
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # 中心化
    centroid_s = source.mean(axis=0)
    centroid_t = target.mean(axis=0)
    s_centered = source - centroid_s
    t_centered = target - centroid_t

    # 协方差矩阵
    H = s_centered.T @ t_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # 保证右手系 (det(R) = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])

    R = Vt.T @ sign_matrix @ U.T
    t = centroid_t - R @ centroid_s

    return R, t


def transform_mesh_vertices(vertices, R, t):
    """
    应用刚体变换: v_new = R @ v + t

    Args:
        vertices: (V, 3)
        R: (3, 3)
        t: (3,)
    Returns:
        (V, 3) 变换后顶点
    """
    return (R @ vertices.T).T + t


def _simplify_mesh_o3d(vertices, faces, target_faces):
    """用 Open3D quadric decimation 简化 mesh"""
    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    simplified = o3d_mesh.simplify_quadric_decimation(target_faces)
    return (np.asarray(simplified.vertices),
            np.asarray(simplified.triangles))


def _simplify_mesh_pymeshlab(vertices, faces, target_faces):
    """用 pymeshlab 简化 mesh (回退方案)"""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(m)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    out = ms.current_mesh()
    return out.vertex_matrix(), out.face_matrix()


def _simplify_mesh(vertices, faces, target_faces):
    """简化 mesh, 优先 Open3D, 回退 pymeshlab"""
    if faces.shape[0] <= target_faces:
        return vertices, faces

    try:
        return _simplify_mesh_o3d(vertices, faces, target_faces)
    except Exception as e:
        print(f"    Open3D 简化失败 ({e}), 尝试 pymeshlab...")

    try:
        return _simplify_mesh_pymeshlab(vertices, faces, target_faces)
    except Exception as e:
        print(f"    pymeshlab 简化也失败 ({e}), 使用随机采样面片...")

    # 最终回退: 随机采样面片
    rng = np.random.RandomState(42)
    idx = rng.choice(faces.shape[0], target_faces, replace=False)
    return vertices, faces[idx]


def _extract_edges_from_faces(faces):
    """从面片索引提取唯一边"""
    edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
    # 去重: 排序每条边使 (a,b) 和 (b,a) 统一
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def load_and_simplify_meshes(scene_dir, coord_transform_path,
                             target_faces=3000):
    """
    加载所有 object_*_mesh_posed.ply，简化并转换到 Physion 空间。

    Args:
        scene_dir: 3DGS 场景目录
        coord_transform_path: coord_transform.json 路径
        target_faces: 简化目标面数

    Returns:
        meshes_physion: list of dict, 每个含:
            'vertices': (V, 3) Physion 空间顶点
            'faces': (F, 3) 面片索引
            'edges': (E, 2) 边索引
            'name': str 物体名称
        如果没有 mesh 文件则返回 None
    """
    import trimesh

    mesh_files = sorted(glob.glob(
        os.path.join(scene_dir, "object_*_mesh_posed.ply")))

    if not mesh_files:
        print("  警告: 未找到 object_*_mesh_posed.ply, 跳过线框渲染")
        return None

    # 加载坐标变换参数
    with open(coord_transform_path, 'r') as f:
        ct = json.load(f)
    offset = np.array([ct['center_x'], ct['ground_y'], ct['center_z']])
    coord_scale = ct['scale']  # align_to_physion 的缩放 (通常 1.0)
    scene_scale_factor = ct.get('scene_scale_factor', 1.0)
    gc = ct.get('ground_clearance', 0.0)

    # 尝试加载物体名称
    det_path = os.path.join(scene_dir, 'detection_result.json')
    obj_names = []
    if os.path.exists(det_path):
        with open(det_path, 'r') as f:
            det = json.load(f)
        for obj in det.get('objects', []):
            obj_names.append(obj.get('label', obj.get('name', '')))

    meshes = []
    for i, mf in enumerate(mesh_files):
        print(f"    加载 mesh {i}: {os.path.basename(mf)}...")
        mesh = trimesh.load_mesh(mf, process=True)
        if hasattr(mesh, 'geometry'):
            # Scene 类型 → 合并
            mesh = trimesh.util.concatenate(mesh.geometry.values())

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        print(f"      原始: {verts.shape[0]} 顶点, {faces.shape[0]} 面片")

        # 简化
        verts_s, faces_s = _simplify_mesh(verts, faces, target_faces)
        print(f"      简化: {verts_s.shape[0]} 顶点, {faces_s.shape[0]} 面片")

        # 转换到 Physion 空间
        # 正确变换链: 先缩放(与粒子生成一致), 再减偏移(offset在缩放后空间中计算)
        verts_physion = (verts_s * scene_scale_factor - offset) * coord_scale
        verts_physion[:, 1] += gc

        # 多米诺方向旋转（与 build_physion_input 中的旋转一致）
        domino_R = ct.get('domino_rotation_matrix', None)
        if domino_R is not None:
            R = np.array(domino_R, dtype=np.float64)
            verts_physion = verts_physion @ R.T

        # 提取边
        edges = _extract_edges_from_faces(faces_s)
        print(f"      边数: {edges.shape[0]}")

        name = obj_names[i] if i < len(obj_names) else f"object_{i}"

        meshes.append({
            'vertices': verts_physion.astype(np.float64),
            'faces': faces_s,
            'edges': edges,
            'name': name,
        })

    # 加载地面 mesh (如果存在)
    ground_mesh_path = os.path.join(scene_dir, 'ground_mesh_posed.ply')
    if os.path.exists(ground_mesh_path):
        print(f"    加载地面 mesh: {os.path.basename(ground_mesh_path)}...")
        g_mesh = trimesh.load_mesh(ground_mesh_path, process=True)
        if hasattr(g_mesh, 'geometry'):
            g_mesh = trimesh.util.concatenate(g_mesh.geometry.values())

        g_verts = np.asarray(g_mesh.vertices, dtype=np.float64)
        g_faces = np.asarray(g_mesh.faces, dtype=np.int64)
        print(f"      原始: {g_verts.shape[0]} 顶点, {g_faces.shape[0]} 面片")

        # 简化 (地面 mesh 通常较小，一般不需要简化)
        g_verts_s, g_faces_s = _simplify_mesh(g_verts, g_faces, target_faces)
        if g_faces_s.shape[0] != g_faces.shape[0]:
            print(f"      简化: {g_verts_s.shape[0]} 顶点, {g_faces_s.shape[0]} 面片")

        # 转换到 Physion 空间 (与物体 mesh 相同的变换链)
        g_verts_physion = (g_verts_s * scene_scale_factor - offset) * coord_scale
        g_verts_physion[:, 1] += gc

        g_edges = _extract_edges_from_faces(g_faces_s)
        print(f"      边数: {g_edges.shape[0]}")

        meshes.append({
            'vertices': g_verts_physion.astype(np.float64),
            'faces': g_faces_s,
            'edges': g_edges,
            'name': 'ground',
            'is_static': True,
        })

    return meshes


def _compute_face_normals(vertices, faces):
    """计算面法线"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return normals / norms


def project_edges_to_2d(vertices_3d, edges, faces, camera,
                        backface_cull=True):
    """
    将 3D mesh 边投影到 2D 像素坐标。

    Args:
        vertices_3d: (V, 3) Physion 空间顶点
        edges: (E, 2) 边索引
        faces: (F, 3) 面片索引
        camera: CameraInPhysion 实例
        backface_cull: 是否启用背面剔除

    Returns:
        segments_2d: list of [(u1,v1), (u2,v2)], 可见边的 2D 端点
    """
    W, H = camera.image_width, camera.image_height

    if backface_cull and faces is not None:
        # 计算面法线
        face_normals = _compute_face_normals(vertices_3d, faces)

        # 面中心
        face_centers = vertices_3d[faces].mean(axis=1)

        # 相机位置在 Physion 空间
        cam_pos = camera.position

        # 视线方向: 相机 → 面中心
        view_dirs = cam_pos - face_centers
        norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        view_dirs = view_dirs / norms

        # 面向相机: dot(normal, view_dir) > 0
        dots = np.einsum('ij,ij->i', face_normals, view_dirs)
        visible_mask = dots > 0

        # 提取可见面的边
        visible_faces = faces[visible_mask]
        visible_edges = _extract_edges_from_faces(visible_faces)
    else:
        visible_edges = edges

    if len(visible_edges) == 0:
        return []

    # 投影所有相关顶点
    unique_verts = np.unique(visible_edges.flatten())
    vert_map = {v: i for i, v in enumerate(unique_verts)}

    uv, depth = camera.project(vertices_3d[unique_verts], return_depth=True)

    # 构建 2D 线段
    segments = []
    for e in visible_edges:
        i0, i1 = vert_map[e[0]], vert_map[e[1]]
        # 两端点深度都 > 0 (在相机前方)
        if depth[i0] <= 0 or depth[i1] <= 0:
            continue
        u0, v0 = uv[i0]
        u1, v1 = uv[i1]
        # 至少一个端点在图像内
        in0 = (0 <= u0 < W) and (0 <= v0 < H)
        in1 = (0 <= u1 < W) and (0 <= v1 < H)
        if not in0 and not in1:
            continue
        segments.append([(u0, v0), (u1, v1)])

    return segments


def render_wireframe_frame(ax, meshes_physion, positions,
                           instance_idx, camera,
                           orig_image=None, object_names=None,
                           line_width=0.5, alpha=0.7,
                           backface_cull=True,
                           step_idx=0, n_steps=1,
                           show_particles=False, particle_size=3,
                           particle_alpha=0.3):
    """
    渲染单帧: mesh 线框投影叠加在原图上。

    Args:
        ax: matplotlib axes
        meshes_physion: load_and_simplify_meshes() 输出
        positions: (n_particles, 3) 当前帧粒子位置 (Physion)
        instance_idx: 物体粒子索引边界列表
        camera: CameraInPhysion 实例
        orig_image: 原始图像 (H, W, 3)
        object_names: 物体名称列表
        line_width: 线框线宽
        alpha: 线框透明度
        backface_cull: 背面剔除
        step_idx: 当前步骤
        n_steps: 总步骤
        show_particles: 是否同时显示粒子散点
        particle_size: 粒子散点大小
        particle_alpha: 粒子散点透明度
    """
    W, H = camera.image_width, camera.image_height

    if orig_image is not None:
        ax.imshow(orig_image, extent=[0, W, H, 0])
    else:
        ax.set_facecolor('white')

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    n_objects = len(meshes_physion)
    instance_idx = list(instance_idx)

    # 获取 step 0 的粒子 (用于计算 R,T)
    # positions_list[0] 需要外部传入，这里用 instance_idx 对齐

    for obj_id in range(min(n_objects, len(instance_idx) - 1)):
        color = OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]
        mesh_info = meshes_physion[obj_id]

        st = instance_idx[obj_id]
        ed = instance_idx[obj_id + 1]
        obj_pts = positions[st:ed]

        # mesh 顶点已在 step 0 时与粒子对齐
        # 通过 R,T 变换到当前步的位置
        # R,T 在外部已经应用（见 create_wireframe_gif）
        # 这里 mesh_info['vertices'] 已经是当前帧的位置

        segments = project_edges_to_2d(
            mesh_info['vertices'],
            mesh_info['edges'],
            mesh_info['faces'],
            camera,
            backface_cull=backface_cull,
        )

        if segments:
            lc = LineCollection(segments, colors=color,
                                linewidths=line_width, alpha=alpha)
            ax.add_collection(lc)

        # 可选: 叠加粒子散点
        if show_particles and len(obj_pts) > 0:
            uv, depth, visible = camera.project_with_visibility(obj_pts)
            if len(uv) > 0:
                ax.scatter(uv[:, 0], uv[:, 1], c=color,
                           s=particle_size, alpha=particle_alpha,
                           edgecolors='none')

        # 图例标签
        name = object_names[obj_id] if object_names and obj_id < len(object_names) else mesh_info['name']
        # 在右上角显示物体名称
        ax.text(W - 10, 20 + obj_id * 20, f"{name}",
                color=color, fontsize=8, ha='right', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.7, edgecolor='none'))

    # 帧计数器
    ax.text(10, 20, f"Step {step_idx} / {n_steps - 1}",
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='black', alpha=0.6, edgecolor='none'))


def _save_gif(frames, output_path, fps=6):
    """保存帧列表为 GIF"""
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def create_wireframe_gif(positions_list, phases_dict, camera,
                         meshes_physion, output_path,
                         orig_image=None,
                         fps=8, max_frames=40,
                         line_width=0.5, alpha=0.7,
                         backface_cull=True,
                         show_particles=False):
    """
    生成 mesh 线框投影动画 GIF。

    对每一帧:
      1. 从粒子位置反算 R,T（相对于 step 0）
      2. 将 R,T 应用到初始 mesh 顶点
      3. 投影线框到 2D 并渲染

    Args:
        positions_list: list of (n_particles, 3), 每步粒子位置
        phases_dict: Physion 格式 phases_dict
        camera: CameraInPhysion 实例
        meshes_physion: load_and_simplify_meshes() 输出（初始 Physion 空间）
        output_path: GIF 输出路径
        orig_image: 原始图像 (H, W, 3)
        fps: 帧率
        max_frames: 最大帧数
        line_width: 线框线宽
        alpha: 透明度
        backface_cull: 背面剔除
        show_particles: 是否同时显示粒子散点
    """
    instance_idx = phases_dict['instance_idx']
    n_objects = min(len(meshes_physion), len(instance_idx) - 1)
    object_names = []
    for name in phases_dict.get('instance', []):
        if isinstance(name, bytes):
            name = name.decode()
        object_names.append(name)

    W, H = camera.image_width, camera.image_height
    dpi = 100
    figsize = (W / dpi, H / dpi)

    # 帧采样
    n_steps = len(positions_list)
    if n_steps > max_frames:
        step_interval = n_steps / max_frames
        selected = [int(i * step_interval) for i in range(max_frames)]
        if (n_steps - 1) not in selected:
            selected.append(n_steps - 1)
    else:
        selected = list(range(n_steps))

    # 保存初始 mesh 顶点（用于每帧刚体变换）
    initial_verts = [m['vertices'].copy() for m in meshes_physion]

    frames = []
    for frame_i, step_idx in enumerate(selected):
        # 对每个物体, 从粒子反算 R,T 并变换 mesh
        for obj_id in range(n_objects):
            # 静态物体 (地面) 不做 SVD 变换
            if meshes_physion[obj_id].get('is_static', False):
                continue

            st = instance_idx[obj_id]
            ed = instance_idx[obj_id + 1]

            pts_0 = positions_list[0][st:ed]
            pts_t = positions_list[step_idx][st:ed]

            if len(pts_0) < 3:
                # 粒子太少, 无法计算刚体变换
                meshes_physion[obj_id]['vertices'] = initial_verts[obj_id]
                continue

            try:
                R, t = calc_rigid_transform(pts_0, pts_t)
                meshes_physion[obj_id]['vertices'] = transform_mesh_vertices(
                    initial_verts[obj_id], R, t)
            except Exception:
                meshes_physion[obj_id]['vertices'] = initial_verts[obj_id]

        # 渲染
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        render_wireframe_frame(
            ax, meshes_physion, positions_list[step_idx],
            instance_idx, camera,
            orig_image=orig_image, object_names=object_names,
            line_width=line_width, alpha=alpha,
            backface_cull=backface_cull,
            step_idx=step_idx, n_steps=n_steps,
            show_particles=show_particles,
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

        if (frame_i + 1) % 10 == 0:
            print(f"    渲染线框帧 {frame_i + 1}/{len(selected)}...")

    # 保存 GIF
    _save_gif(frames, output_path, fps)
    print(f"  线框 GIF: {output_path} ({len(frames)} 帧, {W}x{H})")

    # 保存关键帧 PNG
    key_steps = [selected[0], selected[len(selected) // 2], selected[-1]]
    base_dir = os.path.dirname(output_path)
    for ks in key_steps:
        # 变换 mesh 到该帧
        for obj_id in range(n_objects):
            # 静态物体 (地面) 不做 SVD 变换
            if meshes_physion[obj_id].get('is_static', False):
                continue
            st = instance_idx[obj_id]
            ed = instance_idx[obj_id + 1]
            pts_0 = positions_list[0][st:ed]
            pts_t = positions_list[ks][st:ed]
            if len(pts_0) >= 3:
                try:
                    R, t = calc_rigid_transform(pts_0, pts_t)
                    meshes_physion[obj_id]['vertices'] = transform_mesh_vertices(
                        initial_verts[obj_id], R, t)
                except Exception:
                    meshes_physion[obj_id]['vertices'] = initial_verts[obj_id]

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        render_wireframe_frame(
            ax, meshes_physion, positions_list[ks],
            instance_idx, camera,
            orig_image=orig_image, object_names=object_names,
            line_width=line_width, alpha=alpha,
            backface_cull=backface_cull,
            step_idx=ks, n_steps=n_steps,
            show_particles=show_particles,
        )

        key_path = os.path.join(base_dir,
                                f'camera_view_wireframe_step_{ks:03d}.png')
        fig.savefig(key_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  关键帧: {key_path}")

    # 恢复初始顶点
    for obj_id in range(n_objects):
        meshes_physion[obj_id]['vertices'] = initial_verts[obj_id]


def generate_wireframe_visualizations(positions_list, phases_dict,
                                      camera, scene_dir, trial_dir,
                                      output_dir, orig_image=None,
                                      fps=8, target_faces=3000,
                                      backface_cull=True,
                                      show_particles=False):
    """
    主入口: 生成所有 mesh 线框可视化。

    Args:
        positions_list: list of (n_particles, 3)
        phases_dict: Physion phases_dict
        camera: CameraInPhysion 实例
        scene_dir: 3DGS 场景目录（含 object_*_mesh_posed.ply）
        trial_dir: trial 目录（含 coord_transform.json）
        output_dir: 可视化输出目录
        orig_image: 原始图像 (H, W, 3)
        fps: GIF 帧率
        target_faces: mesh 简化目标面数
        backface_cull: 背面剔除
        show_particles: 是否同时显示粒子散点
    """
    print("\n" + "-" * 50)
    print("  Mesh 线框投影可视化")
    print("-" * 50)

    coord_transform_path = os.path.join(trial_dir, 'coord_transform.json')
    if not os.path.exists(coord_transform_path):
        print("  警告: coord_transform.json 不存在, 跳过线框渲染")
        return

    # Step 1: 加载并简化 mesh
    print("  加载并简化 mesh...")
    meshes_physion = load_and_simplify_meshes(
        scene_dir, coord_transform_path, target_faces=target_faces)

    if meshes_physion is None:
        return

    # Step 2: 生成线框 GIF
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, 'camera_view_wireframe.gif')

    print("  生成线框 GIF...")
    create_wireframe_gif(
        positions_list, phases_dict, camera,
        meshes_physion, gif_path,
        orig_image=orig_image,
        fps=fps,
        backface_cull=backface_cull,
        show_particles=show_particles,
    )

    print("  线框可视化完成!")
