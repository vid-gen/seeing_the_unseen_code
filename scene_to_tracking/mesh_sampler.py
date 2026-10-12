import numpy as np
from scipy.spatial import cKDTree
from utils import scene_to_physion, project_physion_to_pixel, calc_rigid_transform


def compute_vertex_normals(mesh):
    return mesh.vertex_normals.copy()


def backface_cull(vertices_phy, normals_phy, camera):
    verts = vertices_phy
    norms_dir = normals_phy
    if 'domino_R_inv' in camera:
        R_inv = camera['domino_R_inv']
        verts = (R_inv @ vertices_phy.T).T
        norms_dir = (R_inv @ normals_phy.T).T
    cam_pos = np.array(camera['position'])
    view_dirs = cam_pos - verts
    vd_len = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    vd_len = np.where(vd_len > 1e-10, vd_len, 1e-10)
    view_dirs = view_dirs / vd_len
    dots = np.sum(norms_dir * view_dirs, axis=1)
    front_mask = dots > 0
    return front_mask


def filter_closest_by_depth(uv, depth, front_mask, pixel_radius=0.5):
    valid = front_mask.copy()
    uv_int = np.round(uv[valid]).astype(int)
    depths_valid = depth[valid]
    valid_indices = np.where(valid)[0]

    pixel_map = {}
    for i, (idx) in enumerate(valid_indices):
        key = (uv_int[i, 0], uv_int[i, 1])
        if key not in pixel_map or depths_valid[i] < pixel_map[key][1]:
            pixel_map[key] = (idx, depths_valid[i])

    closest_set = set(v[0] for v in pixel_map.values())
    result_mask = np.zeros(len(front_mask), dtype=bool)
    for idx in closest_set:
        result_mask[idx] = True
    return result_mask


def sample_visible_tracking_points(data, grid_size=70):
    meshes = data['meshes']
    masks = data['masks']
    camera = data['camera']
    coord_transform = data['coord_transform']
    W = camera['image_width']
    H = camera['image_height']

    all_verts_phy = []
    all_normals_phy = []
    all_obj_ids = []
    all_vert_indices = []

    for obj_id, mesh in enumerate(meshes):
        verts_scene = np.array(mesh.vertices)
        normals_scene = compute_vertex_normals(mesh)
        verts_phy = scene_to_physion(verts_scene, coord_transform)
        normals_phy = scene_to_physion(
            verts_scene + normals_scene * 0.001, coord_transform
        ) - verts_phy
        norms = np.linalg.norm(normals_phy, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1e-10)
        normals_phy = normals_phy / norms

        front_mask = backface_cull(verts_phy, normals_phy, camera)
        uv, depth = project_physion_to_pixel(verts_phy, camera)
        in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (depth > 0)
        visible_mask = front_mask & in_frame
        visible_mask = filter_closest_by_depth(uv, depth, visible_mask)

        visible_indices = np.where(visible_mask)[0]
        all_verts_phy.append(verts_phy[visible_indices])
        all_normals_phy.append(normals_phy[visible_indices])
        all_obj_ids.append(np.full(len(visible_indices), obj_id))
        all_vert_indices.append(visible_indices)

    per_obj_verts_phy = [v.copy() for v in all_verts_phy]
    per_obj_vert_indices = [v.copy() for v in all_vert_indices]

    step = W // 64
    xs = np.linspace(step, W - step, grid_size)
    ys = np.linspace(step, H - step, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_uv = np.stack([xx.flatten(), yy.flatten()], axis=1)
    n_points = len(grid_uv)

    sample_obj_id = np.full(n_points, -1, dtype=int)
    sample_vert_3d = np.zeros((n_points, 3))
    sample_uv = grid_uv.copy()
    sample_depth = np.zeros(n_points)
    sample_mesh_idx = np.full(n_points, -1, dtype=int)

    per_obj_kdtrees = []
    per_obj_uv = []
    per_obj_depth = []
    for obj_id in range(len(meshes)):
        uv_obj, depth_obj = project_physion_to_pixel(per_obj_verts_phy[obj_id], camera)
        per_obj_kdtrees.append(cKDTree(uv_obj))
        per_obj_uv.append(uv_obj)
        per_obj_depth.append(depth_obj)

    mesh_match_radius = 3.0

    for i in range(n_points):
        u, v = int(round(grid_uv[i, 0])), int(round(grid_uv[i, 1]))
        u_c = min(max(u, 0), W - 1)
        v_c = min(max(v, 0), H - 1)

        best_obj = -1
        best_dist = float('inf')
        best_vert_idx = -1

        for obj_id in range(len(meshes)):
            dist, idx = per_obj_kdtrees[obj_id].query(grid_uv[i])
            if dist < mesh_match_radius and dist < best_dist:
                best_dist = dist
                best_obj = obj_id
                best_vert_idx = idx

        if best_obj < 0:
            for obj_id in range(len(meshes)):
                if masks[obj_id][v_c, u_c] > 0:
                    dist, idx = per_obj_kdtrees[obj_id].query(grid_uv[i])
                    if best_obj < 0 or dist < best_dist:
                        best_dist = dist
                        best_obj = obj_id
                        best_vert_idx = idx

        if best_obj >= 0:
            sample_obj_id[i] = best_obj
            sample_vert_3d[i] = per_obj_verts_phy[best_obj][best_vert_idx]
            sample_depth[i] = per_obj_depth[best_obj][best_vert_idx]
            sample_mesh_idx[i] = per_obj_vert_indices[best_obj][best_vert_idx]

    max_fg_depth = sample_depth[sample_obj_id >= 0].max() if (sample_obj_id >= 0).any() else 1.0
    sample_depth[sample_obj_id < 0] = max_fg_depth * 1.2

    result = {
        'grid_uv': grid_uv,
        'obj_id': sample_obj_id,
        'vert_3d_physion': sample_vert_3d,
        'uv_frame0': sample_uv,
        'depth_frame0': sample_depth,
        'mesh_vertex_index': sample_mesh_idx,
        'n_points': n_points,
        'per_obj_verts_phy_visible': per_obj_verts_phy,
        'per_obj_vert_indices': per_obj_vert_indices,
        'per_obj_normals_phy': [n.copy() for n in all_normals_phy],
    }

    n_bg = (sample_obj_id == -1).sum()
    print(f"Sampling complete: {n_points} grid points")
    for oid in range(len(meshes)):
        n_obj = (sample_obj_id == oid).sum()
        print(f"  Object {oid}: {n_obj}")
    print(f"  Background: {n_bg}")

    return result


def _get_mesh_physion_data(mesh, coord_transform):
    verts_scene = np.array(mesh.vertices)
    normals_scene = compute_vertex_normals(mesh)
    verts_phy = scene_to_physion(verts_scene, coord_transform)
    normals_phy = scene_to_physion(
        verts_scene + normals_scene * 0.001, coord_transform
    ) - verts_phy
    norms = np.linalg.norm(normals_phy, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1e-10)
    normals_phy = normals_phy / norms
    return verts_phy, normals_phy


def _check_visibility_vectorized(verts, normals, camera):
    cam_pos = np.array(camera['position'])
    W, H = camera['image_width'], camera['image_height']
    v = verts
    n = normals
    if 'domino_R_inv' in camera:
        R_inv = camera['domino_R_inv']
        v = (R_inv @ verts.T).T
        n = (R_inv @ normals.T).T
    view_dirs = cam_pos - v
    vd_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    vd_norms = np.where(vd_norms > 1e-10, vd_norms, 1e-10)
    view_dirs = view_dirs / vd_norms
    dots = np.sum(n * view_dirs, axis=1)
    front = dots > 0
    uv, depth = project_physion_to_pixel(verts, camera)
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (depth > 0)
    visible = front & in_frame
    return visible, uv, depth


def sample_future_visible_tracking_points(data, dense_spacing=4):
    meshes = data['meshes']
    camera = data['camera']
    coord_transform = data['coord_transform']
    rollout = data['rollout_positions']
    phases = data['phases_dict']
    instance_idx = phases['instance_idx']
    n_steps = data['n_steps']
    W, H = camera['image_width'], camera['image_height']

    all_fv_verts = []
    all_fv_normals = []
    all_fv_obj_ids = []
    all_fv_query_frames = []
    all_fv_mesh_indices = []
    all_fv_uv_at_qf = []
    all_fv_depth_at_qf = []

    for obj_id, mesh in enumerate(meshes):
        verts_phy, normals_phy = _get_mesh_physion_data(mesh, coord_transform)
        visible_0, _, _ = _check_visibility_vectorized(verts_phy, normals_phy, camera)

        invisible_indices = np.where(~visible_0)[0]
        n_inv = len(invisible_indices)
        print(f"  Object {obj_id}: {n_inv} invisible vertices at frame 0")
        if n_inv == 0:
            continue

        inv_verts = verts_phy[invisible_indices]
        inv_normals = normals_phy[invisible_indices]
        query_frames = np.full(n_inv, -1, dtype=int)

        p_start = instance_idx[obj_id]
        p_end = instance_idx[obj_id + 1]
        src_particles = rollout[0][p_start:p_end]

        for t in range(1, n_steps):
            unassigned_mask = query_frames == -1
            if not unassigned_mask.any():
                break

            R, tr = calc_rigid_transform(src_particles, rollout[t][p_start:p_end])

            ua_idx = np.where(unassigned_mask)[0]
            verts_t = (R @ inv_verts[ua_idx].T).T + tr
            normals_t = (R @ inv_normals[ua_idx].T).T

            visible_t, _, _ = _check_visibility_vectorized(verts_t, normals_t, camera)
            became_visible = ua_idx[visible_t]
            query_frames[became_visible] = t

        ever_visible = query_frames >= 0
        ev_idx = np.where(ever_visible)[0]
        print(f"  Object {obj_id}: {len(ev_idx)} vertices become visible later")
        if len(ev_idx) == 0:
            continue

        ev_verts = inv_verts[ev_idx]
        ev_normals = inv_normals[ev_idx]
        ev_qf = query_frames[ev_idx]
        ev_mesh_idx = invisible_indices[ev_idx]

        uv_at_qf = np.zeros((len(ev_idx), 2))
        depth_at_qf = np.zeros(len(ev_idx))
        for t in np.unique(ev_qf):
            mask_t = ev_qf == t
            R, tr = calc_rigid_transform(src_particles, rollout[t][p_start:p_end])
            verts_t = (R @ ev_verts[mask_t].T).T + tr
            uv_t, dep_t = project_physion_to_pixel(verts_t, camera)
            uv_at_qf[mask_t] = uv_t
            depth_at_qf[mask_t] = dep_t

        all_fv_verts.append(ev_verts)
        all_fv_normals.append(ev_normals)
        all_fv_obj_ids.append(np.full(len(ev_idx), obj_id))
        all_fv_query_frames.append(ev_qf)
        all_fv_mesh_indices.append(ev_mesh_idx)
        all_fv_uv_at_qf.append(uv_at_qf)
        all_fv_depth_at_qf.append(depth_at_qf)

    if not all_fv_verts:
        print("No future visible vertices found.")
        return None

    fv_verts = np.concatenate(all_fv_verts)
    fv_normals = np.concatenate(all_fv_normals)
    fv_obj_ids = np.concatenate(all_fv_obj_ids)
    fv_qf = np.concatenate(all_fv_query_frames)
    fv_mesh_idx = np.concatenate(all_fv_mesh_indices)
    fv_uv = np.concatenate(all_fv_uv_at_qf)
    fv_depth = np.concatenate(all_fv_depth_at_qf)

    cell_w = dense_spacing
    cell_h = dense_spacing
    cell_map = {}
    for i in range(len(fv_verts)):
        u, v = fv_uv[i]
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
        cx = int(u / cell_w)
        cy = int(v / cell_h)
        cell_center_u = (cx + 0.5) * cell_w
        cell_center_v = (cy + 0.5) * cell_h
        dist_sq = (u - cell_center_u) ** 2 + (v - cell_center_v) ** 2
        key = (cx, cy)
        if key not in cell_map or fv_qf[i] < cell_map[key][1] or \
           (fv_qf[i] == cell_map[key][1] and dist_sq < cell_map[key][2]):
            cell_map[key] = (i, fv_qf[i], dist_sq)

    selected = sorted([v[0] for v in cell_map.values()])
    n_selected = len(selected)

    result = {
        'grid_uv': fv_uv[selected],
        'obj_id': fv_obj_ids[selected],
        'vert_3d_physion': fv_verts[selected],
        'depth_frame0': fv_depth[selected],
        'mesh_vertex_index': fv_mesh_idx[selected],
        'query_frames': fv_qf[selected],
        'uv_at_query_frame': fv_uv[selected],
        'depth_at_query_frame': fv_depth[selected],
        'n_points': n_selected,
    }

    n_objects = len(meshes)
    per_obj_vert_indices = [None] * n_objects
    per_obj_normals_phy = [None] * n_objects
    for oid in range(n_objects):
        obj_mask = result['obj_id'] == oid
        if obj_mask.any():
            per_obj_vert_indices[oid] = result['mesh_vertex_index'][obj_mask]
            sel_in_concat = np.array(selected)[obj_mask]
            per_obj_normals_phy[oid] = fv_normals[sel_in_concat]

    result['per_obj_verts_phy_visible'] = [None] * n_objects
    result['per_obj_vert_indices'] = per_obj_vert_indices
    result['per_obj_normals_phy'] = per_obj_normals_phy

    print(f"Future visible sampling: {n_selected} points (dense_spacing={dense_spacing}px)")
    for oid in range(n_objects):
        n_obj = (result['obj_id'] == oid).sum()
        if n_obj > 0:
            qf_obj = result['query_frames'][result['obj_id'] == oid]
            print(f"  Object {oid}: {n_obj} points, query_frame range [{qf_obj.min()}, {qf_obj.max()}]")

    return result
