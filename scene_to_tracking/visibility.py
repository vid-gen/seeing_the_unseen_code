import numpy as np
from scipy.ndimage import minimum_filter
from utils import project_physion_to_pixel, scene_to_physion, calc_rigid_transform


def compute_visibility(tracks_result, sample_result, data):
    tracks_2d = tracks_result['tracks_2d']
    tracks_depth = tracks_result['tracks_depth']
    tracks_3d = tracks_result['tracks_3d']
    camera = data['camera']
    W = camera['image_width']
    H = camera['image_height']

    n_steps, n_points, _ = tracks_2d.shape
    visibility = np.ones((n_steps, n_points), dtype=np.float32)

    obj_id = sample_result['obj_id']

    for t in range(n_steps):
        u = tracks_2d[t, :, 0]
        v = tracks_2d[t, :, 1]
        z = tracks_depth[t, :]

        out_of_frame = (u < 0) | (u >= W) | (v < 0) | (v >= H)
        behind_camera = z <= 0
        visibility[t, out_of_frame | behind_camera] = 0.0

    n_objects = data.get('n_objects', 2)
    for t in range(n_steps):
        for oid in range(n_objects):
            obj_mask = obj_id == oid
            if not obj_mask.any():
                continue

            pts_3d = tracks_3d[t, obj_mask]
            normals_0 = sample_result.get('per_obj_normals_phy', [None] * n_objects)

            if normals_0[oid] is not None:
                mesh_idx = sample_result['mesh_vertex_index'][obj_mask]
                per_obj_vis_indices = sample_result['per_obj_vert_indices'][oid]

                rollout = data['rollout_positions']
                phases = data['phases_dict']
                instance_idx = phases['instance_idx']
                p_start = instance_idx[oid]
                p_end = instance_idx[oid + 1]

                from utils import calc_rigid_transform
                R, _ = calc_rigid_transform(
                    rollout[0][p_start:p_end],
                    rollout[t][p_start:p_end]
                )

                local_idx_map = {int(gi): li for li, gi in enumerate(per_obj_vis_indices)}
                normals_frame0 = normals_0[oid]

                cam_pos = np.array(camera['position'])
                R_inv = camera.get('domino_R_inv', None)
                obj_indices = np.where(obj_mask)[0]
                for j, global_j in enumerate(obj_indices):
                    mi = int(sample_result['mesh_vertex_index'][global_j])
                    if mi in local_idx_map:
                        n0 = normals_frame0[local_idx_map[mi]]
                        n_t = R @ n0
                        pt = pts_3d[j]
                        if R_inv is not None:
                            n_t = R_inv @ n_t
                            pt = R_inv @ pt
                        view_dir = cam_pos - pt
                        if np.dot(n_t, view_dir) <= 0:
                            visibility[t, global_j] = 0.0

    return visibility  # (T, N)


def apply_inter_object_occlusion(visibility, tracks_2d, tracks_depth, sample_result, data):
    camera = data['camera']
    W, H = camera['image_width'], camera['image_height']
    rollout = data['rollout_positions']
    instance_idx = data['phases_dict']['instance_idx']
    meshes = data['meshes']
    coord_transform = data['coord_transform']
    n_steps = data['n_steps']
    obj_id = sample_result['obj_id']
    n_objects = len(meshes)

    mesh_verts_phy = []
    mesh_normals_phy = []
    for mesh in meshes:
        vs = np.array(mesh.vertices)
        ns = mesh.vertex_normals.copy()
        vp = scene_to_physion(vs, coord_transform)
        nr = scene_to_physion(vs + ns * 0.001, coord_transform) - vp
        norms = np.linalg.norm(nr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1e-10)
        mesh_verts_phy.append(vp)
        mesh_normals_phy.append(nr / norms)

    cam_pos = np.array(camera['position'])
    particles_0 = rollout[0]
    visibility = visibility.copy()
    total_occluded = 0
    total_self_occluded = 0

    for t in range(n_steps):
        depth_maps = []
        for oid in range(n_objects):
            ps, pe = instance_idx[oid], instance_idx[oid + 1]
            R, tr = calc_rigid_transform(particles_0[ps:pe], rollout[t][ps:pe])

            verts_t = (R @ mesh_verts_phy[oid].T).T + tr
            normals_t = (R @ mesh_normals_phy[oid].T).T

            verts_for_view = verts_t
            normals_for_view = normals_t
            R_inv = camera.get('domino_R_inv', None)
            if R_inv is not None:
                verts_for_view = (R_inv @ verts_t.T).T
                normals_for_view = (R_inv @ normals_t.T).T
            view = cam_pos - verts_for_view
            view_n = np.linalg.norm(view, axis=1, keepdims=True)
            view_n = np.where(view_n > 1e-10, view_n, 1e-10)
            view /= view_n
            front = np.sum(normals_for_view * view, axis=1) > 0

            uv, depth = project_physion_to_pixel(verts_t, camera)
            valid = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < H) & (depth > 0)

            dmap = np.full((H, W), np.inf)
            vu = np.round(uv[valid]).astype(int)
            vu[:, 0] = np.clip(vu[:, 0], 0, W - 1)
            vu[:, 1] = np.clip(vu[:, 1], 0, H - 1)
            np.minimum.at(dmap, (vu[:, 1], vu[:, 0]), depth[valid])
            dmap = minimum_filter(dmap, size=7)
            depth_maps.append(dmap)

        for oid in range(n_objects):
            mask = (obj_id == oid) & (visibility[t] > 0)
            if not mask.any():
                continue
            indices = np.where(mask)[0]
            u = np.clip(np.round(tracks_2d[t, indices, 0]).astype(int), 0, W - 1)
            v = np.clip(np.round(tracks_2d[t, indices, 1]).astype(int), 0, H - 1)
            pd = tracks_depth[t, indices]

            for other in range(n_objects):
                if other == oid:
                    continue
                occluded = depth_maps[other][v, u] < pd
                visibility[t, indices[occluded]] = 0.0
                total_occluded += occluded.sum()

            self_occluded = depth_maps[oid][v, u] + 0.03 < pd
            visibility[t, indices[self_occluded]] = 0.0
            total_self_occluded += self_occluded.sum()

        bg_mask = (obj_id == -1) & (visibility[t] > 0)
        if bg_mask.any():
            bg_indices = np.where(bg_mask)[0]
            u_bg = np.clip(np.round(tracks_2d[t, bg_indices, 0]).astype(int), 0, W - 1)
            v_bg = np.clip(np.round(tracks_2d[t, bg_indices, 1]).astype(int), 0, H - 1)
            pd_bg = tracks_depth[t, bg_indices]
            for oid in range(n_objects):
                occluded = depth_maps[oid][v_bg, u_bg] < pd_bg
                visibility[t, bg_indices[occluded]] = 0.0
                total_occluded += occluded.sum()

    print(f"  Inter-object occlusion: {total_occluded} point-frames marked invisible")
    print(f"  Self-occlusion: {total_self_occluded} point-frames marked invisible")
    return visibility
