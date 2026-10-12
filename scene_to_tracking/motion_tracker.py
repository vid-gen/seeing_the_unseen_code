import numpy as np
from utils import calc_rigid_transform, project_physion_to_pixel


def compute_all_frames(sample_result, data):
    rollout = data['rollout_positions']  # (T, N_particles, 3)
    phases = data['phases_dict']
    camera = data['camera']
    n_steps = data['n_steps']
    instance_idx = phases['instance_idx']

    obj_id = sample_result['obj_id']
    vert_3d_frame0 = sample_result['vert_3d_physion']
    grid_uv = sample_result['grid_uv']
    n_points = sample_result['n_points']

    tracks_2d = np.zeros((n_steps, n_points, 2))
    tracks_depth = np.zeros((n_steps, n_points))
    tracks_3d = np.zeros((n_steps, n_points, 3))

    bg_mask = obj_id == -1
    tracks_2d[:, bg_mask, :] = grid_uv[bg_mask][np.newaxis, :, :]
    tracks_depth[:, bg_mask] = sample_result['depth_frame0'][bg_mask]

    particles_frame0 = rollout[0]  # (N_particles, 3)
    n_objects = data.get('n_objects', 2)

    for t in range(n_steps):
        particles_t = rollout[t]

        for oid in range(n_objects):
            obj_mask = obj_id == oid
            if not obj_mask.any():
                continue

            p_start = instance_idx[oid]
            p_end = instance_idx[oid + 1]
            src = particles_frame0[p_start:p_end]
            dst = particles_t[p_start:p_end]

            R, tr = calc_rigid_transform(src, dst)

            verts_0 = vert_3d_frame0[obj_mask]
            verts_t = (R @ verts_0.T).T + tr

            uv_t, depth_t = project_physion_to_pixel(verts_t, camera)

            tracks_3d[t, obj_mask] = verts_t
            tracks_2d[t, obj_mask] = uv_t
            tracks_depth[t, obj_mask] = depth_t

    tracks_3d[:, bg_mask] = vert_3d_frame0[bg_mask][np.newaxis, :, :]

    fg_mask = obj_id >= 0
    if fg_mask.any():
        offset_2d = grid_uv[fg_mask] - tracks_2d[0, fg_mask]
        tracks_2d[:, fg_mask] += offset_2d[np.newaxis, :, :]

    return {
        'tracks_2d': tracks_2d,      # (T, N, 2)
        'tracks_depth': tracks_depth,  # (T, N)
        'tracks_3d': tracks_3d,        # (T, N, 3)
    }
