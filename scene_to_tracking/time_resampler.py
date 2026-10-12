import numpy as np
from utils import calc_rigid_transform, project_physion_to_pixel


def build_time_mapping(n_video_frames, sim_dt, n_sim_steps, sim_duration=None):
    if sim_duration is None:
        sim_duration = (n_sim_steps - 1) * sim_dt
    max_step = min(sim_duration / sim_dt, n_sim_steps - 1)

    mapping = []
    for i in range(n_video_frames):
        frac = i / (n_video_frames - 1) if n_video_frames > 1 else 0.0
        step_float = frac * max_step

        step_lo = int(np.floor(step_float))
        step_hi = int(np.ceil(step_float))
        step_lo = min(step_lo, n_sim_steps - 1)
        step_hi = min(step_hi, n_sim_steps - 1)

        alpha = 0.0 if step_lo == step_hi else step_float - step_lo
        mapping.append((step_lo, step_hi, alpha))

    return mapping


def resample_tracks(tracks_result, visibility, time_mapping, sample_result, data):
    rollout = data['rollout_positions']
    phases = data['phases_dict']
    camera = data['camera']
    instance_idx = phases['instance_idx']

    obj_id = sample_result['obj_id']
    vert_3d_frame0 = sample_result['vert_3d_physion']
    grid_uv = sample_result['grid_uv']
    n_points = sample_result['n_points']
    n_video_frames = len(time_mapping)

    new_tracks_2d = np.zeros((n_video_frames, n_points, 2))
    new_tracks_depth = np.zeros((n_video_frames, n_points))
    new_tracks_3d = np.zeros((n_video_frames, n_points, 3))
    new_visibility = np.zeros((n_video_frames, n_points), dtype=np.float32)

    bg_mask = obj_id == -1
    new_tracks_2d[:, bg_mask, :] = grid_uv[bg_mask][np.newaxis, :, :]
    new_tracks_depth[:, bg_mask] = sample_result['depth_frame0'][bg_mask]
    new_tracks_3d[:, bg_mask] = vert_3d_frame0[bg_mask][np.newaxis, :, :]
    particles_frame0 = rollout[0]
    n_objects = data.get('n_objects', 2)

    for vi, (step_lo, step_hi, alpha) in enumerate(time_mapping):
        vis_lo = visibility[step_lo]
        vis_hi = visibility[step_hi]
        resampled_vis = np.where((vis_lo > 0.5) & (vis_hi > 0.5), 1.0, 0.0)
        new_visibility[vi] = resampled_vis

        if alpha == 0.0:
            particles_t = rollout[step_lo]
        else:
            particles_t = (1 - alpha) * rollout[step_lo] + alpha * rollout[step_hi]

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

            new_tracks_3d[vi, obj_mask] = verts_t
            new_tracks_2d[vi, obj_mask] = uv_t
            new_tracks_depth[vi, obj_mask] = depth_t

    return {
        'tracks_2d': new_tracks_2d,
        'tracks_depth': new_tracks_depth,
        'tracks_3d': new_tracks_3d,
    }, new_visibility


def remap_query_frames(query_frames_sim, sim_dt, n_video_frames, sim_duration=None, n_sim_steps=168):
    if sim_duration is None:
        sim_duration = (n_sim_steps - 1) * sim_dt
    max_step = sim_duration / sim_dt
    frac = query_frames_sim / max_step
    query_frames_video = np.ceil(frac * (n_video_frames - 1)).astype(int)
    query_frames_video = np.clip(query_frames_video, 0, n_video_frames - 1)
    return query_frames_video
