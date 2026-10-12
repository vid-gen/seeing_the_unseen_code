import numpy as np


def calc_rigid_transform(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    centroid_s = source.mean(axis=0)
    centroid_t = target.mean(axis=0)
    H = (source - centroid_s).T @ (target - centroid_t)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = centroid_t - R @ centroid_s
    return R, t


def scene_to_physion(verts_scene, coord_transform):
    offset = np.array([
        coord_transform['center_x'],
        coord_transform['ground_y'],
        coord_transform['center_z'],
    ])
    scale = coord_transform['scale']
    gc = coord_transform.get('ground_clearance', 0.0)
    ssf = coord_transform.get('scene_scale_factor', 1.0)
    verts_phy = (verts_scene * ssf - offset) * scale
    if 'domino_rotation_matrix' in coord_transform:
        R = np.array(coord_transform['domino_rotation_matrix'])
        verts_phy = (R @ verts_phy.T).T
    verts_phy[:, 1] += gc
    return verts_phy


def project_physion_to_pixel(points, camera):
    pts = points.copy()
    if 'domino_R_inv' in camera:
        pts = (camera['domino_R_inv'] @ pts.T).T
    pts[:, 1] -= camera['ground_clearance']
    p_cam = pts - np.array(camera['position'])
    p_moge = p_cam * np.array([-1.0, -1.0, 1.0])
    z = p_moge[:, 2].copy()
    z_safe = np.where(np.abs(z) > 1e-8, z, 1e-8)
    intr = camera['intrinsics_norm']
    fx_n, cx_n = intr[0][0], intr[0][2]
    fy_n, cy_n = intr[1][1], intr[1][2]
    W, H = camera['image_width'], camera['image_height']
    u_norm = fx_n * p_moge[:, 0] / z_safe + cx_n
    v_norm = fy_n * p_moge[:, 1] / z_safe + cy_n
    u_px = u_norm * W
    v_px = v_norm * H
    uv = np.stack([u_px, v_px], axis=1)
    return uv, z
