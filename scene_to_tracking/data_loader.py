import json
import pickle
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path


def _match_masks_to_meshes(meshes, masks, coord_transform, camera):
    from utils import scene_to_physion, project_physion_to_pixel

    n_objects = len(meshes)
    mesh_centers_x = []
    for mesh in meshes:
        verts_phy = scene_to_physion(np.array(mesh.vertices), coord_transform)
        uv, _ = project_physion_to_pixel(verts_phy, camera)
        mesh_centers_x.append((uv[:, 0].min() + uv[:, 0].max()) / 2)

    mask_centers_x = []
    for mask in masks:
        xs = np.where(mask.max(axis=0) > 0)[0] if mask.ndim == 2 else np.where(mask.any(axis=(0, 2)))[0]
        if len(xs) == 0:
            mask_centers_x.append(-1)
        else:
            mask_centers_x.append((xs.min() + xs.max()) / 2)

    used_masks = set()
    reordered = [None] * n_objects
    for mi in range(n_objects):
        best_dist = float('inf')
        best_mask = -1
        for ki in range(n_objects):
            if ki in used_masks:
                continue
            dist = abs(mesh_centers_x[mi] - mask_centers_x[ki])
            if dist < best_dist:
                best_dist = dist
                best_mask = ki
        used_masks.add(best_mask)
        reordered[mi] = masks[best_mask]
        print(f"  Mask matching: mesh_{mi} (cx={mesh_centers_x[mi]:.0f}) -> mask_{best_mask} (cx={mask_centers_x[best_mask]:.0f})")

    return reordered


def load_all_data(data_dir):
    data_dir = Path(data_dir)
    data = {}

    with open(data_dir / 'trial' / 'coord_transform.json') as f:
        data['coord_transform'] = json.load(f)

    with open(data_dir / 'rollout' / 'camera_physion.json') as f:
        cam = json.load(f)
    cam['ground_clearance'] = data['coord_transform'].get('ground_clearance', 0.0)
    domino_R = data['coord_transform'].get('domino_rotation_matrix', None)
    if domino_R is not None:
        cam['domino_R_inv'] = np.array(domino_R).T
    data['camera'] = cam

    with open(data_dir / 'trial' / 'phases_dict.pkl', 'rb') as f:
        data['phases_dict'] = pickle.load(f)

    n_objects_phases = data['phases_dict']['n_objects']

    available_meshes = 0
    for i in range(n_objects_phases):
        if (data_dir / 'scene_3dgs' / f'object_{i}_mesh_posed.ply').exists():
            available_meshes += 1
        else:
            break
    n_objects = available_meshes
    if n_objects < n_objects_phases:
        print(f"  Note: phases_dict has {n_objects_phases} objects but only {n_objects} meshes found, using {n_objects}")
    data['n_objects'] = n_objects

    rollout = np.load(data_dir / 'rollout' / 'rollout_positions.npz')
    n_steps = len(rollout.files)
    positions = []
    for i in range(n_steps):
        positions.append(rollout[f'step_{i}'])
    data['rollout_positions'] = np.array(positions)  # (T, N_particles, 3)
    data['n_steps'] = n_steps

    meshes = []
    for i in range(n_objects):
        mesh_path = data_dir / 'scene_3dgs' / f'object_{i}_mesh_posed.ply'
        mesh = trimesh.load(mesh_path, process=False)
        meshes.append(mesh)
    data['meshes'] = meshes

    masks = []
    for i in range(n_objects):
        mask_path = data_dir / 'scene_3dgs' / f'mask_{i}.png'
        mask = np.array(Image.open(mask_path))
        masks.append(mask)

    masks = _match_masks_to_meshes(meshes, masks, data['coord_transform'], data['camera'])
    data['masks'] = masks

    for candidate in ['1280_704_image.png', 'first_frame_1280x704.png', 'first_frame.png']:
        img_path = data_dir / candidate
        if img_path.exists():
            img = np.array(Image.open(img_path))
            data['first_frame'] = img
            break
    else:
        raise FileNotFoundError(f"No first frame image found in {data_dir}")

    return data
