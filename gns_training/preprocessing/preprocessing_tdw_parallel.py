# preprocessing - parallel version
import h5py
import copy
import os
import trimesh
import pickle
import numpy as np
import random
import argparse
import imageio
from scipy.spatial.transform import Rotation as R
from utils_geom import save_obj, mesh_to_particles
from utils import mkdir, get_query_dir
from data import store_data
import copy
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED
import time

training_data_dir = get_query_dir("training_data_dir")
testing_data_dir = get_query_dir("testing_data_dir")
output_prefix = get_query_dir("dpi_data_dir")

models_full_check_path = "./files/models_full_check_window.txt"

bad_meshes_ratio = dict()
with open(models_full_check_path, "r") as f:
    for line in f:
        obj_name, x_ratio, y_ratio, z_ratio = line.split(",")
        bad_meshes_ratio[obj_name.encode('UTF-8')] = [float(x_ratio), float(y_ratio), float(z_ratio)]

spacing = 0.05
dt = 0.01

def process_single_trial(trial_info, data_source_root, output_root_base, flex_engine):
    """Process a single trial HDF5 file."""
    trial_hdf5_name, output_root = trial_info

    # Skip if already processed
    phases_dict_path = os.path.join(output_root, 'phases_dict.pkl')
    if os.path.exists(phases_dict_path):
        return f"Skipped {trial_hdf5_name} (already processed)"

    # Create unique temp directory for this process
    tmp_path = tempfile.mkdtemp(prefix=f"physion_tmp_{os.getpid()}_")

    try:
        mkdir(output_root)
        filename = os.path.join(data_source_root, trial_hdf5_name)

        # dummy node variance
        positions_stat = np.zeros((3,3))
        velocities_stat = np.zeros((3,3))
        store_data(["positions", "velocities"], [positions_stat, velocities_stat], os.path.join(output_root, 'stat.h5'))

        try:
            f = h5py.File(filename, "r")
        except OSError as e:
            if os.path.exists(output_root):
                shutil.rmtree(output_root)
            return f"ERROR: Cannot open {filename}: {e}"

        object_ids = f["static"]["object_ids"][:].tolist()
        scales = f["static"]["scale"][:]
        object_names = f["static"]["model_names"][:]

        if b'ramp_with_platform_30' in object_names:
            if(object_names[2] == b'ramp_with_platform_30'):
                object_name1 = copy.deepcopy(object_names[1])
                object_names[1] = copy.deepcopy(object_names[2])
                object_names[2] = object_name1
                scales1 = copy.deepcopy(scales[1])
                scales[1] = copy.deepcopy(scales[2])
                scales[2] = scales1
            else:
                assert(object_names[1] == b'ramp_with_platform_30')
                assert(np.linalg.norm(scales[1] - np.array([0.2, 1, 0.5])) < 0.000001)

        nobjects = len(object_ids)
        nonexists = np.zeros((nobjects), dtype=bool)
        clusters = []
        root_num = []
        root_des_radius = []
        instance_idx = [0]
        count_nodes = 0
        instance = []
        material = []
        obj_points = []

        for idx, object_id in enumerate(object_ids):
            if flex_engine:
                points = f["frames"]["0000"]["particles"][str(object_id)][:]
                obj_points.append(points)
            else:
                vertices = f["static"]["mesh"][f"vertices_{idx}"][:]
                faces = f["static"]["mesh"][f"faces_{idx}"][:]
                obj_name = f["static"]["model_names"][:][idx]

                if vertices.shape[0] == 0:
                    nonexists[idx] = 1
                    if obj_name in [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']:
                        return f"ERROR: critical object with empty mesh {obj_name} in {filename}"
                    continue

                if obj_name in bad_meshes_ratio:
                    if obj_name in [b"cork_plastic_black", b'tapered_cork_w_hole']:
                        scales[idx, :] *= bad_meshes_ratio[obj_name][0]
                    else:
                        scales[idx, :] *= np.array(bad_meshes_ratio[obj_name])

                if max(scales[idx]) > 25:
                    if not(abs(scales[idx,0] - 0.05) < 0.00001 and abs(scales[idx,1] - 0.05) < 0.00001 and abs(scales[idx,2] - 100) < 0.00001):
                        if not (obj_name == b"889242_mesh" and np.max(abs(scales[idx] - 37.092888)) < 0.00001):
                            if obj_name in [b"cork_plastic_black", b'tapered_cork_w_hole']:
                                scales[idx] = 25

                obj_filename = os.path.join(tmp_path, f"{idx}.obj")
                vertices[:,0] *= scales[idx, 0]
                vertices[:,1] *= scales[idx, 1]
                vertices[:,2] *= scales[idx, 2]

                if vertices.shape[0] == 0:
                    nonexists[idx] = 1
                    continue

                save_obj(vertices, faces, obj_filename)
                points = mesh_to_particles(obj_filename, spacing=spacing, visualization=False)

                if points.shape[0] == 0:
                    nonexists[idx] = 1
                    continue
                obj_points.append(points)
                os.remove(obj_filename)

            npts = points.shape[0]
            count_nodes += npts
            clusters.append([[np.array([0]* npts, dtype=np.int32)]])
            root_num.append([1])
            root_des_radius.append([spacing])
            instance_idx.append(count_nodes)

            obj_name = f["static"]["model_names"][:][idx]
            if obj_name == b'cloth_square':
                material.append("cloth")
            else:
                material.append("rigid")
            instance.append(obj_name)

        rollout_dir = output_root
        mkdir(rollout_dir)
        nsteps = len(f["frames"])
        n_objects = len(obj_points)
        n_particles = np.sum([obj_pts.shape[0] for obj_pts in obj_points])

        yellow_id = f["static"]["zone_id"][()]
        red_id = f["static"]["target_id"][()]

        yellow_id_order = [order_id for order_id, id_ in enumerate(object_ids) if id_ == yellow_id]
        assert(len(yellow_id_order) == 1)
        yellow_id_order = yellow_id_order[0]

        red_id_order = [order_id for order_id, id_ in enumerate(object_ids) if id_ == red_id]
        assert(len(red_id_order) == 1)
        red_id_order = red_id_order[0]

        assert(yellow_id_order==0)

        safe_up_to_idx = max(yellow_id_order, red_id_order)
        for id_ in range(safe_up_to_idx):
            if nonexists[id_]:
                if os.path.exists(output_root):
                    shutil.rmtree(output_root)
                return f"WARNING: Object {id_} does not exist but is needed. Skipping {trial_hdf5_name}"

        # save global info for the trial
        phases_dict = dict()
        phases_dict["instance_idx"] = instance_idx
        phases_dict["root_des_radius"] = root_des_radius
        phases_dict["root_num"] = root_num
        phases_dict["clusters"] = clusters
        phases_dict["instance"] = instance
        phases_dict["material"] = material
        phases_dict["time_step"] = nsteps
        phases_dict["n_objects"] = n_objects
        phases_dict["n_particles"] = n_particles
        phases_dict["obj_points"] = obj_points
        phases_dict["dt"] = dt
        phases_dict["yellow_id"] = yellow_id_order
        phases_dict["red_id"] = red_id_order

        # Validate
        assert(phases_dict["n_objects"] == len(phases_dict["instance_idx"]) - 1)
        assert(phases_dict["n_objects"] == len(phases_dict["root_des_radius"]))
        assert(phases_dict["n_objects"] == len(phases_dict["root_num"]))
        assert(phases_dict["n_objects"] == len(phases_dict["clusters"]))
        assert(phases_dict["n_objects"] == len(phases_dict["instance"]))
        assert(phases_dict["n_objects"] == len(phases_dict["material"]))
        assert(phases_dict["n_objects"] == len(phases_dict["obj_points"]))
        for obj_pts in obj_points:
            assert(obj_pts.shape[0] > 0)

        with open(os.path.join(rollout_dir, 'phases_dict.pkl'), "wb") as pf:
            pickle.dump(phases_dict, pf)

        for step in range(nsteps):
            if flex_engine:
                positions = []
                velocities = []
                for idx, object_id in enumerate(object_ids):
                    pos = f["frames"][f"{step:04}"]["particles"][str(object_id)][:][:,:3]
                    vel = f["frames"][f"{step:04}"]["velocities"][str(object_id)][:][:,:3]
                    positions.append(pos)
                    velocities.append(vel)
                positions = np.concatenate(positions, axis=0)
                velocities = np.concatenate(velocities, axis=0)
                data_names = ['particle_positions', "particle_velocities"]
                data = [positions, velocities]
                store_data(data_names, data, os.path.join(rollout_dir, str(step) + '.h5'))
            else:
                positions = f["frames"][f"{step:04}"]["objects"]["positions"][:]
                rotations = f["frames"][f"{step:04}"]["objects"]["rotations"][:]

                if np.sum(nonexists) > 0:
                    pos_ = []
                    ros_ = []
                    for idx, object_id in enumerate(object_ids):
                        if not nonexists[idx]:
                            pos_.append(positions[idx])
                            ros_.append(rotations[idx])
                    positions = np.stack(pos_, axis=0)
                    rotations = np.stack(ros_, axis=0)
                    assert(positions.shape[0] == phases_dict["n_objects"])
                    assert(rotations.shape[0] == phases_dict["n_objects"])

                data_names = ['obj_positions', 'obj_rotations']
                data = [positions, rotations]
                store_data(data_names, data, os.path.join(rollout_dir, str(step) + '.h5'))

                if step == 0:
                    assert(n_objects == positions.shape[0]), f"nobjects does not match number of positions: {n_objects} vs {positions.shape[0]}"

        f.close()
        return f"Processed {trial_hdf5_name}"

    except Exception as e:
        if os.path.exists(output_root):
            shutil.rmtree(output_root)
        return f"ERROR processing {trial_hdf5_name}: {str(e)}"

    finally:
        # Clean up temp directory
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--scenario', default='')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for each trial (default: 60)')
    args = parser.parse_args()

    mode = args.mode
    if mode == "train":
        data_source_root = training_data_dir
    elif mode == "test":
        data_source_root = testing_data_dir
    else:
        raise ValueError

    output_root_base = os.path.join(output_prefix, mode)

    scenario = args.scenario
    scene_path = os.path.join(data_source_root, scenario)
    trial_names = [file for file in os.listdir(scene_path) if file.endswith("hdf5")]
    trial_names.sort()

    trial_ndata = [os.path.join(scenario, trial_name) for trial_name in trial_names]
    output_roots = [os.path.join(output_root_base, path)[:-5] for path in trial_ndata]

    flex_engine = "Drape" in scenario

    # Prepare trial info list
    trial_infos = list(zip(trial_ndata, output_roots))

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers else mp.cpu_count()
    timeout = args.timeout
    print(f"Processing {len(trial_infos)} trials with {num_workers} workers (timeout: {timeout}s)...")

    # Create partial function with fixed arguments
    process_func = partial(process_single_trial,
                          data_source_root=data_source_root,
                          output_root_base=output_root_base,
                          flex_engine=flex_engine)

    # Process in parallel with timeout support
    results = []
    timeout_files = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and track submission order
        future_to_trial = {}
        submission_order = []

        for trial_info in trial_infos:
            future = executor.submit(process_func, trial_info)
            future_to_trial[future] = trial_info
            submission_order.append(future)

        pending = set(future_to_trial.keys())
        last_progress_time = time.time()

        # Process results with progress bar
        with tqdm(total=len(trial_infos), desc=f"Processing {scenario}") as pbar:
            while pending:
                # Wait for any task to complete, checking every 5 seconds
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)

                # Process completed tasks
                if done:
                    last_progress_time = time.time()
                    for future in done:
                        trial_info = future_to_trial[future]
                        trial_hdf5_name = trial_info[0]
                        output_root = trial_info[1]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            results.append(f"ERROR: {trial_hdf5_name}: {str(e)}")
                        pbar.update(1)

                # Check for progress stall timeout
                elapsed_since_progress = time.time() - last_progress_time
                if elapsed_since_progress > timeout and pending:
                    # Progress stalled for too long, skip 1 task at a time
                    # to minimize loss (we don't know which task is stuck)
                    skipped_count = 0
                    for future in submission_order:
                        if future in pending and skipped_count < 1:
                            trial_info = future_to_trial[future]
                            trial_hdf5_name = trial_info[0]
                            output_root = trial_info[1]
                            future.cancel()
                            if os.path.exists(output_root):
                                shutil.rmtree(output_root)
                            timeout_msg = f"TIMEOUT: {trial_hdf5_name} exceeded {timeout}s (stalled), skipped"
                            results.append(timeout_msg)
                            timeout_files.append(trial_hdf5_name)
                            pending.discard(future)
                            skipped_count += 1
                            pbar.update(1)

                    # Reset progress timer
                    last_progress_time = time.time()

    # Print results summary
    processed = sum(1 for r in results if r.startswith("Processed"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    timeouts = sum(1 for r in results if r.startswith("TIMEOUT"))
    errors = sum(1 for r in results if r.startswith("ERROR") or r.startswith("WARNING"))

    print(f"\n{'='*50}")
    print(f"Summary: {processed} processed, {skipped} skipped, {timeouts} timeout, {errors} errors/warnings")
    if timeout_files:
        print(f"Timeout files ({len(timeout_files)}):")
        for tf in timeout_files:
            print(f"  - {tf}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
