# Source: htung0101/Physion-particles @ 5ac002a (upstream file) + local modifications used to preprocess the
# training data of this release; see patches/preprocessing_vs_upstream_5ac002a.patch and README.md.
import os
import random
from utils import mkdir, get_query_dir



data_path = get_query_dir("dpi_data_dir")

# All 8 Physion scenarios
scenario_names = ["Dominoes", "Support", "Collide", "Contain", "Drop", "Roll", "Link", "Drape"]
train_valid_ratio = 0.9

modes = ["train"]
for mode in modes:
    data_root = os.path.join(data_path, mode)
    for scenario_name in scenario_names:
        scene_path = os.path.join(data_root, scenario_name)

        # Skip if scenario directory doesn't exist
        if not os.path.exists(scene_path):
            print(f"Skipping {scenario_name} (not preprocessed yet)")
            continue

        trial_names = [file for file in os.listdir(scene_path) if not file.endswith(".txt") and not file.endswith(".swp")]

        if mode == "train":
            random.shuffle(trial_names)

            f = open(os.path.join(scene_path, "train.txt"), "w")
            total_trials = len(trial_names)
            n_trains = int(total_trials * train_valid_ratio)
            for trial_name in trial_names[:n_trains]:
                f.write(trial_name + "\n")

            f2 = open(os.path.join(scene_path, "valid.txt"), "w")
            for trial_name in trial_names[n_trains:]:
                f2.write(trial_name + "\n")

            print(mode, total_trials, "train, valid:", n_trains, ",", total_trials - n_trains)




