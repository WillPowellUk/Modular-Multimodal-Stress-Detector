
import os
import sys
from datetime import datetime
import torch

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the desired directory
target_dir = current_dir
while "src" not in os.listdir(target_dir) and target_dir != os.path.dirname(target_dir):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "src" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError("Could not find 'src' directory in the path hierarchy")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the desired directory
target_dir = current_dir
while "experiments" not in os.listdir(target_dir) and target_dir != os.path.dirname(
    target_dir
):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "experiments" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError("Could not find 'experiments' directory in the path hierarchy")

from src.ml_pipeline.utils.utils import get_active_key
import pickle
import h5py
import numpy as np
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class FeatureVisualization:
    def __init__(self, features_path, config_path):
        self.features_path = features_path
        self.config_path = config_path
        self.include_sensors = get_active_key(config_path, "sensors")
        self.include_features = get_active_key(config_path, "features", recursive=True)

    def save_dataset(self):
        data = []
        labels = []
        with h5py.File(self.features_path, "r") as hdf5_file:
            all_data = []
            all_labels = []
            sample_idx = 0
            for subject in hdf5_file.keys():
                subject_id = int(subject.split("_")[1])

                for aug in hdf5_file[subject].keys():
                    is_augmented = aug.split("_")[1] == "True"

                    if is_augmented:
                        continue  # skip augmented data

                    for label in hdf5_file[subject][aug].keys():
                        for batch in hdf5_file[subject][aug][label].keys():
                            for sensor in hdf5_file[subject][aug][label][batch].keys():
                                if sensor not in self.include_sensors:  # skip unwanted sensors
                                    continue
                                for feature in hdf5_file[subject][aug][label][batch][sensor].keys():
                                    # if feature not in self.include_features:  # skip unwanted features
                                    #     continue
                                    feature_data = hdf5_file[subject][aug][label][batch][sensor][feature][:]
                                    data.append(feature_data)
                
            
        # Convert lists to arrays
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)

        # Perform PCA to reduce dimensions to 2D
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(all_data)

        # Plot the 2D visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=all_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Labels')
        plt.title('2D Visualization of High-Dimensional Features')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

WINDOW_LENGTH = 60
SLIDING_LENGTH = 5
SPLIT_LENGTH = 10 # this will split each 60 second segments into 6 x 10 second segments
WRIST_CONFIG = 'config_files/dataset/wesad_wrist_configuration.json'
WRIST_FE = f'src/wesad/WESAD/manual_fe/wrist_manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/wrist_features.hdf5'

visualizer = FeatureVisualization(WRIST_CONFIG, WRIST_FE)
visualizer.visualize()
