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



from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader

DATASET_TYPE = "cv_5"

WRIST_CONFIG = "config_files/dataset/wesad_wrist_bvp_eda_configuration.json"

# Set the sensors to use
# SENSORS = "all"
SENSORS = "bvp_eda"

# Load Val Dataloaders for LOSOCV
VAL_WINDOW_LENGTH = 5
VAL_SLIDING_LENGTH = VAL_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
VAL_SPLIT_LENGTH = VAL_WINDOW_LENGTH  # this will not sub-split the data
VAL_WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{VAL_WINDOW_LENGTH}s_{VAL_SLIDING_LENGTH}s_{VAL_SPLIT_LENGTH}s/wrist_features.hdf5"
VAL_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{VAL_WINDOW_LENGTH}s_{VAL_SLIDING_LENGTH}s_{VAL_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
VAL_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{VAL_WINDOW_LENGTH}s_{VAL_SLIDING_LENGTH}s_{VAL_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

val_dataloader_params = {
    "batch_size": 1,
    "shuffle": False,
    "drop_last": False,
}
val_losocv_loader = LOSOCVSensorDataLoader(
    VAL_WRIST_FE, WRIST_CONFIG, **val_dataloader_params
)
val_dataloaders, val_input_dims = val_losocv_loader.get_data_loaders(
    VAL_DATASETS_PATH, dataset_type=DATASET_TYPE, val_only=True
)

for i, dataloader in val_dataloaders.items():
    print(dataloader)
