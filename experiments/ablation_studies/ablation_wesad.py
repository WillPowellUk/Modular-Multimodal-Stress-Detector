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

from experiments.manual_fe.moscan.moscan import moscan
from src.ml_pipeline.models.attention_models.ablation_study_models import *


models = [MOSCANSelfAttention, MOSCANCrossAttention, MOSCANSlidingCasualBCSA,MOSCANSlidingCasualBCSACached, MOSCANSlidingBCSACached]


# Load Train Dataloaders for LOSOCV
BATCHED_WINDOW_LENGTH = 30
BATCHED_SPLIT_LENGTH = int(
    BATCHED_WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
BATCHED_SLIDING_LENGTH = BATCHED_SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples
BATCHED_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"



MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"

### Unimodal first:
DATASET_CONFIG = "config_files/dataset/ubfc_bvp_configuration.json"
moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, name='MoscanSelfAttention_BVP_Unimodal')
DATASET_CONFIG = "config_files/dataset/ubfc_w_eda_configuration.json"
moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, name="MoscanSelfAttention_EDA_Unimodal")

#### Both modalities
DATASET_CONFIG = "config_files/dataset/ubfc_bvp_eda_configuration.json"
for model in models:
    moscan(model, MOSCAN_CONFIG, name=model.__name__)