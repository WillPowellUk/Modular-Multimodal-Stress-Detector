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
from src.ml_pipeline.models.attention_models.co_attention_models import MOSCAN
from src.ml_pipeline.models.attention_models.ablation_study_models import *
from src.ml_pipeline.utils.utils import modify_nested_key


models = [
    MOSCANCrossAttention,
    MOSCANSlidingCasualBCSA,
    MOSCANSlidingCasualBCSACached,
    MOSCANSlidingBCSACached,
]

DATASET_TYPE = "losocv"

# Load Train Dataloaders for LOSOCV
TYPE = "wrist"
SENSORS = "bvp_w_eda_temp_w_acc"
# SENSORS = "all"

BATCHED_WINDOW_LENGTH = 30
BATCHED_SPLIT_LENGTH = int(
    BATCHED_WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
BATCHED_SLIDING_LENGTH = BATCHED_SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples
BATCHED_FE = f"src/wesad/WESAD/manual_fe/{TYPE}_manual_fe/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{TYPE}_features.hdf5"
BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/{TYPE}/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/{TYPE}_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{TYPE}_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/{TYPE}/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"


MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"
DATASET_CONFIG = f"config_files/dataset/wesad_{TYPE}_all_configuration.json"

### Unimodal first:
# modify_nested_key(DATASET_CONFIG, ['sensors', 'bvp'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_temp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_acc'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH, GROUP_LABELS=None, NAME='MoscanSelfAttention_BVP_Unimodal_WESAD')

# modify_nested_key(DATASET_CONFIG, ['sensors', 'bvp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_eda'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_temp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_acc'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH, GROUP_LABELS=None, NAME='MoscanSelfAttention_w_EDA_Unimodal_WESAD')

# modify_nested_key(DATASET_CONFIG, ['sensors', 'bvp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_temp'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_acc'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH, GROUP_LABELS=None, NAME='MoscanSelfAttention_temp_Unimodal_WESAD')

# modify_nested_key(DATASET_CONFIG, ['sensors', 'bvp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_temp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'w_acc'], True)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH, GROUP_LABELS=None, NAME='MoscanSelfAttention_w_acc_Unimodal_WESAD')

# Multimodal:
modify_nested_key(DATASET_CONFIG, ["sensors", "bvp"], True)
modify_nested_key(DATASET_CONFIG, ["sensors", "w_eda"], True)
modify_nested_key(DATASET_CONFIG, ["sensors", "w_temp"], True)
modify_nested_key(DATASET_CONFIG, ["sensors", "w_acc"], True)
for model in models:
    moscan(
        MOSCAN,
        MOSCAN_CONFIG,
        DATASET_CONFIG,
        DATASET_TYPE,
        BATCHED_FE,
        BATCHED_DATASETS_PATH,
        NON_BATCHED_FE,
        NON_BATCHED_DATASETS_PATH,
        NON_BATCHED_WINDOW_LENGTH,
        NON_BATCHED_SLIDING_LENGTH,
        NON_BATCHED_SPLIT_LENGTH,
        GROUP_LABELS=None,
        NAME=f"{model.__name__}_Multimodal_WESAD",
    )

### Unimodal first:
# modify_nested_key(DATASET_CONFIG, ['sensors', 'acc'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'ecg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'emg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'resp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'temp'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH)

# modify_nested_key(DATASET_CONFIG, ['sensors', 'acc'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'ecg'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'emg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'resp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'temp'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH)

# modify_nested_key(DATASET_CONFIG, ['sensors', 'acc'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'ecg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'emg'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'resp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'temp'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH)

# modify_nested_key(DATASET_CONFIG, ['sensors', 'acc'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'ecg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'emg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'eda'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'resp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'temp'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH)

# modify_nested_key(DATASET_CONFIG, ['sensors', 'acc'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'ecg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'emg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'resp'], True)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'temp'], False)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH)

# modify_nested_key(DATASET_CONFIG, ['sensors', 'acc'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'ecg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'emg'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'eda'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'resp'], False)
# modify_nested_key(DATASET_CONFIG, ['sensors', 'temp'], True)
# moscan(MOSCANSelfAttention, MOSCAN_CONFIG, DATASET_CONFIG, DATASET_TYPE, BATCHED_FE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, NON_BATCHED_WINDOW_LENGTH, NON_BATCHED_SLIDING_LENGTH, NON_BATCHED_SPLIT_LENGTH)
