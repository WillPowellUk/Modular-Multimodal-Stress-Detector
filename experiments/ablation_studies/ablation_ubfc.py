from experiments.manual_fe.moscan.moscan import moscan
from src.ml_pipeline.models.attention_models.ablation_study_models import *
from src.ml_pipeline.utils.utils import modify_key

models = [
    MOSCANSelfAttention,
    MOSCANCrossAttention,
    MOSCANSlidingCasualBCSA,
    MOSCANSlidingCasualBCSACached,
    MOSCANSlidingBCSACached,
]

# Set either losocv or kfold
# DATASET_TYPE = "losocv"
DATASET_TYPE = "cv_7"

# Load Train Dataloaders for LOSOCV
BATCHED_WINDOW_LENGTH = 30
BATCHED_SPLIT_LENGTH = int(
    BATCHED_WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
BATCHED_SLIDING_LENGTH = BATCHED_SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples
BATCHED_FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/all_features.hdf5"

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"

MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"

# T1 v (T2 + T3)
# Configure labels for group
GROUP_LABELS = {2: [3]}  # Label 3 is merged into label 2
modify_key(MOSCAN_CONFIG, "num_classes", 2)

### Unimodal first:
SENSORS = "bvp"
DATASET_CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
moscan(
    MOSCANSelfAttention,
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
    GROUP_LABELS=GROUP_LABELS,
    NAME="MoscanSelfAttention_BVP_Unimodal",
)

SENSORS = "eda"
DATASET_CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
moscan(
    MOSCANSelfAttention,
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
    GROUP_LABELS=GROUP_LABELS,
    NAME="MoscanSelfAttention_EDA_Unimodal",
)

#### Both modalities
SENSORS = "bvp_eda"
DATASET_CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
DATASET_CONFIG = "config_files/dataset/ubfc_bvp_eda_configuration.json"
for model in models:
    print(f"Running {model.__name__}")
    moscan(
        model,
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
        GROUP_LABELS=GROUP_LABELS,
        NAME=model.__name__,
    )

# T1 v v T2 v T3
# Configure labels for group
GROUP_LABELS = None
modify_key(MOSCAN_CONFIG, "num_classes", 3)

### Unimodal first:
SENSORS = "bvp"
DATASET_CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
moscan(
    MOSCANSelfAttention,
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
    GROUP_LABELS=GROUP_LABELS,
    NAME="MoscanSelfAttention_BVP_Unimodal",
)

SENSORS = "eda"
DATASET_CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
moscan(
    MOSCANSelfAttention,
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
    GROUP_LABELS=GROUP_LABELS,
    NAME="MoscanSelfAttention_EDA_Unimodal",
)

#### Both modalities
SENSORS = "bvp_eda"
DATASET_CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
DATASET_CONFIG = "config_files/dataset/ubfc_bvp_eda_configuration.json"
for model in models:
    print(f"Running {model.__name__}")
    moscan(
        model,
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
        GROUP_LABELS=GROUP_LABELS,
        NAME=model.__name__,
    )
