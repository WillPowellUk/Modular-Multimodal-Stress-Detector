from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader

DATASET_TYPE = "cv_5"

WRIST_CONFIG = "config_files/dataset/wesad_wrist_bvp_eda_configuration.json"

# Set the sensors to use
# SENSORS = "all"
SENSORS = "bvp_eda"

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

val_dataloader_params = {
    "batch_size": 1,
    "shuffle": False,
    "drop_last": False,
}
val_losocv_loader = LOSOCVSensorDataLoader(
    NON_BATCHED_FE, WRIST_CONFIG, **val_dataloader_params
)
val_dataloaders, val_input_dims = val_losocv_loader.get_data_loaders(
    NON_BATCHED_DATASETS_PATH, dataset_type=DATASET_TYPE, val_only=True
)

for i, dataloader in val_dataloaders.items():
    print(dataloader)
