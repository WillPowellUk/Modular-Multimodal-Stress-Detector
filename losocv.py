from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader

n_folds = 5

WINDOW_LENGTH = 5
SLIDING_LENGTH = WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
SPLIT_LENGTH = WINDOW_LENGTH  # this will not sub-split the data

SENSORS = 'bvp_eda_temp'

# WRIST_CONFIG = 'config_files/dataset/wesad_wrist_configuration.json'
WRIST_CONFIG = f"config_files/dataset/wesad_wrist_{SENSORS}_configuration.json"
WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/wrist_features.hdf5"
dataloader_params = {
    "batch_size": 32,
    "shuffle": True,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(WRIST_FE, WRIST_CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s",
    n_folds=n_folds,
)


WINDOW_LENGTH = 30
SPLIT_LENGTH = int(
    WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
SLIDING_LENGTH = SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

# WRIST_CONFIG = 'config_files/dataset/wesad_wrist_configuration.json'
WRIST_CONFIG = f"config_files/dataset/wesad_wrist_{SENSORS}_configuration.json"
WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/wrist_features.hdf5"
dataloader_params = {
    "batch_size": 32,
    "shuffle": True,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(WRIST_FE, WRIST_CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s",
    n_folds=n_folds,
)
