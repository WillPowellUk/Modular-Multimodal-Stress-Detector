from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader


'''
EDA
'''
SENSORS = 'eda'
WINDOW_LENGTH = 5
SLIDING_LENGTH = WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
SPLIT_LENGTH = WINDOW_LENGTH  # this will not sub-split the data

CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

dataloader_params = {
    "batch_size": 32,
    "shuffle": False,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(FE, CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s"
)

############################################################################################################


WINDOW_LENGTH = 30
SPLIT_LENGTH = int(
    WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
SLIDING_LENGTH = SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

dataloader_params = {
    "batch_size": 32,
    "shuffle": False,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(FE, CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s"
)

'''
BVP
'''
SENSORS = 'bvp'
WINDOW_LENGTH = 5
SLIDING_LENGTH = WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
SPLIT_LENGTH = WINDOW_LENGTH  # this will not sub-split the data

CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

dataloader_params = {
    "batch_size": 32,
    "shuffle": False,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(FE, CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s"
)

############################################################################################################


WINDOW_LENGTH = 30
SPLIT_LENGTH = int(
    WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
SLIDING_LENGTH = SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

dataloader_params = {
    "batch_size": 32,
    "shuffle": False,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(FE, CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s"
)


'''
BVP + EDA
'''
SENSORS = 'bvp_eda'
WINDOW_LENGTH = 5
SLIDING_LENGTH = WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
SPLIT_LENGTH = WINDOW_LENGTH  # this will not sub-split the data

CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

dataloader_params = {
    "batch_size": 32,
    "shuffle": False,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(FE, CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s"
)

############################################################################################################


WINDOW_LENGTH = 30
SPLIT_LENGTH = int(
    WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
SLIDING_LENGTH = SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

CONFIG = f"config_files/dataset/ubfc_{SENSORS}_configuration.json"
FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

dataloader_params = {
    "batch_size": 32,
    "shuffle": False,
    # 'num_workers': 4
}
losocv_loader = LOSOCVSensorDataLoader(FE, CONFIG, **dataloader_params)

# Prepare the datasets
DATASETS_PATH = losocv_loader.prepare_datasets(
    f"src/ubfc_phys/UBFC-PHYS/datasets/manual_fe/{SENSORS}/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s"
)