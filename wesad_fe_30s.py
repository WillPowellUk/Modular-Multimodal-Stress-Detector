from src.ml_pipeline.data_loader import DataAugmenter
from src.ml_pipeline.feature_extraction import ManualFE

# WINDOW_LENGTH = 5
# SLIDING_LENGTH = WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
# SPLIT_LENGTH = WINDOW_LENGTH  # this will not sub-split the data

WINDOW_LENGTH = 30
SPLIT_LENGTH = int(
    WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
SLIDING_LENGTH = SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

WRIST_CONFIG = "config_files/dataset/wesad_wrist_configuration.json"

wrist_augmenter = DataAugmenter(
    "src/wesad/WESAD/cleaned/wrist_preprocessed.pkl", WRIST_CONFIG
)
batches = wrist_augmenter.augment_data(WINDOW_LENGTH, SLIDING_LENGTH)
wrist_splitted_segments = wrist_augmenter.split_segments(
    batches, WINDOW_LENGTH // SPLIT_LENGTH
)

WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/wrist_features.hdf5"

manual_fe = ManualFE(wrist_splitted_segments, WRIST_FE, WRIST_CONFIG)
manual_fe.extract_features()
