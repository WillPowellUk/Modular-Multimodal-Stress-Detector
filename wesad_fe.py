from src.ml_pipeline.data_loader import DataAugmenter
from src.ml_pipeline.feature_extraction import ManualFE

WINDOW_LENGTH = 60
SLIDING_LENGTH = 5
SPLIT_LENGTH = 10  # this will split each 60 second segments into 6 x 10 second segments

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
