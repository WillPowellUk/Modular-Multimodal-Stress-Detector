from src.ml_pipeline.data_loader import DataAugmenter
from src.ml_pipeline.feature_extraction import ManualFE

WINDOW_LENGTH = 30
SPLIT_LENGTH = int(WINDOW_LENGTH / 6) # this will sub-split the data 6 times each of 5 seconds
SLIDING_LENGTH = SPLIT_LENGTH # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

CONFIG = 'config_files/dataset/ubfc_bvp_w_eda_configuration.json'

augmenter = DataAugmenter(
    "src/ubfc_phys/UBFC-PHYS/cleaned/preprocessed.pkl", CONFIG
)
batches = augmenter.augment_data(WINDOW_LENGTH, SLIDING_LENGTH)
splitted_segments = augmenter.split_segments(
    batches, WINDOW_LENGTH // SPLIT_LENGTH
)

FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

manual_fe = ManualFE(splitted_segments, FE, CONFIG)
manual_fe.extract_features()


############################################################################################################


WINDOW_LENGTH = 5
SLIDING_LENGTH = WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
SPLIT_LENGTH = WINDOW_LENGTH  # this will not sub-split the data

CONFIG = 'config_files/dataset/ubfc_bvp_w_eda_configuration.json'

augmenter = DataAugmenter(
    "src/ubfc_phys/UBFC-PHYS/cleaned/preprocessed.pkl", CONFIG
)
batches = augmenter.augment_data(WINDOW_LENGTH, SLIDING_LENGTH)
splitted_segments = augmenter.split_segments(
    batches, WINDOW_LENGTH // SPLIT_LENGTH
)

FE = f"src/ubfc_phys/UBFC-PHYS/manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/all_features.hdf5"

manual_fe = ManualFE(splitted_segments, FE, CONFIG)
manual_fe.extract_features()
