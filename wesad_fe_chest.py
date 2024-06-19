from src.ml_pipeline.data_loader import DataAugmenter
from src.ml_pipeline.feature_extraction import ManualFE

WINDOW_LENGTH = 60
SLIDING_LENGTH = 5
SPLIT_LENGTH = 10 # this will split each 60 second segments into 6 x 10 second segments

CHEST_CONFIG = 'config_files/dataset/wesad_chest_configuration.json'

chest_augmenter = DataAugmenter('src/wesad/WESAD/cleaned/chest_preprocessed.pkl', CHEST_CONFIG) 
batches = chest_augmenter.augment_data(WINDOW_LENGTH, SLIDING_LENGTH)
chest_splitted_segments = chest_augmenter.split_segments(batches, WINDOW_LENGTH//SPLIT_LENGTH)

CHEST_FE = f'src/wesad/WESAD/manual_fe/chest_manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/wrist_features.hdf5'

manual_fe = ManualFE(chest_splitted_segments, CHEST_FE, CHEST_CONFIG)
manual_fe.extract_features()