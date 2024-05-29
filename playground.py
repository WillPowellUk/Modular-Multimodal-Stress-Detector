from src.ml_pipeline.data_loader import DataAugmenter
from src.ml_pipeline.feature_extraction import ManualFE

WINDOW_LENGTH = 60
SLIDING_LENGTH = 5

CHEST_CONFIG = 'src/wesad/wesad_chest_configuration.json'
WRIST_CONFIG = 'src/wesad/wesad_wrist_configuration.json'

wrist_augmenter = DataAugmenter('src/wesad/WESAD/cleaned/wrist_preprocessed.pkl', WRIST_CONFIG)
batches = wrist_augmenter.segment_data(WINDOW_LENGTH, SLIDING_LENGTH)

manual_fe = ManualFE(batches, 'src/wesad/WESAD/manual_fe/wrist_manual_fe.hdf5', WRIST_CONFIG)
manual_fe.extract_features()