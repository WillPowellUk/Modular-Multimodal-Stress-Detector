import pandas as pd
import numpy as np
import pickle
from src.ml_pipeline.utils.utils import get_max_sampling_rate

class DataAugmenter:
    def __init__(self, dataframe_path, config_path):
        self.dataframe = self.load_dataframe(dataframe_path)
        self.sampling_rate = get_max_sampling_rate(config_path)
        
    def load_dataframe(self, dataframe_path):
        with open(dataframe_path, 'rb') as file:
            dataframe = pickle.load(file)
        return dataframe

    def segment_data(self, window_size=60, sliding_length=5):
        print('Segmenting data...')
        segments = []
        grouped = self.dataframe.groupby('sid')

        sample_rate = self.sampling_rate
        window_size_samples = window_size * sample_rate
        sliding_step_samples = sliding_length * sample_rate

        for sid, group in grouped:
            # Compute the number of segments
            num_segments = (len(group) - window_size_samples) // sliding_step_samples + 1

            for i in range(num_segments):
                start_idx = i * sliding_step_samples
                end_idx = start_idx + window_size_samples
                segment = group.iloc[start_idx:end_idx].copy()  # Make a copy of the slice

                segment['is_augmented'] = i != 0
                segments.append(segment)

        print('Segmentation complete.')
        return segments
