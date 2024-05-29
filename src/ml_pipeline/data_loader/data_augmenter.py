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
        for sid, group in grouped:
            start_idx = 0
            sample_rate = self.sampling_rate
            end_idx = window_size * sample_rate
            sliding_step = sliding_length * sample_rate

            while end_idx <= len(group):
                segment = group.iloc[start_idx:end_idx].copy()  # Make a copy of the slice
                if start_idx % (window_size * sample_rate) == 0:
                    segment.loc[:, 'is_augmented'] = False
                else:
                    segment.loc[:, 'is_augmented'] = True
                if len(segment) == window_size * sample_rate:
                    segments.append(segment)
                start_idx += sliding_step
                end_idx += sliding_step
        print('Segmentation complete.')
        return segments