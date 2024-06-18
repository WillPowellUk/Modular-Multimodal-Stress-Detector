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

    def augment_data(self, window_size=60, sliding_length=5):
        print('Segmenting data...')
        segments = []
        grouped = self.dataframe.groupby('sid')
        for sid, group in grouped:
            start_idx = 0
            sample_rate = self.sampling_rate
            end_idx = window_size * sample_rate
            sliding_step = sliding_length * sample_rate

            while end_idx <= len(group):
                segment = group.iloc[start_idx:end_idx].copy()
                if segment['label'].nunique() == 1:  # Check if all labels in the segment are the same
                    segment.loc[:, 'is_augmented'] = False if start_idx % (window_size * sample_rate) == 0 else True
                    if len(segment) == window_size * sample_rate:
                        segments.append(segment)
                start_idx += sliding_step
                end_idx += sliding_step
        return segments

    def split_segments(self, segments, num_splits):
        """
        Splits each segment into a specified number of splits, remaining grouped.

        Args:
            segments (list of pd.DataFrame): List of segmented data.
            num_splits (int): Number of splits for each segment.

        Returns:
            list of pd.DataFrame: List of split segments.
        """
        print('Splitting segments...')
        split_segments = []
        for segment in segments:
            segment_length = len(segment)
            split_length = segment_length // num_splits

            split_segment = []
            for i in range(num_splits):
                start_idx = i * split_length
                end_idx = start_idx + split_length

                # Only include full splits
                if end_idx <= segment_length:
                    split = segment.iloc[start_idx:end_idx].copy()
                    split_segment.append(split)
            split_segments.append(split_segment)
        print('Splitting complete.')
        return split_segments