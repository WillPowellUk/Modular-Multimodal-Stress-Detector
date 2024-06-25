import pandas as pd
import numpy as np
from math import floor
import pickle
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key

class DataAugmenter:
    def __init__(self, dataframe_path, config_path):
        self.dataframe = self.load_dataframe(dataframe_path)
        self.sampling_rate = get_max_sampling_rate(config_path)
        self.labels = get_active_key(config_path, 'labels')
        
    def load_dataframe(self, dataframe_path):
        with open(dataframe_path, 'rb') as file:
            dataframe = pickle.load(file)
        return dataframe
    
    def augment_data(self, window_size=60, sliding_length=5):
        # iterate through each subject, copying the original data across in segments first (non-augmented) a
        # and then creating synthetic samples by iterating through the data again and again at a cumulative offset of sliding_step until complete
        print('Segmenting data...')
        grouped = self.dataframe.groupby('sid')
        sample_rate = self.sampling_rate
        window_step = window_size * sample_rate
        sliding_step = sliding_length * sample_rate
        num_of_iterations = floor(window_size / sliding_length)

        segments = []
        for sid, group in grouped:
            start_idx = 0
            end_idx = window_step

            for i in range(num_of_iterations):
                while end_idx <= len(group):
                    segment = group.iloc[start_idx:end_idx].copy()
                    if str(segment['label'].iloc[0]) in self.labels:
                        if segment['label'].nunique() == 1:
                            segment.loc[:, 'is_augmented'] = False if start_idx % (window_size * sample_rate) == 0 else True
                            if len(segment) == window_size * sample_rate:
                                segments.append(segment)
                    start_idx += window_step
                    end_idx += window_step
                start_idx = (i+1) * sliding_step
                end_idx = start_idx + window_step
        print(f'Number of segments: {len(segments)}')
        print(f'Synthetic to non-synthetic ratio: {num_of_iterations-1}:{1}')
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