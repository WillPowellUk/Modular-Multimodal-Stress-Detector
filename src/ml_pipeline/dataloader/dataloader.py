import pandas as pd
import numpy as np
import pickle
import json
import re

class DataLoader:
    def __init__(self, dataframe_path, config_path):
        self.dataframe = self.load_dataframe(dataframe_path)
        self.config = self.load_config(config_path)

        # Set the sampling rate to the maximum sampling rate
        self.sampling_rate = max(self.config.values()) 
        
    def load_dataframe(self, dataframe_path):
        with open(dataframe_path, 'rb') as file:
            dataframe = pickle.load(file)
        return dataframe

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    def segment_data(self, window_size=60, sliding_length=5):
        segments = []
        grouped = self.dataframe.groupby('sid')
        for sid, group in grouped:
            start_idx = 0
            sample_rate = self.sampling_rate
            end_idx = window_size * sample_rate
            sliding_step = sliding_length * sample_rate

            while end_idx <= len(group):
                segment = group.iloc[start_idx:end_idx]
                if first_segment or start_idx % (window_size * sample_rate) == 0:
                    segment['is_augmented'] = False
                    first_segment = False
                else:
                    segment['is_augmented'] = True
                if len(segment) == window_size * sample_rate:
                    segments.append(segment)
                start_idx += sliding_step
                end_idx += sliding_step
        return segments