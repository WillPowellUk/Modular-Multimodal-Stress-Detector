import pandas as pd
import numpy as np
import json
import re
import pickle

class DataLoader:
    def __init__(self, dataframe_path, config_file, batch_size):
        self.dataframe = self.load_dataframe(dataframe_path)
        self.config = self.load_config(config_file)
        self.batch_size = batch_size
        self.window_size = 60  # 60 seconds window
        self.sliding_length = 5  # 5 seconds sliding length
        
    def load_dataframe(self, dataframe_path):
        with open(dataframe_path, 'rb') as file:
            dataframe = pickle.load(file)
        return dataframe

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config

    def get_sample_rate(self, column):
        for sensor, rate in self.config.items():
            if re.match(sensor, column):
                return rate
        return None

    def segment_data(self):
        segments = []
        grouped = self.dataframe.groupby('sid')
        for sid, group in grouped:
            start_idx = 0
            sample_rate = self.get_sample_rate(group.columns[1])
            end_idx = start_idx + self.window_size * sample_rate
            sliding_step = self.sliding_length * sample_rate

            while end_idx <= len(group):
                segment = group.iloc[start_idx:end_idx]
                if len(segment) == self.window_size * sample_rate:
                    segments.append(segment)
                start_idx += sliding_step
                end_idx = start_idx + self.window_size * sample_rate
        return segments

    def generate_batches(self):
        segments = self.segment_data()
        np.random.shuffle(segments)  # Shuffle the segments to ensure randomness
        batches = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
        return batches