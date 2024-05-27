import pandas as pd
import numpy as np

class AugmentedDataLoader:
    def __init__(self, features_path, batch_size, window_size=60, sliding_length=5):
        self.features = self.load_features(features_path)
        self.batch_size = batch_size
        self.window_size = window_size
        self.sliding_length = sliding_length

    def load_features(self, features_path):
        return pd.read_pickle(features_path)

    def generate_batches(self, exclude_subjects=None):
        segments = self.segment_data(exclude_subjects)
        np.random.shuffle(segments)
        batches = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
        return batches

    def segment_data(self, exclude_subjects=None):
        if exclude_subjects is None:
            exclude_subjects = []
        segments = self.features[~self.features['sid'].isin(exclude_subjects)]
        return segments.to_dict('records')

# Function to perform on-the-fly augmentation
def augment_data(segment, window_size, sliding_length):
    augmented_segments = []
    start_idx = 0
    end_idx = start_idx + window_size
    while end_idx <= len(segment):
        augmented_segments.append(segment[start_idx:end_idx])
        start_idx += sliding_length
        end_idx = start_idx + window_size
    return augmented_segments
