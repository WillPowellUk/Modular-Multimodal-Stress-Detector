import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, features_path, window_size=60, sliding_length=5, **params):
        self.features = self.load_features(features_path)
        self.window_size = window_size
        self.sliding_length = sliding_length
        self.params = params

    def load_features(self, features_path):
        return pd.read_pickle(features_path)

    def get_dataset(self, exclude_subjects=None):
        pass
        