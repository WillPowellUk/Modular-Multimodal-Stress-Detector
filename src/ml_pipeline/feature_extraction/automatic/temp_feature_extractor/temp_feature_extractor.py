import pandas as pd
import numpy as np


class TempFeatureExtractor:
    def __init__(self, temp_data: pd.DataFrame, sampling_rate: int):
        self.temp_data = temp_data.values
        self.sampling_rate = sampling_rate

    def extract_features(self):
        features = {}

        # Mean and STD of TEMP
        features["mean_temp"] = np.mean(self.temp_data)
        features["std_temp"] = np.std(self.temp_data)

        # Min and max of TEMP
        features["min_temp"] = np.min(self.temp_data)
        features["max_temp"] = np.max(self.temp_data)

        # Slope and dynamic range of TEMP
        features["slope_temp"] = np.polyfit(
            range(len(self.temp_data)), self.temp_data, 1
        )[0]
        features["range_temp"] = np.max(self.temp_data) - np.min(self.temp_data)

        return pd.DataFrame([features])
