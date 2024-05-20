import pandas as pd
import numpy as np

class TempFeatureExtractor:
    def __init__(self, temp_data: pd.DataFrame):
        self.temp_data = temp_data

    def extract_features(self):
        features = {}

        # Mean and STD of TEMP
        features['mean_temp'] = np.mean(self.temp_data['TEMP'])
        features['std_temp'] = np.std(self.temp_data['TEMP'])

        # Min and max of TEMP
        features['min_temp'] = np.min(self.temp_data['TEMP'])
        features['max_temp'] = np.max(self.temp_data['TEMP'])

        # Slope and dynamic range of TEMP
        features['slope_temp'] = np.polyfit(range(len(self.temp_data['TEMP'])), self.temp_data['TEMP'], 1)[0]
        features['range_temp'] = np.max(self.temp_data['TEMP']) - np.min(self.temp_data['TEMP'])

        return pd.DataFrame([features])