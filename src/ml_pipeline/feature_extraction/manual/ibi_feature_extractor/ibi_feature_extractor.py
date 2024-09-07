import pandas as pd
import numpy as np
import neurokit2 as nk

class IBIFeatureExtractor:
    def __init__(self, rr_intervals: pd.DataFrame, sampling_rate: int):
        """
        Initializes the IBIFeatureExtractor.
        
        Parameters:
        - rr_intervals (pd.DataFrame): A DataFrame containing R-R intervals in milliseconds.
        - sampling_rate (int): The sampling rate of the data in Hz.
        """
        self.rr_intervals = rr_intervals.values.flatten() * 1000 # Convert to milliseconds
        self.sampling_rate = sampling_rate

    def extract_features(self):
        features = {}

        # Mean and STD of IBI
        features["mean_ibi"] = np.mean(self.rr_intervals)
        features["std_ibi"] = np.std(self.rr_intervals)

        # Min and max of IBI
        features["min_ibi"] = np.min(self.rr_intervals)
        features["max_ibi"] = np.max(self.rr_intervals)

        # Range of IBI
        features["range_ibi"] = np.max(self.rr_intervals) - np.min(self.rr_intervals)

        # Convert IBI to R-peaks (cumulative sum gives time of R-peaks)
        r_peaks = np.cumsum(self.rr_intervals)
        
        # Create a peaks dictionary expected by the neurokit hrv function
        peaks = {"RRI": self.rr_intervals, "RRI_Time": r_peaks}

        # Use neurokit to extract HRV features
        hrv_features = nk.hrv(peaks, sampling_rate=self.sampling_rate, show=False)

        # Append HRV features to the main features dictionary
        for key, value in hrv_features.items():
            features[key] = value

        return pd.DataFrame([features])
