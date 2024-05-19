import pandas as pd
from scipy.signal import savgol_filter

class TempPreprocessing:
    def __init__(self, df):
        self.df = df

    def process(self):
        self.df['temp'] = self.df['temp'].apply(self.temp_filter)
        return self.df

    def temp_filter(self, signal, window_size=11, poly_order=3):
        # Apply Savitzky–Golay filter
        smoothed_signal = savgol_filter(signal, window_size, poly_order)
        return smoothed_signal