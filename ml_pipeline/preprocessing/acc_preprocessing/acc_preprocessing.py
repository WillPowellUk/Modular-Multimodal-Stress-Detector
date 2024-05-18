import pandas as pd
from scipy.signal import savgol_filter

class ACCPreprocessing:
    def __init__(self, df, window_size=31, poly_order=5):
        self.df = df
        self.window_size = window_size
        self.poly_order = poly_order

    def process(self):
        self.df['acc'] = self.df['acc'].apply(self.acc_filter)
        return self.df

    def acc_filter(self, signal):
        return savgol_filter(signal, self.window_size, self.poly_order)
