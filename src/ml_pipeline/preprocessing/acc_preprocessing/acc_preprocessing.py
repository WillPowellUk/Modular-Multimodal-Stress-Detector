import pandas as pd
from scipy.signal import savgol_filter

class AccPreprocessing:
    def __init__(self, df, window_size=31, poly_order=5, wrist=False):
        self.df = df
        self.wrist = wrist
        self.window_size = window_size
        self.poly_order = poly_order

    def process(self):
        if self.wrist:
            for acc in ['w_acc_x', 'w_acc_y', 'w_acc_z']:
                acc_signal = self.df[acc].values
                filtered_signal = self.acc_filter(acc_signal)
                self.df[acc] = filtered_signal
        else:
            for acc in ['acc1', 'acc2', 'acc3']:
                acc_signal = self.df[acc].values
                filtered_signal = self.acc_filter(acc_signal)
                self.df[acc] = filtered_signal
        return self.df

    def acc_filter(self, signal):
        return savgol_filter(signal, self.window_size, self.poly_order)