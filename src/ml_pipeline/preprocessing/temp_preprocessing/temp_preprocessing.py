import pandas as pd
from scipy.signal import savgol_filter


class TempPreprocessing:
    def __init__(self, df, window_size=31, poly_order=5, wrist=False):
        self.df = df
        self.window_size = window_size
        self.poly_order = poly_order
        self.wrist = wrist

    def process(self):
        key = "w_temp" if self.wrist else "temp"
        if isinstance(self.df , pd.Series):
            back_to_series = True
            self.df = pd.DataFrame(self.df)
            self.df.columns = [key]
        else:
            back_to_series = False
        temp_signal = self.df[key].values
        filtered_signal = self.temp_filter(temp_signal)
        self.df[key] = filtered_signal
        if back_to_series:
            return self.df[key]
        return

    def temp_filter(self, signal):
        return savgol_filter(signal, self.window_size, self.poly_order)
