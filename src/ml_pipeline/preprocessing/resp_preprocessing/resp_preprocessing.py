import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

class RespPreprocessing:
    def __init__(self, df, sg_window_size=11, sg_poly_order=3, bw_order=3, bw_lowcut=0.1, bw_highcut=0.35, fs=700):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.bw_order = bw_order
        self.bw_lowcut = bw_lowcut
        self.bw_highcut = bw_highcut
        self.fs = fs

    def process(self):
        resp_signal = self.df['resp'].values
        smoothed_signal = self.smooth_resp(resp_signal)
        filtered_signal = self.butterworth_filter(smoothed_signal)
        self.df['resp'] = filtered_signal
        return self.df

    def smooth_resp(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def butterworth_filter(self, signal):
        nyquist = 0.5 * self.fs
        low = self.bw_lowcut / nyquist
        high = self.bw_highcut / nyquist
        b, a = butter(self.bw_order, [low, high], btype='band')
        return filtfilt(b, a, signal)