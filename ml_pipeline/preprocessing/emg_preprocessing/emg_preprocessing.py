import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

class EMGPreprocessing:
    def __init__(self, df, sg_window_size=11, sg_poly_order=3, lp_order=3, lp_cutoff=0.5, fs=700):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.lp_order = lp_order
        self.lp_cutoff = lp_cutoff
        self.fs = fs

    def process(self):
        self.df['emg'] = self.df['emg'].apply(self.smooth_emg).apply(self.lowpass_filter)
        return self.df

    def smooth_emg(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def lowpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.lp_cutoff / nyquist
        b, a = butter(self.lp_order, cutoff, btype='low')
        return filtfilt(b, a, signal)
