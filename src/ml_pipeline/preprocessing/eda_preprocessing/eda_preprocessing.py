import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from submodules.pyEDA.main import *

class EDAPreprocessing:
    def __init__(self, df, sg_window_size=11, sg_poly_order=3, lp_order=2, lp_cutoff=5.0, fs=700):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.lp_order = lp_order
        self.lp_cutoff = lp_cutoff
        self.fs = fs

    def process(self, use_pyEDA=False, sample_rate=128):
        if use_pyEDA:
            m, wd, eda_clean = process_statistical(self.df['eda'], use_scipy=True, sample_rate=128, new_sample_rate=40, segment_width=600, segment_overlap=0)
            return eda_clean
        else:
            self.df['eda'] = self.df['eda'].apply(self.smooth_eda).apply(self.lowpass_filter)
        return self.df

    def smooth_eda(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def lowpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.lp_cutoff / nyquist
        b, a = butter(self.lp_order, cutoff, btype='low')
        return filtfilt(b, a, signal)
