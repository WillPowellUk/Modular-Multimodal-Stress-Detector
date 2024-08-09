import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from submodules.pyEDA.main import process_statistical


class EDAPreprocessing:
    def __init__(
        self,
        df,
        sg_window_size=11,
        sg_poly_order=3,
        lp_order=2,
        lp_cutoff=5.0,
        fs=700,
        wrist=False,
    ):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.lp_order = lp_order
        self.lp_cutoff = lp_cutoff
        self.fs = fs
        self.wrist = wrist

    def process(self, use_pyEDA=False, use_neurokit=False, sample_rate=128):
        key = "w_eda" if self.wrist else "eda"
        if use_pyEDA:
            m, wd, eda_clean = process_statistical(
                self.df[key],
                use_scipy=True,
                sample_rate=sample_rate,
                new_sample_rate=40,
                segment_width=600,
                segment_overlap=0,
            )
            self.df[key] = eda_clean
        if use_neurokit:
            eda_clean = eda_clean(
                eda_signal, sampling_rate=sample_rate, method="neurokit"
            )
            self.df[key] = eda_clean
        else:
            eda_signal = self.df[key].values
            if not self.wrist:
                eda_signal = self.smooth_eda(eda_signal)
            filtered_signal = self.lowpass_filter(eda_signal)
            self.df[key] = filtered_signal
        return self.df

    def smooth_eda(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def lowpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        if self.wrist:
            cutoff = 1.0 / nyquist  # Cut-off frequency of 1 Hz for wrist
            order = 6  # Butterworth lowpass filter of order 6 for wrist
        else:
            cutoff = self.lp_cutoff / nyquist  # Custom cut-off frequency
            order = self.lp_order  # Custom filter order
        b, a = butter(order, cutoff, btype="low")
        return filtfilt(b, a, signal)
