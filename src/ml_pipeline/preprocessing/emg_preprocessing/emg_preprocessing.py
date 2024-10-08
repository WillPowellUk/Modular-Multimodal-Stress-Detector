import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
import neurokit2 as nk


class EMGPreprocessing:
    def __init__(
        self, df, sg_window_size=11, sg_poly_order=3, lp_order=3, lp_cutoff=0.5, fs=700
    ):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.lp_order = lp_order
        self.lp_cutoff = lp_cutoff
        self.fs = fs

    def process(self, use_neurokit=False):
        if isinstance(self.df , pd.Series):
            back_to_series = True
            self.df = pd.DataFrame(self.df)
            self.df.columns = ["emg"]
        else:
            back_to_series = False
        
        if use_neurokit:
            signals = nk.emg_process(self.df["emg"], sampling_rate=self.fs)
            self.df["emg"] = signals["EMG_Clean"]
        else:
            emg_signal = self.df["emg"].values
            smoothed_signal = self.smooth_emg(emg_signal)
            filtered_signal = self.lowpass_filter(smoothed_signal)
            self.df["emg"] = filtered_signal

        if back_to_series:
            return self.df["emg"]
        else:
            return self.df

    def smooth_emg(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def lowpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.lp_cutoff / nyquist
        b, a = butter(self.lp_order, cutoff, btype="low")
        return filtfilt(b, a, signal)
    
    def bandpass_filter(self, signal, lowcut, highcut):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(self.lp_order, [low, high], btype="band")
        return filtfilt(b, a, signal)
    
    def highpass_filter(self, signal, highcut):
        nyquist = 0.5 * self.fs
        high = highcut / nyquist
        b, a = butter(self.lp_order, high, btype="high")
        return filtfilt(b, a, signal)
