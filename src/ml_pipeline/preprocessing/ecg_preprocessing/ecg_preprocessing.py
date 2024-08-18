import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, freqz, lfilter
import neurokit2 as nk
import matplotlib.pyplot as plt


class ECGPreprocessing:
    def __init__(
        self,
        df,
        sg_window_size=11,
        sg_poly_order=3,
        bw_order=3,
        bw_lowcut=0.7,
        bw_highcut=3.7,
        fs=700,
    ):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.bw_order = bw_order
        self.bw_lowcut = bw_lowcut
        self.bw_highcut = bw_highcut
        self.fs = fs

    def process(self, use_neurokit=False, plot=False):
        # Process the entire 'ecg' column as a single sequence
        ecg_signal = self.df["ecg"].values

        if plot:
            self.plot_signal(self.df["ecg"].values, title="Raw ECG Signal")

        if use_neurokit:
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.fs)
            self.df["ecg"] = ecg_cleaned
            if plot:
                self.plot_signal(ecg_cleaned, title="NeuroKit Cleaned ECG Signal")
            return self.df

        smoothed_signal = self.smooth_ecg(ecg_signal)
        if plot:
            self.plot_signal(smoothed_signal, title="Smoothed ECG Signal")

        filtered_signal = self.butter_bandpass_filter(smoothed_signal)
        if plot:
            self.plot_signal(filtered_signal, title="Filtered ECG Signal")

        self.df["ecg"] = filtered_signal
        if plot:
            self.plot_signal(self.df["ecg"].values, title="Final Processed ECG Signal")

        return self.df

    def smooth_ecg(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype="band")

    def butter_bandpass_filter(self, data):
        b, a = self.butter_bandpass(
            self.bw_lowcut, self.bw_highcut, self.fs, order=self.bw_order
        )
        y = lfilter(b, a, data)
        return y

    def plot_signal(self, signal, title="Signal", num_points=1000):
        plt.figure(figsize=(10, 4))
        plt.plot(signal[:num_points])
        plt.title(title)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
