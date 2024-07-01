import pandas as pd
import pywt
from scipy.signal import butter, sosfilt, savgol_filter
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np


class RespPreprocessing:
    def __init__(
        self,
        df,
        sg_window_size=11,
        sg_poly_order=3,
        bw_order=3,
        bw_lowcut=0.1,
        bw_highcut=0.35,
        fs=700,
    ):
        self.df = df
        self.sg_window_size = sg_window_size
        self.sg_poly_order = sg_poly_order
        self.bw_order = bw_order
        self.bw_lowcut = bw_lowcut
        self.bw_highcut = bw_highcut
        self.fs = fs

    def process(self):
        resp_signal = self.df["resp"].values
        # self.plot_frequency_response(resp_signal, 'Resp Frequency Response')
        smoothed_signal = self.smooth_resp(resp_signal)
        # self.plot_frequency_response(smoothed_signal, 'Resp Frequency Smoothed Response')
        filtered_signal = self.butterworth_filter(smoothed_signal)
        # self.plot_frequency_response(filtered_signal, 'Resp Frequency Filtered Response')
        self.df["resp"] = filtered_signal
        return self.df

    def smooth_resp(self, signal):
        return savgol_filter(signal, self.sg_window_size, self.sg_poly_order)

    def butterworth_filter(self, signal):
        nyquist = 0.5 * self.fs
        low = self.bw_lowcut / nyquist
        high = self.bw_highcut / nyquist
        sos = butter(self.bw_order, [low, high], btype="bandpass", output="sos")
        return sosfilt(sos, signal)

    def plot_frequency_response(self, signal, title):
        # Compute the FFT of the signal
        N = len(signal)
        T = 1.0 / self.fs
        yf = fft(signal)
        xf = fftfreq(N, T)[: N // 2]

        # Plot the FFT
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0 / N * np.abs(yf[: N // 2]), label="FFT of signal")
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")

        # Plot the cutoff frequencies
        plt.axvline(
            x=self.bw_lowcut,
            color="r",
            linestyle="--",
            label=f"Lowcut: {self.bw_lowcut} Hz",
        )
        plt.axvline(
            x=self.bw_highcut,
            color="g",
            linestyle="--",
            label=f"Highcut: {self.bw_highcut} Hz",
        )

        # Limit x-axis between 0 and 1 Hz
        plt.xlim(0, 1)

        plt.legend()
        plt.grid()
        plt.show()
