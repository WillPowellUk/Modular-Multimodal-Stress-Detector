import pandas as pd
import numpy as np
from scipy.integrate import simps
from scipy.fft import rfft, rfftfreq

"""  """


class AccFeatureExtractor:
    def __init__(self, acc_data: pd.DataFrame, sampling_rate: int = 100):
        self.acc_data = acc_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        features = {}

        # Mean and STD of each axis and summed over all axes
        for axis in ["x", "y", "z"]:
            features[f"mean_acc_{axis}"] = np.mean(self.acc_data[axis])
            features[f"std_acc_{axis}"] = np.std(self.acc_data[axis])
        features["mean_acc_all"] = np.mean(self.acc_data[["x", "y", "z"]].values)
        features["std_acc_all"] = np.std(self.acc_data[["x", "y", "z"]].values)

        # Absolute integral for each/all axes
        for axis in ["x", "y", "z"]:
            features[f"abs_integral_acc_{axis}"] = simps(
                np.abs(self.acc_data[axis]), dx=1 / self.sampling_rate
            )
        features["abs_integral_acc_all"] = simps(
            np.abs(self.acc_data[["x", "y", "z"]].values).sum(axis=1),
            dx=1 / self.sampling_rate,
        )

        # Peak frequency of each axis
        for axis in ["x", "y", "z"]:
            freq, power = self._calculate_peak_frequency(self.acc_data[axis].values)
            features[f"peak_freq_acc_{axis}"] = freq

        return pd.DataFrame([features])

    def _calculate_peak_frequency(self, signal):
        N = len(signal)
        yf = rfft(signal)
        xf = rfftfreq(N, 1 / self.sampling_rate)
        idx_peak = np.argmax(np.abs(yf))
        peak_freq = xf[idx_peak]
        return peak_freq, np.abs(yf[idx_peak])
