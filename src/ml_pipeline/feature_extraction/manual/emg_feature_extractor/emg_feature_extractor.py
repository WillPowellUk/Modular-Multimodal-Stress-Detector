import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import iqr

class EMGFeatureExtractor:
    def __init__(self, emg_data: pd.DataFrame, sampling_rate: int = 1000):
        self.emg_data = emg_data.values.flatten()
        self.sampling_rate = sampling_rate

    def extract_features(self, freq_binning=False):
        features = {}

        # Mean and STD of EMG
        features["mean_EMG"] = np.mean(self.emg_data)
        features["std_EMG"] = np.std(self.emg_data)

        # Dynamic range of EMG
        features["range_EMG"] = np.ptp(self.emg_data)

        # Absolute integral
        features["abs_integral_EMG"] = np.sum(np.abs(self.emg_data))

        # Median of EMG
        features["median_EMG"] = np.median(self.emg_data)

        # 10th and 90th percentile
        features["P10_EMG"] = np.percentile(self.emg_data, 10)
        features["P90_EMG"] = np.percentile(self.emg_data, 90)

     # Mean, median, and peak frequency
        freqs, power_spectrum = welch(self.emg_data, fs=self.sampling_rate)

        # Check if power_spectrum has any complex values
        if np.iscomplexobj(power_spectrum):
            print("Warning: Power spectrum has complex values. Converting to magnitudes.")

        # Convert to magnitudes if any complex values are detected
        power_spectrum = np.abs(power_spectrum)

        # Compute features
        features["mean_freq_EMG"] = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        cumulative_power = np.cumsum(power_spectrum)
        median_freq_index = np.where(cumulative_power >= 0.5 * cumulative_power[-1])[0][0]
        features["median_freq_EMG"] = freqs[median_freq_index]
        features["peak_freq_EMG"] = freqs[np.argmax(power_spectrum)]


        # Energy in seven bands
        band_limits = [0, 4, 8, 12, 16, 20, 24, 28, np.inf]
        for i in range(len(band_limits) - 1):
            band_power = power_spectrum[
                (freqs >= band_limits[i]) & (freqs < band_limits[i + 1])
            ]
            features[f"energy_band_{band_limits[i]}_EMG"] = np.sum(band_power)

        # Number of peaks
        peaks, _ = find_peaks(self.emg_data)
        features["num_peaks_EMG"] = len(peaks)

        # Mean and STD of peak amplitude
        peak_amplitudes = self.emg_data[peaks]
        features["mean_peak_amp_EMG"] = np.mean(peak_amplitudes)
        features["std_peak_amp_EMG"] = np.std(peak_amplitudes)

        # Sum and norm. sum of peak amplitude
        features["sum_peak_amp_EMG"] = np.sum(peak_amplitudes)
        features["norm_sum_peak_amp_EMG"] = np.linalg.norm(peak_amplitudes)

        # Additional Features

        # Variance (VAR)
        features["variance_EMG"] = np.var(self.emg_data)

        # Simple Square Integral (SSI)
        features["ssi_EMG"] = np.sum(self.emg_data ** 2)

        # Asymmetry
        features["asymmetry_EMG"] = np.mean(self.emg_data ** 3) / (np.std(self.emg_data) ** 3)

        # Mean Power Spectral Density (PSD)
        features["mean_psd_EMG"] = np.mean(power_spectrum)

        # Median Frequency (MDF)
        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        median_freq_index = np.where(cumulative_power >= total_power / 2)[0][0]
        features["median_freq_EMG"] = freqs[median_freq_index]

        if freq_binning:
            # Frequency binning (e.g., 0-20 Hz, 20-40 Hz, etc.)
            bin_width = 10
            max_freq = freqs[-1]
            bin_edges = np.arange(0, max_freq + bin_width, bin_width)
            for i in range(len(bin_edges) - 1):
                bin_power = power_spectrum[
                    (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
                ]
                features[f"freq_bin_{bin_edges[i]}_{bin_edges[i + 1]}_power_EMG"] = np.sum(bin_power)

        return pd.DataFrame([features])
