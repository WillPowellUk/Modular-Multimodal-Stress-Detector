import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fft import fft
from scipy.integrate import simpson
from scipy.signal import find_peaks
class FNIRSDerivedHR:
    def __init__(self, df, fs):
        self.df = df
        self.fs = fs

    def detect_hr(self):
        # FDHR signal features
        O2Hb_filtered = self.df['O2Hb_HR'].dropna().values
        HHb_filtered = self.df['HHb_HR'].dropna().values

        peaks1 = self._detect_peaks(O2Hb_filtered)
        peaks2 = self._detect_peaks(HHb_filtered)
        # Add other algorithms here (e.g., AMPD, S functions)

        peaks_combined = self._weighted_mean_algorithm([peaks1, peaks2], len(O2Hb_filtered))  # Combine the peaks detected

        corrected_peaks = self._correct_peaks(peaks_combined, self.fs)
        HR = 60 / np.mean(np.diff(corrected_peaks) / self.fs)
        print(f"Estimated Heart Rate: {HR:.2f} bpm")

        features = self._fdhr_features(corrected_peaks)

        return pd.DataFrame([features])

    def _correct_peaks(self, peaks, fs):
        IBIs = np.diff(peaks) / fs
        corrected_peaks = [peaks[0]]  # Start with the first peak

        for i in range(1, len(peaks)):
            m = np.mean(IBIs[max(0, i-int(4*fs)):i])
            sd = np.std(IBIs[max(0, i-int(4*fs)):i])
            SD = np.std(IBIs[:i])
            k = 0.8 * 15 / sd if sd > 0 else 0

            if m - k * SD < IBIs[i-1] < m + k * SD:
                corrected_peaks.append(peaks[i])
            else:
                # Search for another peak within the 0.25-second window before the current peak
                for j in range(i-1, max(-1, i-int(0.25*fs)), -1):
                    if peaks[j] not in corrected_peaks:
                        corrected_peaks[-1] = peaks[j]  # Replace previous peak with the closer one
                        break
                else:
                    corrected_peaks.append(peaks[i])

        return corrected_peaks

    def _detect_peaks(self, signal):
        first_diff = np.diff(signal)
        max_first_diff = np.max(np.abs(first_diff))
        peaks, _ = find_peaks(signal, distance=self.fs/2)

        valid_peaks = []
        for peak in peaks:
            if (signal[peak] > np.percentile(signal, 95)) and (np.abs(first_diff[peak-1]) > 0.1 * max_first_diff):
                valid_peaks.append(peak)
        
        return valid_peaks

    def _weighted_mean_algorithm(self, peaks_list, signal_length):
        # Assign weights based on the performance of each algorithm
        weights = [5.04, 4.39, 3.78, 3.24]  # Weights from the paper
        weighted_peaks = np.zeros(signal_length)

        for i, peaks in enumerate(peaks_list):
            for peak in peaks:
                weighted_peaks[peak] += weights[i]

        final_peaks = np.where(weighted_peaks > np.percentile(weighted_peaks, 50))[0]  # Threshold to select final peaks
        return final_peaks

    def _fdhr_features(self, peaks):
        # Example feature calculation based on detected peaks
        IBIs = np.diff(peaks) / self.fs
        mean_IBI = np.mean(IBIs)
        std_IBI = np.std(IBIs)
        return {
            'mean_IBI': mean_IBI,
            'std_IBI': std_IBI
        }

class FNIRSFeatureExtractor:
    def __init__(self, df, fs=10):
        self.df = df
        self.fs = fs

    def extract_features(self):
        features = {}
        
        for col in self.df.columns:
            # Time-domain features
            signal = self.df[col].dropna()
            features[f'{col}_mean'] = np.mean(signal)
            features[f'{col}_std'] = np.std(signal)
            features[f'{col}_skewness'] = skew(signal)
            features[f'{col}_kurtosis'] = kurtosis(signal)

            # Frequency-domain features
            freq_features = self._frequency_features(signal.values)
            features.update({f'{col}_freq_{key}': val for key, val in freq_features.items()})

            # Low-to-high frequency power ratio
            lf_hf_ratio = self._lf_hf_ratio(signal)
            features[f'{col}_lf_hf_ratio'] = lf_hf_ratio
            
            # Additional signal stats
            features[f'{col}_min'] = np.min(signal)
            features[f'{col}_max'] = np.max(signal)
            features[f'{col}_mean'] = np.mean(signal)

        # FDHR signal features
        derived_hr = FNIRSDerivedHR(self.df, self.fs)
        hr_features = derived_hr.detect_hr()

        features.update(hr_features.to_dict(orient='records')[0])

        return pd.DataFrame([features])

    def _frequency_features(self, signal):
        n = len(signal)
        freq = np.fft.fftfreq(n)
        fft_vals = np.abs(fft(signal))

        mean_freq = np.mean(freq)
        std_freq = np.std(freq)
        skewness_freq = skew(fft_vals)
        kurtosis_freq = kurtosis(fft_vals)

        return {
            'mean_freq': mean_freq,
            'std_freq': std_freq,
            'skewness_freq': skewness_freq,
            'kurtosis_freq': kurtosis_freq
        }

    def _lf_hf_ratio(self, signal, low_freq=(0.04, 0.15), high_freq=(0.15, 0.4)):
        freqs, psd = welch(signal)
        
        # Low Frequency Power
        low_mask = (freqs >= low_freq[0]) & (freqs <= low_freq[1])
        lf_power = simpson(psd[low_mask], freqs[low_mask])
        
        # High Frequency Power
        high_mask = (freqs >= high_freq[0]) & (freqs <= high_freq[1])
        hf_power = simpson(psd[high_mask], freqs[high_mask])
        
        return lf_power / hf_power if hf_power > 0 else np.nan

