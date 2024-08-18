import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fft import fft
from scipy.integrate import simpson

class FNIRSFeatureExtractor:
    
    def __init__(self, df):
        self.df = df

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
            freq_features = self._frequency_features(signal)
            features.update({f'{col}_freq_{key}': val for key, val in freq_features.items()})

            # Low-to-high frequency power ratio
            lf_hf_ratio = self._lf_hf_ratio(signal)
            features[f'{col}_lf_hf_ratio'] = lf_hf_ratio
            
            # Additional signal stats
            features[f'{col}_min'] = np.min(signal)
            features[f'{col}_max'] = np.max(signal)
            features[f'{col}_mean'] = np.mean(signal)

        # FDHR signal features
        fdhr_signal = self.df['02Hb (µM)'] - self.df['HHb (µM)']
        features.update(self._fdhr_features(fdhr_signal))

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
        lf_power = simps(psd[low_mask], freqs[low_mask])
        
        # High Frequency Power
        high_mask = (freqs >= high_freq[0]) & (freqs <= high_freq[1])
        hf_power = simps(psd[high_mask], freqs[high_mask])
        
        return lf_power / hf_power if hf_power > 0 else np.nan

    def _fdhr_features(self, fdhr_signal):
        features = {}
        features['fdhr_mean'] = np.mean(fdhr_signal)
        features['fdhr_std'] = np.std(fdhr_signal)
        features['fdhr_skewness'] = skew(fdhr_signal)
        features['fdhr_kurtosis'] = kurtosis(fdhr_signal)
        features['fdhr_rms'] = np.sqrt(np.mean(fdhr_signal**2))

        freq_features = self._frequency_features(fdhr_signal)
        features.update({f'fdhr_freq_{key}': val for key, val in freq_features.items()})

        lf_hf_ratio = self._lf_hf_ratio(fdhr_signal)
        features['fdhr_lf_hf_ratio'] = lf_hf_ratio

        return features