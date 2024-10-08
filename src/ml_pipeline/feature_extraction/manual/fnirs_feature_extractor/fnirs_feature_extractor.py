import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fft import fft
from scipy.integrate import simpson
from scipy.signal import find_peaks
class FNIRSDerivedHR:
    def __init__(self, df, sampling_rate=10):
        self.df = df
        self.sampling_rate = sampling_rate

    def detect_hr(self):
        # FDHR signal features
        O2Hb_filtered = self.df['O2Hb_HR'].dropna().values
        HHb_filtered = self.df['HHb_HR'].dropna().values

        peaks1 = self._detect_peaks(O2Hb_filtered)
        peaks2 = self._detect_peaks(HHb_filtered)
        # Add other algorithms here (e.g., AMPD, S functions)

        peaks_combined = self._weighted_mean_algorithm([peaks1, peaks2], len(O2Hb_filtered))  # Combine the peaks detected

        corrected_peaks = self._correct_peaks(peaks_combined, self.sampling_rate)
        HR = 60 / np.mean(np.diff(corrected_peaks) / self.sampling_rate)
        # print(f"Estimated Heart Rate: {HR:.2f} bpm")

        features = self._fdhr_features(corrected_peaks)

        return pd.DataFrame([features])

    def _correct_peaks(self, peaks, sampling_rate):
        IBIs = np.diff(peaks) / sampling_rate
        corrected_peaks = [peaks[0]]  # Start with the first peak

        for i in range(1, len(peaks)):
            m = np.mean(IBIs[max(0, i-int(4*sampling_rate)):i])
            sd = np.std(IBIs[max(0, i-int(4*sampling_rate)):i])
            SD = np.std(IBIs[:i])
            k = 0.8 * 15 / sd if sd > 0 else 0

            if m - k * SD <= IBIs[i-1] <= m + k * SD:
                corrected_peaks.append(peaks[i])
            else:
                # Search for another peak within the 0.25-second window before the current peak
                for j in range(i-1, max(-1, i-int(0.25*sampling_rate)), -1):
                    if peaks[j] not in corrected_peaks:
                        corrected_peaks[-1] = peaks[j]  # Replace previous peak with the closer one
                        break
                else:
                    corrected_peaks.append(peaks[i])

        return corrected_peaks

    def _detect_peaks(self, signal):
        first_diff = np.diff(signal)
        max_first_diff = np.max(np.abs(first_diff))
        peaks, _ = find_peaks(signal, distance=self.sampling_rate/2)

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
        IBIs = np.diff(peaks) / self.sampling_rate
        mean_IBI = np.mean(IBIs)
        std_IBI = np.std(IBIs)
        return {
            'mean_IBI': mean_IBI,
            'std_IBI': std_IBI
        }

class FNIRSFeatureExtractor:
    def __init__(self, df, sampling_rate=10):
        self.df = df
        self.sampling_rate = sampling_rate

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
        derived_hr = FNIRSDerivedHR(self.df, self.sampling_rate)
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks

# Assuming FNIRSDerivedHR class is already imported

def simulate_fnirs_data(duration=60, sampling_rate=10, hr=60, noise_level=0.05):
    """
    Simulate synthetic fNIRS data with a given heart rate.
    
    Parameters:
        duration (int): Duration of the signal in seconds.
        sampling_rate (int): Sampling frequency in Hz.
        hr (int): Heart rate in beats per minute.
        noise_level (float): Level of Gaussian noise to add to the signal.
    
    Returns:
        pd.DataFrame: DataFrame containing the simulated O2Hb and HHb signals.
    """
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    heart_rate_freq = hr / 60.0  # Convert bpm to Hz
    signal_O2Hb = np.sin(2 * np.pi * heart_rate_freq * t)  # Simulate sine wave for O2Hb
    signal_HHb = np.sin(2 * np.pi * heart_rate_freq * t + np.pi / 4)  # Phase-shifted sine wave for HHb

    # Add Gaussian noise
    signal_O2Hb += noise_level * np.random.randn(len(signal_O2Hb))
    signal_HHb += noise_level * np.random.randn(len(signal_HHb))

    # Combine into a DataFrame
    df = pd.DataFrame({'O2Hb_HR': signal_O2Hb, 'HHb_HR': signal_HHb})

    return df

def test_fnirs_derived_hr():
    # Simulation parameters
    duration = 60  # in seconds
    sampling_rate = 10  # sampling frequency in Hz
    hr = 75  # simulated heart rate in bpm
    noise_level = 0.3  # noise level

    # Simulate fNIRS data
    simulated_data = simulate_fnirs_data(duration=duration, sampling_rate=sampling_rate, hr=hr, noise_level=noise_level)

    # Initialize the FNIRSDerivedHR class
    hr_detector = FNIRSDerivedHR(simulated_data, sampling_rate)

    # Run heart rate detection
    result_df = hr_detector.detect_hr()

    # Print the results
    print("Detected Features:")
    print(result_df)

    # Plot the simulated signals and detected peaks for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(simulated_data['O2Hb_HR'], label='O2Hb_HR', color='blue', alpha=0.6)
    plt.plot(simulated_data['HHb_HR'], label='HHb_HR', color='red', alpha=0.6)

    # Detect peaks for plotting
    peaks1 = hr_detector._detect_peaks(simulated_data['O2Hb_HR'].values)
    peaks2 = hr_detector._detect_peaks(simulated_data['HHb_HR'].values)

    # Plot detected peaks
    plt.plot(peaks1, simulated_data['O2Hb_HR'].iloc[peaks1], "x", label='Detected Peaks O2Hb', color='blue')
    plt.plot(peaks2, simulated_data['HHb_HR'].iloc[peaks2], "x", label='Detected Peaks HHb', color='red')

    plt.xlabel("Time (samples)")
    plt.ylabel("Signal Amplitude")
    # plt.title("Simulated fNIRS Signals with Detected Peaks")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_fnirs_derived_hr()
