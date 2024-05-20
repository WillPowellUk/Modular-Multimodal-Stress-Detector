import neurokit2 as nk
import pandas as pd

class EMGFeatureExtractor:
    def __init__(self, emg_data: pd.DataFrame, sampling_rate: int = 1000):
        self.emg_data = emg_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        features = {}

        # Use the preprocessed EMG signal directly
        emg_signal = self.emg_data['EMG'].values

        # Process the EMG signal
        signals, info = nk.emg_process(emg_signal, sampling_rate=self.sampling_rate)
        
        # Extract time domain features
        time_domain = nk.emg_amplitude(signals, sampling_rate=self.sampling_rate)

        # Extract frequency domain features
        frequency_domain = nk.emg_power_spectrum(emg_signal, sampling_rate=self.sampling_rate)

        # Combine all features into a single DataFrame
        all_features = pd.concat([time_domain, frequency_domain], axis=1)

        return all_features
