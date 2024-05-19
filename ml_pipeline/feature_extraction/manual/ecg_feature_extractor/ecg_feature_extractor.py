import neurokit2 as nk
import pandas as pd

class ECGFeatureExtractor:
    def __init__(self, ecg_data: pd.DataFrame, sampling_rate: int = 1000):
        self.ecg_data = ecg_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        # Use the preprocessed ECG signal directly
        ecg_signal = self.ecg_data['ECG'].values
        
        # Detect R-peaks
        r_peaks, _ = nk.ecg_peaks(ecg_signal, sampling_rate=self.sampling_rate)
        
        # Convert R-peaks to a DataFrame
        r_peaks_df = pd.DataFrame({"ECG_R_Peaks": r_peaks})
        
        # Extract time domain features
        time_domain = nk.hrv_time(r_peaks_df, sampling_rate=self.sampling_rate)
        
        # Extract frequency domain features
        frequency_domain = nk.hrv_frequency(r_peaks_df, sampling_rate=self.sampling_rate)

        # Extract nonlinear features
        nonlinear_features = nk.hrv_nonlinear(r_peaks_df, sampling_rate=self.sampling_rate)

        # Combine all features into a single DataFrame
        all_features = pd.concat([time_domain, frequency_domain, nonlinear_features], axis=1)
        
        return all_features