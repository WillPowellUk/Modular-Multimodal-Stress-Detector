import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

class FNIRSPreprocessing:
    def __init__(self, df, fs=10):
        self.df = df
        self.fs = fs

    def interpolate(self, signal_values, timestamps):
        # Interpolate the signals to the uniform sampling frequency
        new_timestamps = np.arange(timestamps.iloc[0], timestamps.iloc[-1], 1/self.fs)
        interpolation_function = interp1d(timestamps, signal_values, kind='linear', fill_value='extrapolate')
        interpolated_signal = interpolation_function(new_timestamps)
        return new_timestamps, interpolated_signal

    def butterworth_bandpass(self, data, lowcut, highcut, fs, order=4):       
        # High-pass filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design high-pass Butterworth filter
        b_high, a_high = butter(order, low, btype='high', analog=False)
        # Apply zero-phase high-pass filter
        high_passed_data = filtfilt(b_high, a_high, data)
        
        # Design low-pass Butterworth filter
        b_low, a_low = butter(order, high, btype='low', analog=False)
        # Apply zero-phase low-pass filter
        filtered_data = filtfilt(b_low, a_low, high_passed_data)
        
        return filtered_data

    def process(self):
        timestamps = self.df.index if 'Timestamp' not in self.df.columns else self.df['Timestamp']
        interpolated_df = pd.DataFrame()

        # preprocess for FNIRS and FNIRS derived HR features
        for col in self.df.columns:
            if col == 'O2Hb' or col == 'HHb' or col == 'Brain oxy':
                signal_values = self.df[col].dropna()
                new_timestamps, interpolated_signal = self.interpolate(signal_values, timestamps)
                # fnirs_filtered = self.butterworth_bandpass(interpolated_signal, 0.04, 4.9, self.fs)
                # interpolated_df[col] = fnirs_filtered
                hr_filtered = self.butterworth_bandpass(interpolated_signal, lowcut=1.0, highcut=1.9, fs=self.fs)
                interpolated_df[f'{col}_HR'] = hr_filtered
                interpolated_df[col] = self.df[col][:len(interpolated_signal)]

        interpolated_df['Timestamp'] = new_timestamps  # Add the new interpolated timestamps
        self.df = interpolated_df.drop(columns=['Timestamp'])  # Optionally drop the timestamp after processing
        return self.df
