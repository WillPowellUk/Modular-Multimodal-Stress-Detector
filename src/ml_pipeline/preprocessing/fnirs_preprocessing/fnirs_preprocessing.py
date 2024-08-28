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
        # preprocess for FNIRS and FNIRS derived HR signal
        for col in self.df.columns:
            if col == 'O2Hb' or col == 'HHb' or col == 'Brain oxy':
                # Taken from paper  
                self.df[col] = self.butterworth_bandpass(self.df[col], 0.0012, 0.8, self.fs, order=5)
                # Taken from paper Hakimi et al.
                self.df[f'{col}_HR'] = self.butterworth_bandpass(self.df[col], lowcut=1.0, highcut=1.9, fs=self.fs)

        return self.df
