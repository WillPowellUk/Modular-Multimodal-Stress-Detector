'''
BVP signal is filtered by a Butterworth band-pass filter of order
3 with cutoff frequencies (f1 = 0.7 Hz and f2 = 3.7 Hz),
which takes into account the heart rate at rest (≈40 BPM)
or high heart rate due to exercise scenarios or tachycardia
(≈220 BPM) - Stress Detection Using Context-Aware Sensor
Fusion From Wearable Devices, 2023.
'''

import pandas as pd
from scipy.signal import butter, filtfilt

class BVPPreprocessing:
    def __init__(self, df, order=3, lowcut=0.7, highcut=3.7, fs=64):
        self.df = df
        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs

    def process(self):
        # Apply the BVP filter to each signal
        self.df['bvp'] = self.df['bvp'].apply(self.bvp_filter)
        return self.df

    def bvp_filter(self, signal):
        # Calculate the Nyquist frequency
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        # Create the Butterworth band-pass filter
        b, a = butter(self.order, [low, high], btype='band')
        # Apply the filter to the signal
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
