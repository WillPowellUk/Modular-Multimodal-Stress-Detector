import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD

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
                # Taken from klein et al. 2019
                # self.df[col] = self.preprocess_fnirs(self.df[col], self.fs)
                # Taken from paper Hakimi et al.
                self.df[f'{col}_HR'] = self.butterworth_bandpass(self.df[col], lowcut=1.0, highcut=1.9, fs=self.fs)

        return self.df

    def bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        """
        Apply a bandpass filter to the input data.

        Parameters:
        - data: ndarray, the input fNIRS data (channels x time points)
        - lowcut: float, the lower cutoff frequency of the filter
        - highcut: float, the upper cutoff frequency of the filter
        - fs: float, the sampling rate of the data
        - order: int, the order of the Butterworth filter

        Returns:
        - filtered_data: ndarray, the bandpass-filtered data
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')

        # Apply filter to each channel
        filtered_data = filtfilt(b, a, data, axis=-1)
        
        return filtered_data

    def global_component_removal(self, data, sigma=48):
        """
        Remove global component from fNIRS data using SVD and Gaussian smoothing.

        Parameters:
        - data: ndarray, the input fNIRS data (channels x time points)
        - sigma: float, standard deviation for the Gaussian kernel

        Returns:
        - clean_data: ndarray, the data after global component removal
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Perform SVD on the data
        svd = TruncatedSVD(n_components=min(data.shape))
        U = svd.fit_transform(data)
        S = svd.singular_values_
        VT = svd.components_

        # Gaussian kernel smoothing
        channel_count = data.shape[0]
        distances = np.zeros((channel_count, channel_count))

        # Calculate pairwise distances between channels (correctly!)
        for i in range(channel_count):
            for j in range(i + 1, channel_count):
                distances[i, j] = distances[j, i] = np.linalg.norm(i - j)

        # Correct shape for G
        G = np.exp(-distances**2 / (2 * sigma**2))

        # VT is of shape (n_components, features), we need to transpose it for multiplication
        VT_smooth = G @ VT.T  # Ensure dimensions match (channels, channels) @ (features, n_components)
        
        # Transpose back to original VT shape after smoothing
        VT_smooth = VT_smooth.T

        # Reconstruct global signal
        global_signal = (U @ np.diag(S)) @ VT_smooth

        # Subtract global component to get clean data
        clean_data = data - global_signal

        return clean_data



    def preprocess_fnirs(self, data, fs, lowcut=0.1, highcut=2.0, sigma=48, order=3):
        """
        Preprocess fNIRS data by applying bandpass filter and global component removal.

        Parameters:
        - data: ndarray, the input fNIRS data (channels x time points)
        - fs: float, the sampling rate of the data
        - lowcut: float, the lower cutoff frequency for bandpass filter
        - highcut: float, the upper cutoff frequency for bandpass filter
        - sigma: float, standard deviation for the Gaussian kernel in GCR
        - order: int, the order of the Butterworth filter

        Returns:
        - preprocessed_data: ndarray, the preprocessed fNIRS data
        """
        # Apply bandpass filter
        filtered_data = self.bandpass_filter(data, lowcut, highcut, fs, order)
        
        # Apply global component removal
        preprocessed_data = self.global_component_removal(filtered_data, sigma)
        
        return preprocessed_data
