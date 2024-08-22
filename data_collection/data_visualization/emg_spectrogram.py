import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram

# Load the EMG data
subject_id = 1
emg_df = pd.read_csv(rf'data_collection\recordings\S{subject_id}\quattrocento\EMG.csv', delimiter=';')
emg_df.columns = ['Time', 'Upper Trapezius', 'Mastoid']

# Sampling rate
sample_rate = 2048  # in Hz

# Bandpass filter design
lowcut = 0.5
highcut = 20.0
nyquist = 0.5 * sample_rate
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(N=4, Wn=[low, high], btype='band')

# # Apply the bandpass filter
# emg_df['Upper Trapezius'] = filtfilt(b, a, emg_df['Upper Trapezius'])
# emg_df['Mastoid'] = filtfilt(b, a, emg_df['Mastoid'])

# Compute the spectrogram for each signal
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

frequencies, times, Sxx = spectrogram(emg_df['Upper Trapezius'], fs=sample_rate, nperseg=256)
axs[0].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
# axs[0].set_ylim(0, 20)  # Set the frequency limits to 0-20 Hz
axs[0].set_ylabel('Frequency [Hz]')
axs[0].set_xlabel('Time [sec]')
axs[0].set_title('Spectrogram - Upper Trapezius')

frequencies, times, Sxx = spectrogram(emg_df['Mastoid'], fs=sample_rate, nperseg=256)
axs[1].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
# axs[1].set_ylim(0, 20)  # Set the frequency limits to 0-20 Hz
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_xlabel('Time [sec]')
axs[1].set_title('Spectrogram - Mastoid')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
