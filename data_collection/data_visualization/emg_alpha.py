import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


import numpy as np
from scipy.signal import iirnotch, lfilter
import matplotlib.pyplot as plt

class EMGPreprocessing():
    def __init__(self) -> None:
        pass
    
    def notch_filter(self, signal, fs, f0, Q):
        b, a = iirnotch(f0, Q, fs)
        return lfilter(b, a, signal)

# Configure Matplotlib to use LaTeX for rendering
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",  # Use serif font in conjunction with LaTeX
#     # Set the default font to be used in LaTeX as a single string
#     "text.latex.preamble": r"\usepackage{times}",
# })


subject_id = 69
emg_df = pd.read_csv(rf'data_collection\recordings\S{subject_id}\quattrocento\alpha.csv', delimiter=';')
emg_df.columns = ['Time', 'Upper Trapezius', 'Occipital', 'Mastoid']

# Sampling rate
sample_rate = 2048  # in Hz

# Define the filter parameters
f0 = 50    # Frequency to be removed from the signal (notch) in Hz
Q = 30     # Quality factor

# Create an instance of the ECGPreprocessing class
emg_preprocessing = EMGPreprocessing()
emg_df['Upper Trapezius'] = emg_preprocessing.notch_filter(emg_df['Upper Trapezius'], sample_rate, f0, Q)
emg_df['Occipital'] = emg_preprocessing.notch_filter(emg_df['Occipital'], sample_rate, f0, Q)
emg_df['Mastoid'] = emg_preprocessing.notch_filter(emg_df['Mastoid'], sample_rate, f0, Q)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

# Calculate time vector
time = emg_df.index / sample_rate

# Function to calculate the FFT and PSD
def calculate_fft_psd(signal, sample_rate):
    freqs, psd = welch(signal, sample_rate, nperseg=int(sample_rate / 0.1))
    return freqs, 10 * np.log10(psd)

# Plotting the EMG signals with FFT subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Increase x-axis tick frequency
tick_freq = 1  # Set the tick frequency for the x-axis

# Upper Trapezius
axs[0].plot(time, emg_df['Upper Trapezius'], label='Upper Trapezius', color='black')
axs[0].set_xlabel('Time (s)', fontsize=14)
axs[0].set_xlim(30, 50)
current_xticks = axs[0].get_xticks()
new_xticks = np.arange(30,51,1)
axs[0].set_xticks(new_xticks)
axs[0].set_xticklabels(new_xticks - 30)
axs[0].set_ylim(-300, 950)
axs[0].set_xticks(np.arange(30, 51, tick_freq))  # Set tick frequency
axs[0].set_ylabel('Amplitude (µV)', fontsize=14)
axs[0].set_title('Upper Trapezius', fontsize=18)
axs[0].grid(True)

# FFT Subplot for Upper Trapezius
freqs, psd = calculate_fft_psd(emg_df['Upper Trapezius'], sample_rate)
peaks, _ = find_peaks(psd, height=np.mean(psd) + np.std(psd) * 0.5)
peak_freqs = freqs[peaks]
print(f"Upper Trapezius FFT Peaks (Hz): {[f'{freq:.2f}' for freq in peak_freqs]}")
inset_ax = axs[0].inset_axes([0.68, 0.65, 0.3, 0.3])  # [x, y, width, height]
inset_ax.plot(freqs, psd, color='black')
inset_ax.set_xlim(0, 15)  # Limit the x-axis to 50 Hz
inset_ax.set_ylim(0,25)
inset_ax.set_xlabel('Frequency (Hz)', fontsize=10)
inset_ax.set_ylabel('PSD (dB)', fontsize=10)
inset_ax.grid(True)
# for peak in peak_freqs:
#     inset_ax.axvline(peak, color='red', linestyle='--')  # Plot vertical line at peak frequency

# Occipital
axs[1].plot(time, emg_df['Occipital'], label='Occipital', color='black')
axs[1].set_xlabel('Time (s)', fontsize=14)
axs[1].set_xlim(30, 50)
current_xticks = axs[1].get_xticks()
new_xticks = np.arange(30,51,1)
axs[1].set_xticks(new_xticks)
axs[1].set_xticklabels(new_xticks - 30)
axs[1].set_ylim(-100, 300)
axs[1].set_xticks(np.arange(30, 51, tick_freq))  # Set tick frequency
axs[1].set_ylabel('Amplitude (µV)', fontsize=14)
axs[1].set_title('Occipital Lobe', fontsize=18)
axs[1].grid(True)

# FFT Subplot for Occipital
freqs, psd = calculate_fft_psd(emg_df['Occipital'], sample_rate)
peaks, _ = find_peaks(psd, height=np.mean(psd) + np.std(psd) * 0.5)
peak_freqs = freqs[peaks]
print(f"Occipital FFT Peaks (Hz): {[f'{freq:.2f}' for freq in peak_freqs]}")
inset_ax = axs[1].inset_axes([0.68, 0.65, 0.3, 0.3])  # [x, y, width, height]
inset_ax.plot(freqs, psd, color='black')
inset_ax.set_xlim(0, 15)  # Limit the x-axis to 50 Hz
inset_ax.set_ylim(0,25)
inset_ax.set_xlabel('Frequency (Hz)', fontsize=10)
inset_ax.set_ylabel('PSD (dB)', fontsize=10)
inset_ax.grid(True)
# for peak in peak_freqs:
#     inset_ax.axvline(peak, color='red', linestyle='--')  # Plot vertical line at peak frequency

# Mastoid
axs[2].plot(time, emg_df['Mastoid'], label='Mastoid', color='black')
axs[2].set_xlabel('Time (s)', fontsize=14)
axs[2].set_xlim(30, 50)
current_xticks = axs[2].get_xticks()
new_xticks = np.arange(30,51,1)
axs[2].set_xticks(new_xticks)
axs[2].set_xticklabels(new_xticks - 30)
axs[2].set_ylim(-100, 300)
axs[2].set_xticks(np.arange(30, 51, tick_freq))  # Set tick frequency
axs[2].set_ylabel('Amplitude (µV)', fontsize=14)
axs[2].set_title('Mastoid', fontsize=18)
axs[2].grid(True)

# FFT Subplot for Mastoid
freqs, psd = calculate_fft_psd(emg_df['Mastoid'], sample_rate)
peaks, _ = find_peaks(psd, height=np.mean(psd) + np.std(psd) * 0.5)
peak_freqs = freqs[peaks]
print(f"Mastoid FFT Peaks (Hz): {[f'{freq:.2f}' for freq in peak_freqs]}")
inset_ax = axs[2].inset_axes([0.68, 0.65, 0.3, 0.3])  # [x, y, width, height]
inset_ax.plot(freqs, psd, color='black')
inset_ax.set_xlim(0, 15)  # Limit the x-axis to 50 Hz
inset_ax.set_ylim(0,25)
inset_ax.set_xlabel('Frequency (Hz)', fontsize=10)
inset_ax.set_ylabel('PSD (dB)', fontsize=10)
inset_ax.grid(True)
# for peak in peak_freqs:
#     inset_ax.axvline(peak, color='red', linestyle='--')  # Plot vertical line at peak frequency
plt.subplots_adjust(hspace=0.5)  # Adjust the hspace value as needed
plt.show()
