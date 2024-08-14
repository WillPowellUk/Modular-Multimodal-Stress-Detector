import pyxdf
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


SAMPLING_FREQUENCY = 10  # Hz
XDF_FILE = r'data_collection\recordings\S69\fNIRS_noisy.xdf'

class fNIRSVisualizer:
    def __init__(self, sampling_frequency, save_path=None):
        self.sampling_frequency = sampling_frequency
        self.save_path = save_path
    
    def plot_segments(self, segments):
        fig, ax = plt.subplots()

        max_length = 0  # To store the maximum length of any segment for consistent time axis

        for i, segment in enumerate(segments):
            t = np.arange(len(segment)) / self.sampling_frequency
            max_length = max(max_length, len(t))  # Update max length if this segment is longer
            
            # Plot each segment
            ax.plot(t, segment, label=f'Segment {i+1}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        # Hide y-axis units
        ax.set_yticklabels([])

        # Crop the plot to the first 25% of the x-axis
        # ax.set_xlim(0, max_length / self.sampling_frequency * 0.25)

        ax.legend()

        # Add PSD plots for each segment
        for i, segment in enumerate(segments):
            fft = np.fft.fft(segment)
            freq = np.fft.fftfreq(len(segment), d=1/self.sampling_frequency)

            # max frequency to plot
            max_freq = self.sampling_frequency / 2

            # Filter out negative frequencies and frequencies above max_freq
            mask = (freq >= 0) & (freq <= max_freq)
            freq = freq[mask]
            fft = fft[mask]

            # Calculate PSD
            psd = ((np.abs(fft) ** 2) / len(segment))
            psd = 10 * np.log10(psd)
            psd -= psd.max()

            # Add subplot for this segment's PSD
            subax = fig.add_axes([0.68, 0.65 - i*0.25, 0.2, 0.2])
            subax.plot(freq, psd)

            # Limit x-axis to positive frequencies between 0 and max_freq
            subax.set_xlim(0, max_freq)

            # Add labels
            subax.set_xlabel('Frequency (Hz)')
            subax.set_ylabel('PSD (dB)')

        if self.save_path:
            plt.savefig(self.save_path)
        else:
            plt.show()

# Load the XDF file
data, header = pyxdf.load_xdf(XDF_FILE)

time_series_headers = ['02Hb', 'HHb', 'Brain oxy', 'Brain state']

# Add time series to dataframe
fnirs_df = pd.DataFrame(data[0]['time_series'], columns=time_series_headers)

# Add timestamps
fnirs_df.insert(0, 'Timestamp', data[0]['time_stamps'])

# Calculate time differences between consecutive timestamps
time_diffs = np.diff(fnirs_df['Timestamp'])

# Calculate average sampling frequency (Hz)
avg_sampling_frequency = 1 / np.mean(time_diffs)
print(f"Average Sampling Frequency: {avg_sampling_frequency:.2f} Hz")

# Interpolate the signals to the average sampling frequency
# Create a new time axis with uniform spacing based on the average frequency
new_timestamps = np.arange(fnirs_df['Timestamp'].iloc[0], fnirs_df['Timestamp'].iloc[-1], 1/SAMPLING_FREQUENCY)

# Create an empty dataframe for the interpolated data
interpolated_df = pd.DataFrame({'Timestamp': new_timestamps})

# Interpolate each column except the timestamp
for column in fnirs_df.columns[1:]:
    interpolation_function = interp1d(fnirs_df['Timestamp'], fnirs_df[column], kind='linear', fill_value='extrapolate')
    interpolated_df[column] = interpolation_function(new_timestamps)

# Remove the timestamp column as requested
interpolated_df = interpolated_df.drop(columns=['Timestamp'])

# Show the head of the interpolated dataframe
print(interpolated_df.head())

visualizer = fNIRSVisualizer(sampling_frequency=SAMPLING_FREQUENCY)
visualizer.plot_segments([interpolated_df['02Hb'].values, interpolated_df['HHb'].values])
