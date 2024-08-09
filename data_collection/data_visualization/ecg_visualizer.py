import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the desired directory
target_dir = current_dir
while "src" not in os.listdir(target_dir) and target_dir != os.path.dirname(
    target_dir
):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "src" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError("Could not find 'src' directory in the path hierarchy")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the desired directory
target_dir = current_dir
while "experiments" not in os.listdir(target_dir) and target_dir != os.path.dirname(
    target_dir
):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "experiments" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError(
        "Could not find 'experiments' directory in the path hierarchy"
    )


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from src.ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from src.ml_pipeline.feature_extraction.manual.ecg_feature_extractor import ECGFeatureExtractor

class ECGVisualizer:
    def __init__(self, sampling_frequency, save_path=None):
        self.sampling_frequency = sampling_frequency
        self.save_path = save_path
    
    def plot_segment(self, segment, ECG_processed=None, peaks=None, colors=['r', 'g', 'c', 'm', 'y', 'k']):
            # Define time array
            t = np.arange(len(segment)) / self.sampling_frequency

            # plot fft if no peaks are given
            if isinstance(peaks, type(None)):
                # Compute FFT
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

                # Plot raw ECG segment
                fig, ax = plt.subplots()
                ax.plot(t, segment)

                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')

                # Hide y-axis units
                ax.set_yticklabels([])

                # Crop the plot to the first 25% of the x-axis
                ax.set_xlim(t.max()*0, t.max()*0.25)

                # Add subplot
                subax = fig.add_axes([0.68, 0.65, 0.2, 0.2])
                subax.plot(freq, psd)

                # Limit x-axis to positive frequencies between 0 and max_freq
                subax.set_xlim(0, max_freq)

                # add labels
                subax.set_xlabel('Frequency (Hz)')
                subax.set_ylabel('PSD (dB)')
                if self.save_path:
                    plt.savefig(self.save_path)
                else:
                    plt.show()

            # otherwise, plot peaks
            else:
                # Plot raw ECG segment
                plt.figure()
                plt.plot(t, segment)

                # Create Line2D objects for each peak type with corresponding color
                lines = [Line2D([0], [0], linestyle='--', color=colors[i]) for i in range(len(peaks))]

                # Plot peaks
                for i, peak in enumerate(peaks):
                    peak_inds = np.where(ECG_processed[peak] == 1)[0]
                    for ind in peak_inds:
                        plt.axvline(x=t[ind], linestyle='--', color=colors[i])

                # Add legend with the created Line2D objects and corresponding labels
                plt.legend(handles=lines, labels=peaks, loc='lower right')

                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                # Hide y-axis units
                plt.gca().set_yticklabels([])
                # Crop the plot to the first 25% of the x-axis
                plt.xlim(t.max()*0, t.max()*0.1)

                if self.save_path:
                    plt.savefig(self.save_path)
                else:
                    plt.show()

if __name__ == '__main__':
    # Define the subject ID
    subject_id = 3
    sampling_frequency = 130

    save_path = f'data_collection/data_visualization/plots/S{subject_id}_ECG.pdf'
    save_path = None

    # Create an ECGVisualizer object
    ecg_visualizer = ECGVisualizer(sampling_frequency=sampling_frequency, save_path=save_path)

    # Load the ECG recording
    segment = np.loadtxt(f'data_collection/recordings/S{subject_id}/ECG.csv', delimiter=',')

    # Define the start and end times for the segment you want to crop (in seconds)
    start_time = 10  # e.g., start at 5 seconds
    end_time = 50   # e.g., end at 10 seconds

    # Convert time to sample indices
    start_index = int(start_time * sampling_frequency)
    end_index = int(end_time * sampling_frequency)

    # Crop the segment
    cropped_segment = segment[start_index:end_index]

    # # Plot the segment
    # ecg_visualizer.plot_segment(cropped_segment)

    # load cropped segment as dataframe
    df = pd.DataFrame(cropped_segment, columns=['ecg'])

    ecg_processor = ECGPreprocessing(df, fs=sampling_frequency)
    df = ecg_processor.process(use_neurokit=True, plot=False)

    # Plot the cleaned ECG segment
    ecg_visualizer.plot_segment(df['ecg'].values)

    # Extract peaks
    ecg_feature_extractor = ECGFeatureExtractor(df, sampling_rate=sampling_frequency)
    ecg_feature_extractor.extract_features(show_plot=True)