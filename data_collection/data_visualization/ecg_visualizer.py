import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib.lines import Line2D

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
                max_freq = 100

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
    subject_id = 2

    # Create an ECGVisualizer object
    ecg_visualizer = ECGVisualizer(sampling_frequency=130, save_path=f'data_collection/data_visualization/plots/S{subject_id}_ECG.pdf')

    # Load the ECG recording
    segment = np.loadtxt(f'data_collection/recordings/S{subject_id}/ECG.csv', delimiter=',')

    # Plot the segment
    ecg_visualizer.plot_segment(segment)