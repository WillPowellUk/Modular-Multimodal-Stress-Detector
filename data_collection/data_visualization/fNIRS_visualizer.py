import pyxdf
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from src.ml_pipeline.preprocessing.fnirs_preprocessing import FNIRSPreprocessing
from src.ml_pipeline.feature_extraction.manual.fnirs_feature_extractor import FNIRSFeatureExtractor


# Configure Matplotlib to use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use serif font in conjunction with LaTeX
    # Set the default font to be used in LaTeX as a single string
    "text.latex.preamble": r"\usepackage{times}",
})


subject_id = 5
SAMPLING_FREQUENCY = 10  # Hz
XDF_FILE = f'data_collection/recordings/S{subject_id}/myndsens/FNIRS.xdf'

class fNIRSVisualizer:
    def __init__(self, sampling_frequency, save_path=None):
        self.sampling_frequency = sampling_frequency
        self.save_path = save_path

    def plot_plots(self, plots, y_labels, title_labels, legend_labels=None, plot_fft=False, markers=None):
        num_plots = len(plots)
        
        # Create vertical subplots with increased space between them
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 5), constrained_layout=True)
        
        # If there's only one plot, axes might not be an array, so we make it a list
        if num_plots == 1:
            axes = [axes]
        
        max_length = 0  # To store the maximum length of any plot for consistent time axis
        for i, plot in enumerate(plots):
            t = np.arange(len(plot)) / self.sampling_frequency
            max_length = max(max_length, len(t))  # Update max length if this plot is longer
            # Plot each plot in its own subplot
            axes[i].plot(t, plot)
            axes[i].set_xlabel('Time (s)', fontsize=16)
            axes[i].set_ylabel(y_labels[i],fontsize=16)
            axes[i].set_title(title_labels[i], fontsize=20)
            axes[i].grid(True)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            # Add vertical markers and shaded regions if provided
            if markers:
                for j in range(len(markers) - 1):
                    axes[i].axvline(x=markers[j], color='r', linestyle='--')
                    axes[i].axvline(x=markers[j+1], color='r', linestyle='--')
                    axes[i].fill_between(t, plot.min(), plot.max(), where=(t >= markers[j]) & (t < markers[j+1]), color=f'C{j}', alpha=0.1)
                            
        # Adjust the layout to make sure there's enough space between plots
        plt.tight_layout(h_pad=7)  # Increase the height padding between plots
        
        # Add space for the legend
        fig.subplots_adjust(bottom=0.16)
        
        # Create custom legend handles
        if markers and legend_labels:
            legend_handles = [plt.Rectangle((0,0),1,1, fc=f'C{i}', alpha=0.3) for i in range(len(legend_labels))]
            # Add the legend with the plot labels at the bottom
            fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=len(legend_labels), fontsize=16)
        
        if self.save_path:
            plt.savefig(self.save_path, bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    # Load the XDF file
    data, header = pyxdf.load_xdf(XDF_FILE)

    time_series_headers = ['O2Hb', 'HHb', 'Brain oxy', 'Brain state']

    # Add time series to dataframe
    fnirs_df = pd.DataFrame(data[0]['time_series'], columns=time_series_headers)

    # Add timestamps
    fnirs_df.insert(0, 'Timestamp', data[0]['time_stamps'])

    # Calculate time differences between consecutive timestamps
    time_diffs = np.diff(fnirs_df['Timestamp'])

    # Calculate average sampling frequency (Hz)
    avg_sampling_frequency = 1 / np.mean(time_diffs)
    print(f"Average Sampling Frequency: {avg_sampling_frequency:.2f} Hz")

    baseline_sit_timestamp = 0
    baseline_stand_timestamp = 260.5799999999999
    anticipation_timestamp = 560.5860000000002
    interview_timestamp = 941.1890000000003
    arithmetic_timestamp = 1245.7160000000003
    goodbye_timestamp = 1546.3100000000004

    crop_time = 7

    # Instantiate preprocessor and preprocess data
    preprocessor = FNIRSPreprocessing(fnirs_df, fs=SAMPLING_FREQUENCY)
    preprocessed_df = preprocessor.process()

    # Crop the data as needed
    preprocessed_df = preprocessed_df.iloc[crop_time*SAMPLING_FREQUENCY:].reset_index(drop=True)
    preprocessed_df = preprocessed_df.iloc[:int(goodbye_timestamp*SAMPLING_FREQUENCY)].reset_index(drop=True)

    markers = [baseline_sit_timestamp, baseline_stand_timestamp, anticipation_timestamp, interview_timestamp, arithmetic_timestamp, goodbye_timestamp]
    title_labels = ["Oxyhemoglobin Concentration", 'Deoxyhemoglobin Concentration', 'Brain Oxygenation' ]
    legend_labels = ['Baseline Sit', 'Baseline Stand', 'Anticipation', 'Interview', 'Arithmetic']

    # Instantiate visualizer and plot plots
    # visualizer = fNIRSVisualizer(sampling_frequency=SAMPLING_FREQUENCY)
    # visualizer.plot_plots(
    #     [preprocessed_df['O2Hb'].values, preprocessed_df['HHb'].values, preprocessed_df['Brain oxy'].values],
    #     ['O2Hb (µM)', 'HHb (µM)', 'Brain Oxy (%)'],
    #     title_labels=title_labels,
    #     legend_labels=legend_labels,
    #     plot_fft=False,
    #     markers=markers
    # )

    fe = FNIRSFeatureExtractor(preprocessed_df.iloc[:int(anticipation_timestamp*SAMPLING_FREQUENCY)].reset_index(drop=True))
    fe_df = fe.extract_features()

    print("Non-stressed")
    print(fe_df)

    fe = FNIRSFeatureExtractor(preprocessed_df.iloc[int(anticipation_timestamp*SAMPLING_FREQUENCY):].reset_index(drop=True))
    fe_df = fe.extract_features()

    print("Stressed")
    print(fe_df)

