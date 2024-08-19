import pyxdf
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# from src.ml_pipeline.preprocessing.fnirs_preprocessing import FNIRSPreprocessing
# from src.ml_pipeline.feature_extraction.manual.fnirs_feature_extractor import FNIRSFeatureExtractor


# Configure Matplotlib to use LaTeX for rendering
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",  # Use serif font in conjunction with LaTeX
#     # Set the default font to be used in LaTeX as a single string
#     "text.latex.preamble": r"\usepackage{times}",
# })

class fNIRSVisualizer:
    def __init__(self, sampling_frequency, save_path=None):
        self.sampling_frequency = sampling_frequency
        self.save_path = save_path

    def plot_plots(self, plots, y_labels, title_labels, legend_labels=None, plot_fft=False, markers=None, HR=None, BVP=None, BVP_sampling_frequency=None):
        num_plots = len(plots)
        
        # Create vertical subplots with increased space between them
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 5), constrained_layout=True)
        
        # If there's only one plot, axes might not be an array, so we make it a list
        if num_plots == 1:
            axes = [axes]
        
        for i, plot in enumerate(plots):
            # Create time axis for fNIRS data
            t_fnirs = np.arange(len(plot)) / self.sampling_frequency
            
            # Plot the original fNIRS data
            axes[i].plot(t_fnirs, plot, label='fNIRS', color='blue')
            
            # If BVP data is provided, proceed with interpolation and scaling
            if BVP is not None and BVP_sampling_frequency is not None:
                # Create time axis for BVP data
                t_bvp = np.arange(len(BVP)) / BVP_sampling_frequency
                
                # Interpolate fNIRS to match the BVP time axis
                fnirs_interpolator = interp1d(t_fnirs, plot, kind='linear', fill_value="extrapolate")
                fnirs_resampled = fnirs_interpolator(t_bvp)
                
                # Scale BVP data to match the magnitude and mean of fNIRS data
                bvp_mean = np.mean(BVP)
                bvp_std = np.std(BVP)
                fnirs_mean = np.mean(fnirs_resampled)
                fnirs_std = np.std(fnirs_resampled)
                
                # BVP_scaled = (BVP - bvp_mean) * (fnirs_std / bvp_std)
                BVP_scaled = (BVP * 0.01) + 75
                
                # Plot the scaled BVP data
                axes[i].plot(t_bvp, BVP_scaled, label='Scaled BVP', linestyle='--', color='orange')
                
                # Add the legend
                axes[i].legend(title="Legend", loc='upper right', fontsize=12, title_fontsize='13')
            
            # Set labels, titles, grid, and tick parameters
            axes[i].set_xlabel('Time (s)', fontsize=16)
            axes[i].set_ylabel(y_labels[i], fontsize=16)
            axes[i].set_title(title_labels[i], fontsize=20)
            axes[i].grid(True)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            
            # Add vertical markers and shaded regions if provided
            if markers:
                for j in range(len(markers) - 1):
                    axes[i].axvline(x=markers[j], color='r', linestyle='--')
                    axes[i].axvline(x=markers[j+1], color='r', linestyle='--')
                    axes[i].fill_between(t_fnirs, plot.min(), plot.max(), where=(t_fnirs >= markers[j]) & (t_fnirs < markers[j+1]), color=f'C{j}', alpha=0.1)
            
            # Add RR intervals if provided
            if HR is not None:
                for hr in HR:
                    axes[i].axvline(x=hr, color='red', linestyle=':')
                # Add the legend for the RR intervals
                axes[i].legend(title="Legend", loc='upper right', fontsize=12, title_fontsize='13')
        
        # Adjust layout and space for the custom legend
        plt.tight_layout(h_pad=7)
        fig.subplots_adjust(bottom=0.16)
        
        # Create custom legend handles for the shaded regions if markers and legend_labels are provided
        if markers and legend_labels:
            legend_handles = [plt.Rectangle((0,0),1,1, fc=f'C{i}', alpha=0.3) for i in range(len(legend_labels))]
            fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=len(legend_labels), fontsize=16)
        
        # Save or show the plot
        if self.save_path:
            plt.savefig(self.save_path, bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    subject_id = 1
    FNIRS_SAMPLING_FREQUENCY = 10  # Hz
    XDF_FILE = f'data_collection/recordings/S{subject_id}/myndsens/FNIRS.xdf'

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
    preprocessed_df = fnirs_df

    # Crop the data as needed
    preprocessed_df = preprocessed_df.iloc[crop_time*FNIRS_SAMPLING_FREQUENCY:].reset_index(drop=True)
    preprocessed_df = preprocessed_df.iloc[:int(goodbye_timestamp*FNIRS_SAMPLING_FREQUENCY)].reset_index(drop=True)

    markers = [baseline_sit_timestamp, baseline_stand_timestamp, anticipation_timestamp, interview_timestamp, arithmetic_timestamp, goodbye_timestamp]
    title_labels = ["Oxyhemoglobin Concentration", 'Deoxyhemoglobin Concentration', 'Brain Oxygenation' ]
    legend_labels = ['Baseline Sit', 'Baseline Stand', 'Anticipation', 'Interview', 'Arithmetic']

    hr_crop=20

    # Read the RR intervals
    rr_intervals = pd.read_csv(f'data_collection/recordings/S{subject_id}/polar/HR.csv', header=None)
    rr_intervals = rr_intervals.values.flatten()
    hr_intervals = np.cumsum(rr_intervals)  / 1000.0
    hr_intervals = hr_intervals[hr_crop:]

    # BVP
    BVP_SAMPLING_FREQUENCY = 64
    save_path = f'data_collection/data_visualization/plots/S{subject_id}_ECG.pdf'
    segment = np.loadtxt(f'data_collection/recordings/S{subject_id}/empatica/BVP.csv')
    crop_time = 20
    ppg_segment = segment[crop_time * BVP_SAMPLING_FREQUENCY:]

    # Instantiate visualizer and plot plots
    visualizer = fNIRSVisualizer(sampling_frequency=FNIRS_SAMPLING_FREQUENCY)
    visualizer.plot_plots(
        [preprocessed_df['O2Hb'].values, preprocessed_df['HHb'].values, preprocessed_df['Brain oxy'].values],
        ['O2Hb (µM)', 'HHb (µM)', 'Brain Oxy (%)'],
        title_labels=title_labels,
        legend_labels=legend_labels,
        plot_fft=False,
        markers=markers,
        HR=None,
        BVP=ppg_segment,
        BVP_sampling_frequency=BVP_SAMPLING_FREQUENCY  # Provide BVP sampling frequency
    )


    # fe = FNIRSFeatureExtractor(preprocessed_df.iloc[:int(anticipation_timestamp*FNIRS_SAMPLING_FREQUENCY)].reset_index(drop=True))
    # fe_df = fe.extract_features()

    # print("Non-stressed")
    # print(fe_df)

    # fe = FNIRSFeatureExtractor(preprocessed_df.iloc[int(anticipation_timestamp*FNIRS_SAMPLING_FREQUENCY):].reset_index(drop=True))
    # fe_df = fe.extract_features()

    # print("Stressed")
    # print(fe_df)

