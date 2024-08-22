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

    def plot_plots(self, plots, y_labels, title_labels, legend_labels=None, plot_fft=False, markers=None, HR=None, BVP=None, BVP_sampling_frequency=None, ECG=None, ECG_sampling_frequency=None):
        num_plots = len(plots)
        
        # Create vertical subplots with increased space between them
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 5), constrained_layout=True)
        
        # If there's only one plot, axes might not be an array, so we make it a list
        if num_plots == 1:
            axes = [axes]
        

        for i, plot in enumerate(plots):
            # Create time axis for fNIRS data
            t_fnirs = np.arange(len(plot)) / self.sampling_frequency

            # Normalize fNIRS data to range 0-1000
            fnirs_min, fnirs_max = np.min(plot), np.max(plot)
            fnirs_normalized = (plot - fnirs_min) / (fnirs_max - fnirs_min) * 1000

            # Plot the normalized fNIRS data
            axes[i].plot(t_fnirs, fnirs_normalized + 300, label='fNIRS', color='blue')

            # If BVP and ECG data are provided, proceed with interpolation, scaling, and normalization
            if BVP is not None and BVP_sampling_frequency is not None and ECG is not None and ECG_sampling_frequency is not None:
                # Create time axis for BVP and ECG data
                t_bvp = np.arange(len(BVP)) / BVP_sampling_frequency
                t_ecg = np.arange(len(ECG)) / ECG_sampling_frequency

                # Interpolate fNIRS to match the BVP and ECG time axes
                interpolator_fnirs = interp1d(t_fnirs, fnirs_normalized, kind='linear', fill_value="extrapolate")
                fnirs_resampled = interpolator_fnirs(t_ecg)
                
                # Normalize BVP data to range 0-1000
                bvp_min, bvp_max = np.min(BVP), np.max(BVP)
                BVP_normalized = (BVP - bvp_min) / (bvp_max - bvp_min) * 1000

                # Normalize ECG data to range 0-1000
                ecg_min, ecg_max = np.min(ECG), np.max(ECG)
                ECG_normalized = (ECG - ecg_min) / (ecg_max - ecg_min) * 1000

                # Plot the normalized BVP and ECG data
                axes[i].plot(t_bvp, BVP_normalized + 200, label='BVP', color='orange')
                axes[i].plot(t_ecg, ECG_normalized, label='ECG', color='red')

                # Add the legend
                axes[i].legend(loc='upper right', fontsize=12, title_fontsize='13')
            
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
                import neurokit2 as nk
                signals, info = nk.ecg_peaks(ECG_normalized, sampling_rate=130)

                # Extract R-peaks indices
                r_peaks = info["ECG_R_Peaks"]
                r_peaks = r_peaks / 130.0

                # Plot R-peaks
                for r, peak in enumerate(r_peaks):
                    if r == 0:
                        axes[i].axvline(x=peak, color='red', linestyle='--', label='R Peak')
                    else:
                        axes[i].axvline(x=peak, color='red', linestyle='--')

                # for h, hr in enumerate(HR):
                #     if h == 100:
                #         break
                #     axes[i].axvline(x=hr, color='red', linestyle=':')
                # Add the legend for the RR intervals
                axes[i].legend(loc='upper right', fontsize=12, title_fontsize='13')

            axes[i].set_xlim(20, 44)
            current_xticks = axes[i].get_xticks()

            # Calculate the new labels by subtracting the minimum x-value (e.g., 20 seconds) to start from 0
            new_xticklabels = current_xticks - 20

            # Update the x-tick labels with the new labels
            axes[i].set_xticks(current_xticks, new_xticklabels)          
            
            # Calculate the automatic ylim
            y_min, y_max = axes[i].get_ylim()

            # Generate 7 y-ticks between y_min and y_max
            y_ticks = np.linspace(y_min, y_max, 7)

            # Set the y-ticks with no labels
            axes[i].set_yticks(y_ticks)
            axes[i].set_yticklabels([''] * len(y_ticks))
            axes[i].grid(True)  # Enable grid        
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
    subject_id = 69
    FNIRS_SAMPLING_FREQUENCY = 10.04  # Hz
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
    preprocessed_df = fnirs_df[int(1.55*FNIRS_SAMPLING_FREQUENCY):].reset_index(drop=True)

    # Crop the data as needed
    # preprocessed_df = preprocessed_df.iloc[crop_time*FNIRS_SAMPLING_FREQUENCY:].reset_index(drop=True)
    # preprocessed_df = preprocessed_df.iloc[:int(goodbye_timestamp*FNIRS_SAMPLING_FREQUENCY)].reset_index(drop=True)

    markers = [baseline_sit_timestamp, baseline_stand_timestamp, anticipation_timestamp, interview_timestamp, arithmetic_timestamp, goodbye_timestamp]
    title_labels = ["", 'Deoxyhemoglobin Concentration', 'Brain Oxygenation' ] # Oxyhemoglobin Concentration
    legend_labels = ['Baseline Sit', 'Baseline Stand', 'Anticipation', 'Interview', 'Arithmetic']

    # BVP
    BVP_SAMPLING_FREQUENCY = 64
    save_path = f'data_collection/data_visualization/plots/S{subject_id}_ECG.pdf'
    ppg_segment = np.loadtxt(f'data_collection/recordings/S{subject_id}/empatica/BVP.csv')
    ppg_segment = ppg_segment[int(15.3 * BVP_SAMPLING_FREQUENCY):]
    # crop_time = 20
    # ppg_segment = ppg_segment[crop_time * BVP_SAMPLING_FREQUENCY:]

    # ECG
    ECG_SAMPLING_RATE = 130
    ecg_segment = np.loadtxt(f'data_collection/recordings/S{subject_id}/polar/ECG.csv', delimiter=',')
    ecg_segment = ecg_segment[int(4.2 * ECG_SAMPLING_RATE):]
    # crop_time = 40
    # ecg_segment = ecg_segment[crop_time * ECG_SAMPLING_RATE:]

    # RR intervals
    rr_intervals = pd.read_csv(f'data_collection/recordings/S{subject_id}/polar/HR.csv', header=None)
    rr_intervals = rr_intervals.values.flatten()
    hr_intervals = np.cumsum(rr_intervals)  / 1000.0
    # start_index = np.argmax(hr_intervals > crop_time)
    # hr_intervals = hr_intervals[start_index:]

    # Instantiate visualizer and plot plots
    visualizer = fNIRSVisualizer(sampling_frequency=FNIRS_SAMPLING_FREQUENCY)
    visualizer.plot_plots(
        [preprocessed_df['O2Hb'].values, preprocessed_df['HHb'].values, preprocessed_df['Brain oxy'].values],
        ['Amplitude', 'HHb (µM)', 'Brain Oxy (%)'], # O2Hb (µM)
        title_labels=title_labels,
        legend_labels=legend_labels,
        plot_fft=False,
        markers=markers,
        HR=hr_intervals,
        BVP=ppg_segment,
        BVP_sampling_frequency=BVP_SAMPLING_FREQUENCY,  # Provide BVP sampling frequency
        ECG=ecg_segment,
        ECG_sampling_frequency=ECG_SAMPLING_RATE  # Provide ECG sampling frequency
    )


    # fe = FNIRSFeatureExtractor(preprocessed_df.iloc[:int(anticipation_timestamp*FNIRS_SAMPLING_FREQUENCY)].reset_index(drop=True))
    # fe_df = fe.extract_features()

    # print("Non-stressed")
    # print(fe_df)

    # fe = FNIRSFeatureExtractor(preprocessed_df.iloc[int(anticipation_timestamp*FNIRS_SAMPLING_FREQUENCY):].reset_index(drop=True))
    # fe_df = fe.extract_features()

    # print("Stressed")
    # print(fe_df)

