import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
import math

# Configure Matplotlib to use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use serif font in conjunction with LaTeX
    # Set the default font to be used in LaTeX as a single string
    "text.latex.preamble": r"\usepackage{times}",
    })


class ECGFeatureExtractor:
    min_HR = 40
    max_HR = 220

    def __init__(self, ecg_data: pd.DataFrame, sampling_rate: int = 1000):
        np.seterr(divide="ignore", invalid="ignore")
        self.ecg_data = ecg_data.values
        if self.ecg_data.ndim != 1:
            self.ecg_data = self.ecg_data.flatten()
            
        self.sampling_rate = sampling_rate

    def plot_segment(
        self,
        segment,
        ecg_processed=None,
        peaks=None,
        colors=["r", "g", "c", "m", "y", "k"],
    ):
        t = np.arange(len(segment)) / self.sampling_rate
        if isinstance(peaks, type(None)):
            fft = np.fft.fft(self.ecg_data)
            freq = np.fft.fftfreq(len(self.ecg_data), d=1 / self.sampling_rate)
            max_freq = 100
            mask = (freq >= 0) & (freq <= max_freq)
            freq = freq[mask]
            fft = fft[mask]
            psd = (np.abs(fft) ** 2) / len(self.ecg_data)
            psd = 10 * np.log10(psd)
            psd -= psd.max()
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(t, self.ecg_data)
            ax.set_xlabel("Time (s)", fontsize=40)
            ax.set_ylabel("Amplitude", fontsize=40)
            ax.set_yticklabels([])
            subax = fig.add_axes([0.68, 0.65, 0.2, 0.2])
            subax.plot(freq, psd)
            subax.set_xlim(0, max_freq)
            subax.set_xlabel("Frequency (Hz)")
            subax.set_ylabel("PSD (dB)")
        else:
            plt.figure()
            plt.plot(t, self.ecg_data)
            lines = [
                Line2D([0], [0], linestyle="--", color=colors[i])
                for i in range(len(peaks))
            ]
            for i, peak in enumerate(peaks):
                # peak_inds = np.where(ecg_processed[peak] == 1)[0]
                peak_inds = ecg_processed[peak]
                for ind in peak_inds:
                    if not math.isnan(ind):
                        plt.axvline(x=t[ind], linestyle="--", color=colors[i])
            plt.legend(handles=lines, labels=peaks, loc="lower right", fontsize=40)
            plt.xlabel("Time (s)", fontsize=40)
            plt.ylabel("Amplitude", fontsize=40)
            plt.xlim(30, 35)
            plt.xticks(ticks=np.linspace(30, 35, 6), labels=np.arange(0, 6), fontsize=40)
            plt.gca().set_yticklabels([])
            plt.show()

    def wave_analysis(self, ecg_processed, plot=False) -> pd.DataFrame:
        peaks = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]
        min_interval = 60e6 / self.max_HR
        max_interval = 60e6 / self.min_HR

        df = pd.DataFrame()
        for peak in peaks:
            intervals = (
                np.diff(np.where(np.atleast_1d(np.array(ecg_processed[peak]) == 1)))
                * self.sampling_rate
            )

            intervals = intervals[
                (intervals >= min_interval) & (intervals <= max_interval)
            ]
            df[f"{peak}_Interval_Mean"] = [np.mean(intervals)]
            df[f"{peak}_Interval_SD"] = [np.std(intervals)]

        if plot:
            self.plot_segment(self.ecg_data, ecg_processed, peaks)

        waves = ["P", "R", "T"]
        max_duration = [120000, 120000, 200000]
        for w, wave in enumerate(waves):
            onsets = np.where(
                np.atleast_1d(np.array(ecg_processed[f"ECG_{wave}_Onsets"]) == 1)
            )[0]
            offsets = np.where(
                np.atleast_1d(np.array(ecg_processed[f"ECG_{wave}_Offsets"]) == 1)
            )[0]

            if len(onsets) == 0 or len(offsets) == 0:
                continue
            idx_offset = np.where(offsets >= onsets[0])[0][0]
            duration_size = min(onsets.size, offsets.size)
            offsets = offsets[idx_offset:duration_size]
            onsets = onsets[:duration_size]
            durations = []
            i = 0
            j = 0
            while i < len(offsets) and j < len(onsets):
                diff = offsets[i] - onsets[j]
                if diff < 0:
                    i += 1
                else:
                    durations.append(diff)
                    i += 1
                    j += 1
            durations = np.array(durations * self.sampling_rate)
            durations = durations[(durations <= max_duration[w])]
            duration_mean = np.mean(durations)
            duration_SD = np.std(durations)
            df[f"ECG_{wave}_Duration_Mean"] = duration_mean
            df[f"ECG_{wave}_Duration_SD"] = duration_SD

        wave_onsets_offsets = [f"ECG_{wave}_Onsets" for wave in waves] + [
            f"ECG_{wave}_Offsets" for wave in waves
        ]

        if plot:
            self.plot_segment(
                self.ecg_data,
                ecg_processed,
                wave_onsets_offsets,
                colors=["r", "r", "g", "g", "b", "b"],
            )

        return df

    def calc_PSD(self) -> pd.DataFrame:
        PSD = nk.signal_psd(
            self.ecg_data,
            sampling_rate=self.sampling_rate,
            method="welch",
            min_frequency=0.5,
            max_frequency=200,
        )
        binned_power = []
        frequency = 0
        bin_range = 10
        nyquist_frequency = self.sampling_rate // 2
        while bin_range <= nyquist_frequency:
            total_power = 0
            for index, row in PSD.iterrows():
                if row["Frequency"] >= frequency and row["Frequency"] < bin_range:
                    total_power += row["Power"]
            if total_power > 0:
                binned_power.append(np.log10(total_power))
            else:
                binned_power.append(-np.inf)
            frequency += 10
            bin_range += 10

        binned_PSD = pd.DataFrame({"Power": binned_power})
        binned_PSD["Frequency Band"] = list(range(10, nyquist_frequency + 1, 10))
        ECG_Frequencies = pd.DataFrame(
            columns=[f"ECG_FQ_{i}" for i in range(10, nyquist_frequency + 1, 10)]
        )
        for i, column in enumerate(ECG_Frequencies.columns):
            ECG_Frequencies[column] = [binned_PSD.iloc[i]["Power"]]

        return ECG_Frequencies

    def calc_EDR(self, r_peaks_df, show_plot=False) -> pd.DataFrame:
        warnings.filterwarnings("ignore")
        ecg_rate = nk.signal_rate(
            r_peaks_df, sampling_rate=self.sampling_rate, desired_length=len(r_peaks_df)
        )
        EDR_sample = nk.ecg_rsp(ecg_rate, sampling_rate=self.sampling_rate)
        if show_plot:
            nk.signal_plot(self.ecg_data)
            nk.signal_plot(EDR_sample)
        EDR_Distances = nk.signal_findpeaks(EDR_sample)["Distance"]
        EDR_Distance = pd.DataFrame(
            [np.average(EDR_Distances)], columns=["EDR_Distance"]
        )
        diff = np.diff(EDR_Distances)
        diff_squared = diff**2
        mean_diff_squared = np.mean(diff_squared)
        rmssd = np.sqrt(mean_diff_squared)
        EDR_RMSSD = pd.Series(rmssd)
        warnings.filterwarnings("default")

        return EDR_Distance, EDR_RMSSD

    def extract_features(self, calc_PSD=False, show_plot=False) -> pd.DataFrame:
        r_peaks = nk.ecg_peaks(self.ecg_data, sampling_rate=self.sampling_rate)[0]
        np.seterr(divide="ignore", invalid="ignore")

        # skip segment if insufficient peaks are detected (otherwise will cause NK error)
        ecg_duration_minutes = (len(self.ecg_data) / self.sampling_rate) / 60

        r_peaks_detected = int(r_peaks[r_peaks == 1].sum().iloc[0])
        # exit if incorrect peaks are detected
        if r_peaks_detected < (
            self.min_HR * ecg_duration_minutes
        ) or r_peaks_detected > (self.max_HR * ecg_duration_minutes):
            return
        time_domain = nk.hrv_time(r_peaks, sampling_rate=self.sampling_rate)
        frequency_domain = nk.hrv_frequency(r_peaks, sampling_rate=self.sampling_rate)
        signals, waves = nk.ecg_delineate(
            self.ecg_data, sampling_rate=self.sampling_rate
        )

        # ecg_processed, info = nk.ecg_process(self.ecg_data, sampling_rate=self.sampling_rate, method='neurokit')
        # rri = nk.ecg_intervalrelated(ecg_processed, sampling_rate=self.sampling_rate)
        # min_scale = 4
        # max_scale = min(len(rri)//2, 50)
        # nonlinear_features = nk.hrv_nonlinear(rri, sampling_rate=self.sampling_rate, scale=range(min_scale, max_scale))

        wave_features = self.wave_analysis(waves, plot=show_plot)
        edr_distance, edr_rmssd = self.calc_EDR(r_peaks, show_plot=False)
        if calc_PSD:
            psd_features = self.calc_PSD()
            additional_features = pd.concat(
                [wave_features, psd_features, edr_distance, edr_rmssd], axis=1
            )
        else:
            additional_features = pd.concat(
                [wave_features, edr_distance, edr_rmssd], axis=1
            )
        all_features = pd.concat(
            [time_domain, frequency_domain, additional_features], axis=1
        )

        return all_features
