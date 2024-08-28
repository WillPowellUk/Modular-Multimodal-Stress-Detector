import neurokit2 as nk
import pandas as pd


class BVPFeatureExtractor:
    def __init__(self, bvp_data: pd.DataFrame, sampling_rate: int = 1000):
        self.bvp_data = bvp_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        # Use the preprocessed BVP signal directly
        bvp_signal = self.bvp_data.values
        bvp_signal = pd.to_numeric(bvp_signal.flatten(), errors='coerce')        

        # Process BVP signal to find peaks
        signals, info = nk.ppg_process(bvp_signal, sampling_rate=self.sampling_rate)

        # Extract time domain features
        time_domain = nk.hrv_time(signals, sampling_rate=self.sampling_rate)

        # Extract frequency domain features
        frequency_domain = nk.hrv_frequency(signals, sampling_rate=self.sampling_rate)

        try:
            # Extract nonlinear features
            nonlinear_features = nk.hrv_nonlinear(
                signals, sampling_rate=self.sampling_rate
            )
        except:
            all_features = pd.concat([time_domain, frequency_domain], axis=1)
            return all_features

        # Combine all features into a single DataFrame
        all_features = pd.concat(
            [time_domain, frequency_domain, nonlinear_features], axis=1
        )

        return all_features
