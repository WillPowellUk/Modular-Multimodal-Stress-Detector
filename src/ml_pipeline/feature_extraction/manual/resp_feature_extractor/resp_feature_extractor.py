import neurokit2 as nk
import pandas as pd
import numpy as np


class RespFeatureExtractor:
    def __init__(self, resp_data: pd.DataFrame, sampling_rate: int = 1000):
        # Assuming resp_data is a DataFrame with a single column containing the RESP signal
        self.resp_data = resp_data.values
        self.sampling_rate = sampling_rate

    def extract_features(self):
        # Initialize dictionary to hold features
        features = {}
        # # Find respiratory peaks
        # peaks = nk.rsp_findpeaks(self.resp_data, sampling_rate=self.sampling_rate)

        # Process the Resp signal to extract phases and other info 
        rsp_processed, info = nk.rsp_process(
            self.resp_data, sampling_rate=self.sampling_rate
        )

        # Extract mean and standard deviation of inhalation and exhalation
        interval_related = nk.rsp_intervalrelated(
            rsp_processed, sampling_rate=self.sampling_rate
        )
        inhalation_durations = interval_related["RSP_Phase_Duration_Inspiration"]
        exhalation_durations = interval_related["RSP_Phase_Duration_Expiration"]
        features["Mean_Inhalation_Duration"] = np.mean(inhalation_durations)
        features["STD_Inhalation_Duration"] = np.std(inhalation_durations)
        features["Mean_Exhalation_Duration"] = np.mean(exhalation_durations)
        features["STD_Exhalation_Duration"] = np.std(exhalation_durations)

        # Calculate the inhalation/exhalation ratio
        if np.mean(exhalation_durations) != 0:
            features["Inhalation_Exhalation_Ratio"] = np.mean(
                inhalation_durations
            ) / np.mean(exhalation_durations)

        # Combine all features into a single DataFrame
        all_features = pd.concat(
            [
                pd.DataFrame([features]),  # Extracted simple features as DataFrame
                interval_related.reset_index(drop=True),  # Interval related features
            ],
            axis=1,
        )

        return all_features
