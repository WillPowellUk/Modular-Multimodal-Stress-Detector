import neurokit2 as nk
import pandas as pd

class RespFeatureExtractor:
    def __init__(self, resp_data: pd.DataFrame, sampling_rate: int = 1000):
        self.resp_data = resp_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        features = {}

        # Use the preprocessed Resp signal directly
        resp_signal = self.resp_data['Resp'].values

        # Process the Resp signal
        signals, info = nk.rsp_process(resp_signal, sampling_rate=self.sampling_rate)

        # Extract time domain features
        time_domain = nk.rsp_amplitude(signals, sampling_rate=self.sampling_rate)
        
        # Extract frequency domain features
        frequency_domain = nk.rsp_frequency(signals, sampling_rate=self.sampling_rate)

        # Extract other features
        rate_features = nk.rsp_rate(signals, sampling_rate=self.sampling_rate)
        event_features = nk.rsp_eventrelated(signals)

        # Combine all features into a single DataFrame
        all_features = pd.concat([time_domain, frequency_domain, rate_features, event_features], axis=1)

        return all_features