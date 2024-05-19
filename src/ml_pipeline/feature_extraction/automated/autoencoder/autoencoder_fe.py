import pandas as pd
from submodules.pyEDA.main import *

class EDAFeatureExtractionAE:
    def __init__(self, ecg_data: pd.DataFrame, sampling_rate: int = 1000):
        self.ecg_data = ecg_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        # Use the preprocessed ECG signal directly
        eda_signal = self.ecg_data['EDA'].values

        prepare_automatic(eda_signal, sample_rate=self.sampling_rate, new_sample_rate=40, k=32, epochs=100, batch_size=10)
        automatic_features = process_automatic(eda_signal)