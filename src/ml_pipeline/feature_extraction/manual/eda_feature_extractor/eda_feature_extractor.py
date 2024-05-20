import neurokit2 as nk
import pandas as pd

class EDAFeatureExtractor:
    def __init__(self, eda_data: pd.DataFrame, sampling_rate: int = 1000):
        self.eda_data = eda_data
        self.sampling_rate = sampling_rate

    def extract_features(self):
        # Use the preprocessed EDA signal directly
        eda_signal = self.eda_data['eda'].values
        
        # Decompose the EDA signal into phasic and tonic components
        eda_phasic, eda_tonic = nk.eda_phasic(eda_signal, sampling_rate=self.sampling_rate)
        
        # Extract features from the tonic component
        tonic_features = nk.eda_tonic(eda_tonic, sampling_rate=self.sampling_rate)
        
        # Extract features from the phasic component
        phasic_features = nk.eda_phasic_features(eda_phasic, sampling_rate=self.sampling_rate)
        
        # Extract EDA event-related features
        eda_events = nk.eda_events(eda_signal, sampling_rate=self.sampling_rate)
        
        # Convert event-related features to a DataFrame
        eda_events_df = pd.DataFrame({"EDA_Events": eda_events})

        # Combine all features into a single DataFrame
        all_features = pd.concat([tonic_features, phasic_features, eda_events_df], axis=1)
        
        return all_features