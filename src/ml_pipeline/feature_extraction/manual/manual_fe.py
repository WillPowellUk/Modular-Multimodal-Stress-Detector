import pandas as pd
import json
import os
import time
import warnings
from .eda_feature_extractor import EDAFeatureExtractor
from .bvp_feature_extractor import BVPFeatureExtractor
from .acc_feature_extractor import AccFeatureExtractor
from .ecg_feature_extractor import ECGFeatureExtractor
from .emg_feature_extractor import EMGFeatureExtractor
from .resp_feature_extractor import RespFeatureExtractor
from .temp_feature_extractor import TempFeatureExtractor

class ManualFE:
    def __init__(self, batches, save_path: str, config_path: str, wrist=False):
        self.batches = batches
        self.save_path = save_path
        self.wrist = wrist
        # Load the JSON data from the file
        with open(config_path, 'r') as file:
            self.sampling_rates = json.load(file)

        # Ignore runtime warning for mean of empty slice
        warnings.filterwarnings("ignore", message="Mean of empty slice")

    def extract_features_from_batch(self, batch):
        features_dict = {}

        if self.wrist:
            if 'w_eda' in self.sampling_rates:
                eda_features = EDAFeatureExtractor(batch['w_eda'], self.sampling_rates['w_eda']).extract_features()
                features_dict['w_eda'] = eda_features
            if 'w_bvp' in self.sampling_rates:
                bvp_features = BVPFeatureExtractor(batch['w_bvp'], self.sampling_rates['w_bvp']).extract_features()
                features_dict['w_bvp'] = bvp_features
            if 'w_acc' in self.sampling_rates:
                acc_df = pd.DataFrame({
                    'x': batch['w_acc_x'],
                    'y': batch['w_acc_y'],
                    'z': batch['w_acc_z']
                })
                acc_features = AccFeatureExtractor(acc_df, self.sampling_rates['w_acc']).extract_features()
                features_dict['w_acc'] = acc_features
            if 'w_temp' in self.sampling_rates:
                temp_features = TempFeatureExtractor(batch['w_temp']).extract_features()
                features_dict['w_temp'] = temp_features
        else:
            if 'eda' in self.sampling_rates:
                eda_features = EDAFeatureExtractor(batch['eda'], self.sampling_rates['eda']).extract_features()
                features_dict['eda'] = eda_features
            if 'acc' in self.sampling_rates:
                acc_df = pd.DataFrame({
                    'x': batch['acc1'],
                    'y': batch['acc2'],
                    'z': batch['acc3']
                })
                acc_features = AccFeatureExtractor(acc_df, self.sampling_rates['acc']).extract_features()
                features_dict['acc'] = acc_features
            if 'ecg' in self.sampling_rates:
                ecg_features = ECGFeatureExtractor(batch['ecg'], self.sampling_rates['ecg']).extract_features()
                features_dict['ecg'] = ecg_features
            if 'emg' in self.sampling_rates:
                emg_features = EMGFeatureExtractor(batch['emg'], self.sampling_rates['emg']).extract_features()
                features_dict['emg'] = emg_features
            if 'resp' in self.sampling_rates:
                resp_features = RespFeatureExtractor(batch['resp'], self.sampling_rates['resp']).extract_features()
                features_dict['resp'] = resp_features
            if 'temp' in self.sampling_rates:
                temp_features = TempFeatureExtractor(batch['temp']).extract_features()
                features_dict['temp'] = temp_features

        # Combine all feature DataFrames into one DataFrame for each category
        all_features = {key: pd.concat(val, axis=1) if isinstance(val, list) else val for key, val in features_dict.items()}
        return all_features
    
    def extract_features(self):
        warnings.warn_explicit = warnings.warn = lambda *_, **__: None
        warnings.filterwarnings("ignore")
         
        all_batches_features = []
        total_batches = len(self.batches)
        start_time = time.time()
        
        for i, (sid, batch) in enumerate(self.batches):
            elapsed_time = time.time() - start_time
            average_time_per_batch = elapsed_time / (i + 1)
            remaining_batches = total_batches - (i + 1)
            eta = average_time_per_batch * remaining_batches

            print(f"Extracting features from batch {i+1}/{total_batches} | ETA: {eta:.2f} seconds")

            batch_features = self.extract_features_from_batch(batch)
            all_batches_features.append(batch_features)

            if i ==3:
                break

        # Ensure the directory exists
        dir_name = os.path.dirname(self.save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        # Save the features
        with open(self.save_path, 'wb') as file:
            pd.to_pickle(all_batches_features, file)

        return all_batches_features
    