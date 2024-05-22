import pandas as pd
import json
import pickle
import os
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

    def extract_features_from_batch(self, batch):
        features_list = []

        if self.wrist:
            if 'w_eda' in self.sampling_rates:
                print("Extracting wrist EDA features...")
                eda_features = EDAFeatureExtractor(batch['w_eda'], self.sampling_rates['w_eda']).extract_features()
                features_list.append(eda_features)
            if 'w_bvp' in self.sampling_rates:
                print("Extracting wrist BVP features...")
                bvp_features = BVPFeatureExtractor(batch['w_bvp'], self.sampling_rates['w_bvp']).extract_features()
                features_list.append(bvp_features)
            if 'w_acc' in self.sampling_rates:
                print("Extracting wrist ACC features...")
                acc_df = pd.DataFrame({
                    'x': batch['w_acc_x'],
                    'y': batch['w_acc_y'],
                    'z': batch['w_acc_z']
                })
                acc_features = AccFeatureExtractor(acc_df, self.sampling_rates['w_acc']).extract_features()
                features_list.append(acc_features)
            if 'w_temp' in self.sampling_rates:
                print("Extracting wrist TEMP features...")
                temp_features = TempFeatureExtractor(batch['w_temp']).extract_features()
                features_list.append(temp_features)
        else:
            # if 'eda' in self.sampling_rates:
            #     print("Extracting EDA features...")
            #     eda_features = EDAFeatureExtractor(batch['eda'], self.sampling_rates['eda']).extract_features()
            #     features_list.append(eda_features)
            # if 'acc' in self.sampling_rates:
            #     print("Extracting ACC features...")
            #     acc_df = pd.DataFrame({
            #         'x': batch['acc1'],
            #         'y': batch['acc2'],
            #         'z': batch['acc3']
            #     })
            #     acc_features = AccFeatureExtractor(acc_df, self.sampling_rates['acc']).extract_features()
            #     features_list.append(acc_features)
            if 'ecg' in self.sampling_rates:
                print("Extracting ECG features...")
                ecg_features = ECGFeatureExtractor(batch['ecg'], self.sampling_rates['ecg']).extract_features()
                features_list.append(ecg_features)
            if 'emg' in self.sampling_rates:
                print("Extracting EMG features...")
                emg_features = EMGFeatureExtractor(batch['emg'], self.sampling_rates['emg']).extract_features()
                features_list.append(emg_features)
            if 'resp' in self.sampling_rates:
                print("Extracting RESP features...")
                resp_features = RespFeatureExtractor(batch['resp'], self.sampling_rates['resp']).extract_features()
                features_list.append(resp_features)
            if 'temp' in self.sampling_rates:
                print("Extracting TEMP features...")
                temp_features = TempFeatureExtractor(batch['temp']).extract_features()
                features_list.append(temp_features)

        # Combine all features into a single DataFrame
        all_features = pd.concat(features_list, axis=1)
        return all_features

    def extract_features(self):
        all_batches_features = []
        for batch in self.batches:
            for df in batch:
                batch_features = self.extract_features_from_batch(df)
                all_batches_features.append(batch_features)

        # Concatenate all batch features
        final_features = pd.concat(all_batches_features, axis=0)
        
        # Ensure the save path directory exists
        save_dir = os.path.dirname(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save features as pkl
        with open(self.save_path, 'wb') as file:
            pickle.dump(final_features, file)

        return final_features
