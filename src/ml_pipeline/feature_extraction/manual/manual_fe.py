import pandas as pd
import json
from .eda_feature_extractor import EDAFeatureExtractor
from .bvp_feature_extractor import BVPFeatureExtractor
from .acc_feature_extractor import AccFeatureExtractor
from .ecg_feature_extractor import ECGFeatureExtractor
from .emg_feature_extractor import EMGFeatureExtractor
from .resp_feature_extractor import RespFeatureExtractor
from .temp_feature_extractor import TempFeatureExtractor

class ManualFE():
    def __init__(self, dataframe_path: str, save_path: str, config_path: str, wrist=False):
        self.dataframe = pd.read_pickle(dataframe_path)
        self.save_path = save_path
        self.wrist = wrist
        # Load the JSON data from the file
        with open(config_path, 'r') as file:
            self.sampling_rates = json.load(file)

    def extract_features(self):
        features_list = []

        if self.wrist:
            if 'w_eda' in self.sampling_rates:
                print("Extracting wrist EDA features...")
                eda_features = EDAFeatureExtractor(self.dataframe['w_eda'], self.sampling_rates['w_eda']).extract_features()
                features_list.append(eda_features)
            if 'w_bvp' in self.sampling_rates:
                print("Extracting wrist BVP features...")
                bvp_features = BVPFeatureExtractor(self.dataframe['w_bvp'], self.sampling_rates['w_bvp']).extract_features()
                features_list.append(bvp_features)
            if 'w_acc' in self.sampling_rates:
                print("Extracting wrist ACC features...")
                acc_df = pd.DataFrame({
                    'x': self.dataframe['w_acc_x'],
                    'y': self.dataframe['w_acc_y'],
                    'z': self.dataframe['w_acc_z']
                })
                acc_features = AccFeatureExtractor(acc_df, self.sampling_rates['w_acc']).extract_features()
                features_list.append(acc_features)
            if 'w_temp' in self.sampling_rates:
                print("Extracting wrist TEMP features...")
                temp_features = TempFeatureExtractor(self.dataframe['w_temp']).extract_features()
                features_list.append(temp_features)
        else:
            if 'eda' in self.sampling_rates:
                print("Extracting EDA features...")
                eda_features = EDAFeatureExtractor(self.dataframe['eda'], self.sampling_rates['eda']).extract_features()
                features_list.append(eda_features)
            if 'acc' in self.sampling_rates:
                print("Extracting ACC features...")
                acc_df = pd.DataFrame({
                    'x': self.dataframe['acc1'],
                    'y': self.dataframe['acc2'],
                    'z': self.dataframe['acc3']
                })
                acc_features = AccFeatureExtractor(acc_df, self.sampling_rates['acc']).extract_features()
                features_list.append(acc_features)
            if 'ecg' in self.sampling_rates:
                print("Extracting ECG features...")
                ecg_features = ECGFeatureExtractor(self.dataframe['ecg'], self.sampling_rates['ecg']).extract_features()
                features_list.append(ecg_features)
            if 'emg' in self.sampling_rates:
                print("Extracting EMG features...")
                emg_features = EMGFeatureExtractor(self.dataframe['emg'], self.sampling_rates['emg']).extract_features()
                features_list.append(emg_features)
            if 'resp' in self.sampling_rates:
                print("Extracting RESP features...")
                resp_features = RespFeatureExtractor(self.dataframe['resp'], self.sampling_rates['resp']).extract_features()
                features_list.append(resp_features)
            if 'temp' in self.sampling_rates:
                print("Extracting TEMP features...")
                temp_features = TempFeatureExtractor(self.dataframe['temp']).extract_features()
                features_list.append(temp_features)

        # Combine all features into a single DataFrame
        all_features = pd.concat(features_list, axis=1)

        # Save features as pkl 
        all_features.to_pickle(self.save_path)

        return all_features
