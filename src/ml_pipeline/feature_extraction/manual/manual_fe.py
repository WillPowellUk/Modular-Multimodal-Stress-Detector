import pandas as pd
import re
import os
import time
import warnings
import numpy as np
import h5py
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key
from .eda_feature_extractor import EDAFeatureExtractor
from .bvp_feature_extractor import BVPFeatureExtractor
from .acc_feature_extractor import AccFeatureExtractor
from .ecg_feature_extractor import ECGFeatureExtractor
from .emg_feature_extractor import EMGFeatureExtractor
from .resp_feature_extractor import RespFeatureExtractor
from .temp_feature_extractor import TempFeatureExtractor

class ManualFE:
    def __init__(self, batches, save_path: str, config_path: str):
        self.batches = batches
        self.save_path = save_path
        self.sensors = get_active_key(config_path, 'sensors')
        self.sampling_rate = get_max_sampling_rate(config_path)

        # Ignore runtime warning for mean of empty slice
        warnings.filterwarnings("ignore", message="Mean of empty slice")

    def extract_features_from_batch(self, batch):
        features_dict = {}

        sid = batch['sid'].iloc[0]
        is_augmented = batch['is_augmented'].iloc[0]
        features_dict['sid'] = sid
        features_dict['is_augmented'] = is_augmented

        for sensor in self.sensors:
            match sensor:
                case 'w_eda':
                    eda_features = EDAFeatureExtractor(batch['w_eda'], self.sampling_rate).extract_features()
                    features_dict['w_eda'] = eda_features
                case 'bvp':
                    bvp_features = BVPFeatureExtractor(batch['bvp'], self.sampling_rate).extract_features()
                    features_dict['bvp'] = bvp_features
                case 'w_temp':
                    temp_features = TempFeatureExtractor(batch['w_temp'], self.sampling_rate).extract_features()
                    features_dict['w_temp'] = temp_features
                case 'eda':
                    eda_features = EDAFeatureExtractor(batch['eda'], self.sampling_rate).extract_features()
                    features_dict['eda'] = eda_features
                case 'ecg':
                    ecg_features = ECGFeatureExtractor(batch['ecg'], self.sampling_rate).extract_features()
                    features_dict['ecg'] = ecg_features
                case 'emg':
                    emg_features = EMGFeatureExtractor(batch['emg'], self.sampling_rate).extract_features()
                    features_dict['emg'] = emg_features
                case 'resp':
                    resp_features = RespFeatureExtractor(batch['resp'], self.sampling_rate).extract_features()
                    features_dict['resp'] = resp_features
                case 'temp':
                    temp_features = TempFeatureExtractor(batch['temp'], self.sampling_rate).extract_features()
                    features_dict['temp'] = temp_features
                case _:
                    if re.search(r'w_acc', sensor):
                        acc_df = pd.DataFrame({
                            'x': batch['w_acc_x'],
                            'y': batch['w_acc_y'],
                            'z': batch['w_acc_z']
                        })
                        acc_features = AccFeatureExtractor(acc_df, self.sampling_rate).extract_features()
                        features_dict['w_acc'] = acc_features
                    elif re.search(r'(?<!w_)acc', sensor):
                        acc_df = pd.DataFrame({
                            'x': batch['acc1'],
                            'y': batch['acc2'],
                            'z': batch['acc3']
                        })
                        acc_features = AccFeatureExtractor(acc_df, self.sampling_rate).extract_features()
                        features_dict['acc'] = acc_features

        features_dict['label'] = batch['label'].iloc[0]

        return features_dict
    
    def save_to_hdf5(self, all_batches_features):
        with h5py.File(self.save_path, 'w') as hdf5_file:
            for features in all_batches_features:
                sid = str(int(features.pop('sid')))
                is_augmented = 'augmented_True' if features.pop('is_augmented') else 'augmented_False'
                label = str(features.pop('label'))

                subject_group = hdf5_file.require_group(f'subject_{sid}')
                augmented_group = subject_group.require_group(is_augmented)
                label_group = augmented_group.require_group(label)

                for sensor_name, feature_data in features.items():
                    sensor_group = label_group.require_group(sensor_name)

                    if isinstance(feature_data, pd.DataFrame):
                        for column in feature_data.columns:
                            feature_group = sensor_group.require_group(column)
                            feature_values = feature_data[column].values
                            feature_group.create_dataset('values', data=feature_values)
                    else:
                        feature_group = sensor_group.require_group(sensor_name)
                        feature_group.create_dataset('values', data=feature_data)


    def extract_features(self):
        warnings.warn_explicit = warnings.warn = lambda *_, **__: None
        warnings.filterwarnings("ignore")
         
        all_batches_features = []
        total_batches = len(self.batches)
        start_time = time.time()
        
        for i, batch in enumerate(self.batches):
            elapsed_time = time.time() - start_time
            average_time_per_batch = elapsed_time / (i + 1)
            remaining_batches = total_batches - (i + 1)
            eta = average_time_per_batch * remaining_batches

            if i % 100 == 0:
                print(f"Extracting features from batch {i+1}/{total_batches} | ETA: {eta:.2f} seconds")

            batch_features = self.extract_features_from_batch(batch)
            all_batches_features.append(batch_features)

            if i == 1:
                break

        # Ensure the directory exists
        dir_name = os.path.dirname(self.save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        # Save the features to HDF5
        self.save_to_hdf5(all_batches_features)
