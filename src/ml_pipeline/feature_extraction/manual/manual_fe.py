import pandas as pd
import concurrent.futures
from multiprocessing import cpu_count
import os
import math
import time
import warnings
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

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

    def extract_features_from_split(self, split):
        features_dict = {}
        features_dict['sid'] = split['sid'].iloc[0]
        features_dict['is_augmented'] = split['is_augmented'].iloc[0]

        for sensor in self.sensors:
            match sensor:
                case 'w_eda':
                    eda_features = EDAFeatureExtractor(split['w_eda'], self.sampling_rate).extract_features()
                    features_dict['w_eda'] = eda_features
                case 'bvp':
                    bvp_features = BVPFeatureExtractor(split['bvp'], self.sampling_rate).extract_features()
                    features_dict['bvp'] = bvp_features
                case 'w_temp':
                    temp_features = TempFeatureExtractor(split['w_temp'], self.sampling_rate).extract_features()
                    features_dict['w_temp'] = temp_features
                case 'eda':
                    eda_features = EDAFeatureExtractor(split['eda'], self.sampling_rate).extract_features()
                    features_dict['eda'] = eda_features
                case 'ecg':
                    ecg_features = ECGFeatureExtractor(split['ecg'], self.sampling_rate).extract_features()
                    features_dict['ecg'] = ecg_features
                case 'emg':
                    emg_features = EMGFeatureExtractor(split['emg'], self.sampling_rate).extract_features()
                    features_dict['emg'] = emg_features
                case 'resp':
                    resp_features = RespFeatureExtractor(split['resp'], self.sampling_rate).extract_features()
                    features_dict['resp'] = resp_features
                case 'temp':
                    temp_features = TempFeatureExtractor(split['temp'], self.sampling_rate).extract_features()
                    features_dict['temp'] = temp_features
                case 'w_acc':
                    acc_df = pd.DataFrame({
                        'x': split['w_acc_x'],
                        'y': split['w_acc_y'],
                        'z': split['w_acc_z']
                    })
                    acc_features = AccFeatureExtractor(acc_df, self.sampling_rate).extract_features()
                    features_dict['w_acc'] = acc_features
                case 'acc':
                    acc_df = pd.DataFrame({
                        'x': split['acc1'],
                        'y': split['acc2'],
                        'z': split['acc3']
                    })
                    acc_features = AccFeatureExtractor(acc_df, self.sampling_rate).extract_features()
                    features_dict['acc'] = acc_features

        features_dict['label'] = split['label'].iloc[0]

        return features_dict
    
    def save_to_hdf5(self, all_batches_features):
        print(f'Saving features to {self.save_path}...')
        with h5py.File(self.save_path, 'w') as hdf5_file:
            for b, batch in enumerate(all_batches_features):
                for i, features in enumerate(batch):
                    sid = str(int(features.pop('sid')))
                    is_augmented = 'augmented_True' if features.pop('is_augmented') else 'augmented_False'
                    label = str(features.pop('label'))

                    subject_group = hdf5_file.require_group(f'subject_{sid}')
                    augmented_group = subject_group.require_group(is_augmented)
                    label_group = augmented_group.require_group(label)
                    batch_group = label_group.require_group(str(b))

                    for j, (sensor_name, feature_data) in enumerate(features.items()):
                        sensor_group = batch_group.require_group(sensor_name)

                        if isinstance(feature_data, pd.DataFrame):
                            for k, column in enumerate(feature_data.columns):
                                # Concatenate all values for each feature across each batch
                                if column in sensor_group:
                                    dataset = sensor_group[column]
                                    dataset.resize((dataset.shape[0] + feature_data.shape[0],))
                                    dataset[-feature_data.shape[0]:] = feature_data[column].values
                                else:
                                    sensor_group.create_dataset(column, data=feature_data[column].values, maxshape=(None,))
                        else:
                            raise ValueError(f"Unknown feature data type: {type(feature_data)}")
        print('Features saved successfully')


    def impute_and_normalize_features(self, all_batches_features):
        print('Scaling and imputing missing values...')
        
        # Collecting all features for scaling and imputation
        feature_data = {}        
        for batch in all_batches_features:
            for minibatch in batch:
                for key, value in minibatch.items():
                    if key in ['sid', 'label', 'is_augmented']:
                        continue
                    if key not in feature_data:
                        feature_data[key] = []
                    feature_data[key].append(value)

        # Scale and then impute missing values
        scaler = StandardScaler()
        for key, data_list in feature_data.items():
            if all(isinstance(d, pd.DataFrame) for d in data_list):
                # For DataFrame features
                combined_df = pd.concat(data_list)
                
                # Replace inf and -inf with NaN to handle them uniformly
                combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Normalize the features
                normalized_df = pd.DataFrame(scaler.fit_transform(combined_df), columns=combined_df.columns)
                
                # Impute missing values
                mean_values = normalized_df.mean()
                mean_values.fillna(0, inplace=True)  # Handling case where mean itself is NaN
                
                for i, df in enumerate(data_list):
                    data_list[i] = normalized_df.iloc[i * len(df):(i + 1) * len(df)].copy()
                    data_list[i].fillna(mean_values, inplace=True)

            else:
                raise ValueError(f"Unknown feature data type in {data_list}")

        # Replace scaled and imputed features back into all_batches_features
        for batch in all_batches_features:
            for minibatch in batch:
                for key in minibatch.keys():
                    if key in ['sid', 'label', 'is_augmented']:
                        continue
                    minibatch[key] = feature_data[key].pop(0)

        print('Scaling and imputation complete')
        return all_batches_features
    
    def extract_features(self):
        warnings.warn_explicit = warnings.warn = lambda *_, **__: None
        warnings.filterwarnings("ignore")
         
        all_batches_features = []
        total_batches = len(self.batches)
        start_time = time.time()
        
        for i, batch in enumerate(self.batches):
            if i % 100 == 0 and i!=0:
                elapsed_time = time.time() - start_time
                average_time_per_batch = elapsed_time / (i + 1)
                remaining_batches = total_batches - (i + 1)
                eta = average_time_per_batch * remaining_batches
                hours = math.floor(eta / 3600)
                minutes = math.floor((eta % 3600) / 60)
                seconds = eta % 60

                # Print the formatted string
                print(f"Extracting features from batch {i+1}/{total_batches} | ETA: {hours}h {minutes}m {seconds:.2f}s")
            
            batch_features = []
            for split in batch:
                split_features = self.extract_features_from_split(split)
                batch_features.append(split_features)
            all_batches_features.append(batch_features)

            if i == 203:
                break

        all_batches_features = self.impute_and_normalize_features(all_batches_features)

        # Ensure the directory exists
        dir_name = os.path.dirname(self.save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        # Save the features to HDF5
        self.save_to_hdf5(all_batches_features)
