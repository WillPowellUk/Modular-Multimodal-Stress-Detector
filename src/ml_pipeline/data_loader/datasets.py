import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class AugmentedDataset(Dataset):
    def __init__(self, features_path, **kwargs):
        self.features_path = features_path
        self.labels = kwargs.get('labels', [])
        self.exclude_subjects = kwargs.get('exclude_subjects', [])
        self.include_subjects = kwargs.get('include_subjects', [])
        self.include_augmented = kwargs.get('include_augmented', True)
        self.include_sensors = kwargs.get('include_sensors', [])
        self.include_features = kwargs.get('include_features', [])

    def preprocess_and_save(self, output_path):
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
                os.makedirs(directory)

        with h5py.File(self.features_path, 'r') as hdf5_file:
            with h5py.File(output_path, 'w') as new_hdf5_file:
                sample_idx = 0
                for subject in hdf5_file.keys():
                    subject_id = int(subject.split('_')[1])
                    
                    # if self.exclude_subjects and subject_id in self.exclude_subjects:
                    #     continue
                    
                    # if self.include_subjects and subject_id not in self.include_subjects:
                    #     continue

                    for aug in hdf5_file[subject].keys():
                        is_augmented = aug.split('_')[1] == 'True'

                        if not self.include_augmented and is_augmented:
                            continue

                        for label in hdf5_file[subject][aug].keys():
                            if label not in self.labels:
                                continue
                            for batch in hdf5_file[subject][aug][label].keys():
                                data = []
                                for sensor in hdf5_file[subject][aug][label][batch].keys():
                                    if sensor not in self.include_sensors:
                                        continue
                                    
                                    for feature in hdf5_file[subject][aug][label][batch][sensor].keys():
                                        if feature not in self.include_features:
                                            print(f'Feature: {feature} for sensor {sensor} not in include_features list')
                                            continue
                                        data.append(hdf5_file[subject][aug][label][batch][sensor][feature][:])
                                
                                # Save the preprocessed sample
                                data_label = np.concatenate((np.array(data).flatten(), np.array([float(label)])))
                                new_hdf5_file.create_dataset(f'data_label_{sample_idx}', data=data_label)
                                sample_idx += 1

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pass

class PerSensorDataset(Dataset):
    def __init__(self, features_path, **kwargs):
        self.features_path = features_path
        self.labels = kwargs.get('labels', [])
        self.exclude_subjects = kwargs.get('exclude_subjects', [])
        self.include_subjects = kwargs.get('include_subjects', [])
        self.include_augmented = kwargs.get('include_augmented', True)
        self.include_sensors = kwargs.get('include_sensors', [])
        self.include_features = kwargs.get('include_features', [])

    def preprocess_and_save(self, output_path):
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
                os.makedirs(directory)

        sample_idx = 0
        with h5py.File(self.features_path, 'r') as hdf5_file:
            with h5py.File(output_path, 'w') as new_hdf5_file:
                for subject in hdf5_file.keys():
                    subject_id = int(subject.split('_')[1])
                    
                    if self.exclude_subjects and subject_id in self.exclude_subjects:
                        continue
                    
                    if self.include_subjects and subject_id not in self.include_subjects:
                        continue

                    for aug in hdf5_file[subject].keys():
                        is_augmented = aug.split('_')[1] == 'True'

                        if not self.include_augmented and is_augmented:
                            continue

                        for label in hdf5_file[subject][aug].keys():
                            if label not in self.labels:
                                continue

                            for b, batch in enumerate(hdf5_file[subject][aug][label].keys()):
                                batch_group = new_hdf5_file.require_group(str(sample_idx))
                                for sensor in hdf5_file[subject][aug][label][batch].keys():
                                    if sensor not in self.include_sensors:
                                        continue
                                    sensor_group = batch_group.require_group(sensor)
                                    data = []
                                    for feature in hdf5_file[subject][aug][label][batch][sensor].keys():
                                        if feature not in self.include_features:
                                            print(f'Feature: {feature} for sensor {sensor} not in include_features list')
                                            continue
                                        data.append(hdf5_file[subject][aug][label][batch][sensor][feature][:])
                                
                                    # Save the preprocessed sample
                                    sensor_group.create_dataset(f'data', data=np.array(data))
                                    sensor_group.create_dataset(f'label', data=float(label))
                                sample_idx += 1
                            print(f'Processed batches for subject {subject_id} and label {label} and aug {aug}')

    def preprocess_and_save(self, train_test_val_split, save_paths):
        # Check if the save paths are valid
        if len(save_paths) != 3:
            raise ValueError("Three save paths are required for train, validation, and test datasets.")

        train_split = train_test_val_split['train']
        val_split = train_test_val_split['val']
        test_split = train_test_val_split['test']
        
        assert train_split + val_split + test_split == 1.0, "The sum of train, validation, and test splits must be 1.0"

        for path in save_paths:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Initialize datasets
        all_data = []

        with h5py.File(self.features_path, 'r') as hdf5_file:
            for subject in hdf5_file.keys():
                subject_id = int(subject.split('_')[1])

                if self.exclude_subjects and subject_id in self.exclude_subjects:
                    continue

                if self.include_subjects and subject_id not in self.include_subjects:
                    continue

                for aug in hdf5_file[subject].keys():
                    is_augmented = aug.split('_')[1] == 'True'

                    for label in hdf5_file[subject][aug].keys():
                        if label not in self.labels:
                            continue

                        for b, batch in enumerate(hdf5_file[subject][aug][label].keys()):
                            sample = {'is_augmented': is_augmented}
                            for sensor in hdf5_file[subject][aug][label][batch].keys():
                                if sensor not in self.include_sensors:
                                    continue
                                data = []
                                for feature in hdf5_file[subject][aug][label][batch][sensor].keys():
                                    if feature not in self.include_features:
                                        print(f'Feature: {feature} for sensor {sensor} not in include_features list')
                                        continue
                                    data.append(hdf5_file[subject][aug][label][batch][sensor][feature][:])
                                
                                # Save the preprocessed sample
                                sample[sensor] = {'data': np.array(data), 'label': float(label)}
                            all_data.append(sample)
                        print(f'Processed batches for subject {subject_id} and label {label} and aug {aug}')

        # Split the data
        train_val_data, test_data = train_test_split(all_data, test_size=test_split, random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=val_split/(train_split + val_split), random_state=42)

        # Filter the augmented data
        train_data = [sample for sample in train_data if sample['is_augmented']]
        val_data = [sample for sample in val_data if not sample['is_augmented']]
        test_data = [sample for sample in test_data if not sample['is_augmented']]

        # Save data to hdf5 files
        for i, (dataset, name) in enumerate(zip([train_data, val_data, test_data], ['train', 'val', 'test'])):
            sample_idx = 0
            with h5py.File(save_paths[i], 'w') as new_hdf5_file:
                for sample in dataset:
                    batch_group = new_hdf5_file.require_group(str(sample_idx))
                    for sensor, sensor_data in sample.items():
                        if sensor == 'is_augmented':
                            continue
                        sensor_group = batch_group.require_group(sensor)
                        sensor_group.create_dataset('data', data=sensor_data['data'])
                        sensor_group.create_dataset('label', data=sensor_data['label'])
                    sample_idx += 1
                print(f'Saved {name} data to {save_paths[i]}')
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pass

class SensorDataset(Dataset):
    def __init__(self, features_path, include_sensors):
        self.features_path = features_path
        self.include_sensors = include_sensors
        with h5py.File(self.features_path, 'r') as hdf5_file:
            self.data_keys = list(hdf5_file.keys())
        self.dataset_length = len(self.data_keys)

    def get_dims(self):
        with h5py.File(self.features_path, 'r') as hdf5_file:
            for key in hdf5_file.keys():
                data_dict = {}
                for sensor in self.include_sensors:
                    data_dict[sensor] = torch.tensor(hdf5_file[key][sensor]['data'][:], dtype=torch.float32)
                break
        
        return {sensor: data_dict[sensor].shape[0] for sensor in self.include_sensors}

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.features_path, 'r') as hdf5_file:
            sample_key = self.data_keys[idx]
            data_dict = {}
            for sensor in self.include_sensors:
                data_dict[sensor] = torch.tensor(hdf5_file[sample_key][sensor][f'data'][:], dtype=torch.float32)
            label = torch.tensor(int(hdf5_file[sample_key][sensor][f'label'][()]), dtype=torch.long)
        
        return data_dict, label
