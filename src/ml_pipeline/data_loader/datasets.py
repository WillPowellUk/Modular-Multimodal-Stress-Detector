import h5py
import numpy as np
from torch.utils.data import Dataset
import os

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