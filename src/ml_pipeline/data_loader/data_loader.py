import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key
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
                            # if label not in self.labels:
                            #     continue
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

class LOSOCVDataLoader:
    def __init__(self, features_path, config_path, **params):
        self.features_path = features_path
        self.config_path = config_path
        self.dataset_config = {
            'include_sensors': get_active_key(config_path, 'sensors'),
            'include_features': get_active_key(config_path, 'features', recursive=True),
            'labels': get_active_key(config_path, 'labels')
        }
        self.subjects = get_active_key(config_path, 'subjects')
        self.params = params
    
    def _get_dataset(self, save_path, exclude_subjects=None, include_subjects=None, include_augmented=True):
        config = {
            **self.dataset_config,
            'exclude_subjects': exclude_subjects,
            'include_subjects': include_subjects,
            'include_augmented': include_augmented
        }
        dataset = AugmentedDataset(self.features_path, **config)
        dataset.preprocess_and_save(save_path)
    
    def prepare_datasets(self, save_path):
        datesets_path = {}
        for subject_id in self.subjects:
            subject_id = int(float(subject_id))
            train_dataset_path = f'{os.path.dirname(self.features_path)}/losocv/train_{subject_id}.hdf5'
            val_dataset_path = f'{os.path.dirname(self.features_path)}/losocv/val_{subject_id}.hdf5'
            self._get_dataset(train_dataset_path, exclude_subjects=[subject_id], include_augmented=True)
            self._get_dataset(val_dataset_path, include_subjects=[subject_id], include_augmented=False)
            datesets_path[subject_id] = {'train': train_dataset_path, 'val': val_dataset_path}
        
        # save dataset paths as pkl file
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
                os.makedirs(directory)
        with open(save_path, 'wb') as f:
            pickle.dump(datesets_path, f)

    def get_data_loaders(self, datasets_path):
        with open(datasets_path, 'rb') as f:
            datasets_path = pickle.load(f)

        dataloaders = {}
        for subject_id in self.subjects:
            subject_id = int(float(subject_id))
            train_dataset = LOSOCVDataset(datasets_path[subject_id]['train'])
            val_dataset = LOSOCVDataset(datasets_path[subject_id]['val'])

            train_loader = DataLoader(train_dataset, **self.params)
            val_loader = DataLoader(val_dataset, **self.params)

            dataloaders[subject_id] = {'train': train_loader, 'val': val_loader}
        
        return dataloaders

class LOSOCVDataset(Dataset):
    def __init__(self, features_path):
        self.features_path = features_path

        with h5py.File(self.features_path, 'r') as hdf5_file:
            self.data_keys = list(hdf5_file.keys())
        self.dataset_length = len(self.data_keys)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.features_path, 'r') as hdf5_file:
            sample_key = self.data_keys[idx]
            data = torch.tensor(hdf5_file[sample_key][:-1], dtype=torch.float32)
            label = torch.tensor(int(hdf5_file[sample_key][-1]), dtype=torch.long)
        
        return data, label
