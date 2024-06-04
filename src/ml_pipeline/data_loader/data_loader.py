import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key

class AugmentedDataset(Dataset):
    def __init__(self, features_path, **kwargs):
        self.features_path = features_path
        self.labels = kwargs.get('labels', [])
        self.exclude_subjects = kwargs.get('exclude_subjects', [])
        self.include_subjects = kwargs.get('include_subjects', [])
        self.include_augmented = kwargs.get('include_augmented', True)
        self.include_sensors = kwargs.get('include_sensors', [])
        self.include_features = kwargs.get('include_features', [])

        self.data_info = self._gather_data_info()

    def _gather_data_info(self):
        data_info = []

        with h5py.File(self.features_path, 'r') as hdf5_file:
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
                        
                        min_feature_length = np.inf
                        for sensor in hdf5_file[subject][aug][label].keys():
                            if sensor not in self.include_sensors:
                                continue
                            
                            for feature in hdf5_file[subject][aug][label][sensor].keys():
                                if feature not in self.include_features:
                                    continue
                                if len(hdf5_file[subject][aug][label][sensor][feature]) < min_feature_length:
                                    min_feature_length = len(hdf5_file[subject][aug][label][sensor][feature])

                        for idx in range(min_feature_length):
                            data_info.append((subject, aug, label, idx))
        
        self.data_info = data_info
        return data_info
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        subject, aug, label, index = self.data_info[idx]
        data = []
        with h5py.File(self.features_path, 'r') as hdf5_file:
            for sensor in hdf5_file[subject][aug][label].keys():
                if sensor not in self.include_sensors:
                    continue
                
                for feature in hdf5_file[subject][aug][label][sensor].keys():
                    if feature not in self.include_features:
                        continue
                    data.append(hdf5_file[subject][aug][label][sensor][feature][index])
        
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(int(float(label)), dtype=torch.long)
        
        return data, label_tensor
    
class LOSOCVDataLoader:
    def __init__(self, features_path, config_path, **params):
        self.features_path = features_path
        self.dataset_config = {
            'include_sensors': get_active_key(config_path, 'sensors'),
            'include_features': get_active_key(config_path, 'features', recursive=True),
            'labels': get_active_key(config_path, 'labels')
        }
        self.subjects = get_active_key(config_path, 'subjects')
        self.params = params

    def get_dataset(self, exclude_subjects=None, include_subjects=None, include_augmented=True):
        config = {
            **self.dataset_config,
            'exclude_subjects': exclude_subjects,
            'include_subjects': include_subjects,
            'include_augmented': include_augmented
        }
        dataset = AugmentedDataset(self.features_path, **config)
        return dataset

    def get_data_loaders(self):
        dataloaders = {}

        for subject_id in self.subjects:
            subject_id = int(float(subject_id))
            train_dataset = self.get_dataset(exclude_subjects=[subject_id], include_augmented=True)
            val_dataset = self.get_dataset(include_subjects=[subject_id], include_augmented=False)

            train_loader = DataLoader(train_dataset, **self.params)
            val_loader = DataLoader(val_dataset, **self.params)

            dataloaders[subject_id] = {'train': train_loader, 'val': val_loader}
        
        return dataloaders
