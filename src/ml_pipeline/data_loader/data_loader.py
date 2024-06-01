import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key

class AugmentedDataset(Dataset):
    def __init__(self, features_path, sensors, labels, exclude_subjects=None, include_subjects=None, include_augmented=True):
        self.features_path = features_path
        self.sensors = sensors
        self.labels = labels
        self.exclude_subjects = exclude_subjects if exclude_subjects is not None else []
        self.include_subjects = include_subjects if include_subjects is not None else []
        self.include_augmented = include_augmented
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
                        
                        num_samples = len(hdf5_file[subject][aug][label].keys())
                        for idx in range(num_samples):
                            data_info.append((subject, aug, label, idx))
        
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        subject, aug, label, data_idx = self.data_info[idx]
        
        with h5py.File(self.features_path, 'r') as hdf5_file:
            feature_data = []
            group = hdf5_file[subject][aug][label]
            for key in group.keys():
                if not key.endswith('_columns'):
                    sensor_data = group[key][data_idx]
                    feature_data.append(sensor_data)
            
            sample = np.concatenate(feature_data)
            label_data = group['label'][data_idx]
        
        data = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label_data, dtype=torch.long)
        
        return data, label
    
class LOSOCVDataLoader:
    def __init__(self, features_path, config_path, **params):
        self.features_path = features_path
        self.sensors = get_active_key(config_path, 'sensors')
        self.subjects = get_active_key(config_path, 'subjects')
        self.labels = get_active_key(config_path, 'labels')
        self.params = params

    def get_dataset(self, exclude_subjects=None, include_subjects=None, include_augmented=True):
        dataset = AugmentedDataset(self.features_path, self.sensors, self.labels, exclude_subjects, include_subjects, include_augmented)
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
