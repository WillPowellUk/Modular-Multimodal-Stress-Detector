import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key

class AugmentedDataset(Dataset):
    def __init__(self, features_path, sensors, labels, exclude_subject=None, include_augmented=True):
        self.features_path = features_path
        self.sensors = sensors
        self.labels = labels
        self.exclude_subject = exclude_subject
        self.include_augmented = include_augmented
        self.data_info = self._gather_data_info()

    def _gather_data_info(self):
        data_info = []
        
        with h5py.File(self.features_path, 'r') as hdf5_file:
            for subject in hdf5_file.keys():
                if self.exclude_subject is not None and subject.split('_')[1] == str(self.exclude_subject):
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
        subject, aug, data_idx = self.data_info[idx]
        
        with h5py.File(self.features_path, 'r') as hdf5_file:
            feature_data = []
            for sensor in self.sensors:
                if sensor in hdf5_file[subject][aug]:
                    sensor_data = hdf5_file[subject][aug][sensor][data_idx]
                    feature_data.append(sensor_data)
            
            sample = np.concatenate(feature_data)
            label = hdf5_file[subject][aug]['label'][data_idx]
        
        data = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return data, label

class LOSOCVDataLoader:
    def __init__(self, features_path, config_path, **params):
        self.features_path = features_path
        self.sensors = get_active_key(config_path, 'sensors')
        self.subjects = get_active_key(config_path, 'subjects')
        self.labels = get_active_key(config_path, 'labels')
        self.params = params

    def get_dataset(self, exclude_subject=None, include_augmented=True):
        dataset = AugmentedDataset(self.features_path, self.sensors, self.labels, exclude_subject, include_augmented)
        return dataset

    def get_dataloaders(self):
        dataloaders = {}

        for subject_id in self.subjects:
            train_dataset = self.get_dataset(exclude_subject=subject_id, include_augmented=True)
            val_dataset = self.get_dataset(exclude_subject=subject_id, include_augmented=False)

            train_loader = DataLoader(train_dataset, **self.params)
            val_loader = DataLoader(val_dataset, **self.params)

            dataloaders[subject_id] = {'train': train_loader, 'val': val_loader}
        
        return dataloaders
