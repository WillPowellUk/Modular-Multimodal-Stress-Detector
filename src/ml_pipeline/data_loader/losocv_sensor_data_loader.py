import pickle
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_active_key
from src.ml_pipeline.data_loader.datasets import PerSensorDataset

class LOSOCVSensorDataLoader:
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
        dataset = PerSensorDataset(self.features_path, **config)
        dataset.preprocess_and_save(save_path)
    
    def prepare_datasets(self):
        datesets_path = {}
        for subject_id in self.subjects:
            subject_id = int(float(subject_id))
            train_dataset_path = f'{os.path.dirname(self.features_path)}/losocv/train_{subject_id}.hdf5'
            val_dataset_path = f'{os.path.dirname(self.features_path)}/losocv/val_{subject_id}.hdf5'
            self._get_dataset(train_dataset_path, exclude_subjects=[subject_id], include_augmented=True)
            self._get_dataset(val_dataset_path, include_subjects=[subject_id], include_augmented=False)
            datesets_path[subject_id] = {'train': train_dataset_path, 'val': val_dataset_path}
        
        # save dataset paths as pkl file
        save_path = os.path.dirname(self.features_path) + '/losocv_datasets.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(datesets_path, f)
        print(f'Datasets saved at: {save_path}')
        return save_path

    def get_data_loaders(self, datasets_path):
        with open(datasets_path, 'rb') as f:
            datasets_path = pickle.load(f)

        dataloaders = {}
        input_dims = {}
        for i, subject_id in enumerate(self.subjects):
            subject_id = int(float(subject_id))
            train_dataset = LOSOCVSesnsorDataset(datasets_path[subject_id]['train'], self.dataset_config['include_sensors'])
            val_dataset = LOSOCVSesnsorDataset(datasets_path[subject_id]['val'], self.dataset_config['include_sensors'])

            if i == 0:
                input_dims = train_dataset.get_dims()

            train_loader = DataLoader(train_dataset, **self.params)
            val_loader = DataLoader(val_dataset, **self.params)

            dataloaders[subject_id] = {'train': train_loader, 'val': val_loader}
        
        return dataloaders, input_dims

class LOSOCVSesnsorDataset(Dataset):
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
                    data_dict[sensor] = torch.tensor(hdf5_file[key][sensor]['data_0'][:], dtype=torch.float32)
                break
        
        return {sensor: data_dict[sensor].shape[0] for sensor in self.include_sensors}

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.features_path, 'r') as hdf5_file:
            sample_key = self.data_keys[idx]
            data_dict = {}
            for sensor in self.include_sensors:
                data_dict[sensor] = torch.tensor(hdf5_file[sample_key][sensor][:-1], dtype=torch.float32)
            label = torch.tensor(int(hdf5_file[sample_key][self.include_sensors[0]][-1]), dtype=torch.long)
        
        return data_dict, label
