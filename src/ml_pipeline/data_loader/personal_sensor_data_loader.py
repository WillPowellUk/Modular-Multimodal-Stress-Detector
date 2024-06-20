import pickle
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_active_key, get_values
from src.ml_pipeline.data_loader.datasets import PerSensorDataset, SensorDataset

class PersonalSensorDataLoader:
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
        self.cv_splits = get_values(config_path, 'cv_splits')
    
    def _get_dataset(self, save_paths_template, include_subjects):
        config = {
            **self.dataset_config,
            'include_subjects': include_subjects
        }
        dataset = PerSensorDataset(self.features_path, **config)
        return dataset.preprocess_and_save_cross_validation(self.cv_splits, save_paths_template)
    
    def prepare_datasets(self, save_path, subject_id):
        print(f'\nPreparing dataset')
        save_paths_template = save_path + f"/subject_{subject_id}" + "/fold_{split}_{type}.hdf5"
        datesets_path = self._get_dataset(save_paths_template, include_subjects=[subject_id])
        print(f'Dataset prepared\n')
        
        # save dataset paths as pkl file
        dataset_save_path = f'{save_path}/personal_datasets.pkl'
        with open(dataset_save_path, 'wb') as f:
            pickle.dump(datesets_path, f)
        print(f'Datasets saved at: {dataset_save_path}')
        return dataset_save_path

    def get_data_loaders(self, datasets_path):
        with open(datasets_path, 'rb') as f:
            datasets_path = pickle.load(f)

        dataloaders = []
        input_dims = {}

        for dataset_path in datasets_path:
            train_dataset = SensorDataset(dataset_path['train'], self.dataset_config['include_sensors'])
            val_dataset = SensorDataset(dataset_path['val'], self.dataset_config['include_sensors'])
            train_loader = DataLoader(train_dataset, **self.params)
            val_loader = DataLoader(val_dataset, **self.params)
            dataloaders.append({'train': train_loader, 'val': val_loader})
            
        input_dims = train_dataset.get_dims()
        return dataloaders, input_dims
