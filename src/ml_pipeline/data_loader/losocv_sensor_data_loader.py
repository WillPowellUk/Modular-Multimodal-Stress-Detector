import pickle
import h5py
import numpy as np
import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from src.ml_pipeline.utils.utils import get_active_key
from src.ml_pipeline.data_loader.datasets import PerSensorDataset, SensorDataset


class LOSOCVSensorDataLoader:
    def __init__(self, features_path, config_path, **params):
        self.features_path = features_path
        self.config_path = config_path
        self.dataset_config = {
            "include_sensors": get_active_key(config_path, "sensors"),
            "include_features": get_active_key(config_path, "features", recursive=True),
            "labels": get_active_key(config_path, "labels"),
        }
        self.subjects = get_active_key(config_path, "subjects")
        self.params = params

    def _get_dataset(
        self,
        save_path,
        exclude_subjects=None,
        include_subjects=None,
        include_augmented=True,
    ):
        config = {
            **self.dataset_config,
            "exclude_subjects": exclude_subjects,
            "include_subjects": include_subjects,
            "include_augmented": include_augmented,
        }
        dataset = PerSensorDataset(self.features_path, **config)
        dataset.preprocess_and_save(save_path)

    def prepare_datasets(self, save_path, n_folds=None):
        datesets_path = {}

        if n_folds is None:
            # Perform LOSOCV
            for subject_id in self.subjects:
                print(f"\nPreparing dataset for subject: {subject_id}")
                subject_id = int(float(subject_id))
                train_dataset_path = f"{save_path}/losocv/train_{subject_id}.hdf5"
                val_dataset_path = f"{save_path}/losocv/val_{subject_id}.hdf5"
                self._get_dataset(
                    train_dataset_path,
                    exclude_subjects=[subject_id],
                    include_augmented=True,
                )
                self._get_dataset(
                    val_dataset_path,
                    include_subjects=[subject_id],
                    include_augmented=False,
                )
                datesets_path[subject_id] = {
                    "train": train_dataset_path,
                    "val": val_dataset_path,
                }
                print(f"Dataset prepared for subject: {subject_id}\n")

            # Save dataset paths as pkl file
            dataset_save_path = f"{save_path}/losocv_datasets.pkl"
        else:
            # Perform N-fold cross-validation
            subjects = [int(float(subject)) for subject in self.subjects]
            random.seed(42069)
            random.shuffle(subjects)
            fold_size = len(subjects) // n_folds
            print(subjects)
            for fold in range(n_folds):
                val_subjects = subjects[fold * fold_size : (fold + 1) * fold_size]
                train_subjects = [s for s in subjects if s not in val_subjects]

                train_dataset_path = f"{save_path}/cv_{n_folds}/train_fold_{fold}.hdf5"
                val_dataset_path = f"{save_path}/cv_{n_folds}/val_fold_{fold}.hdf5"
                self._get_dataset(
                    train_dataset_path,
                    include_subjects=train_subjects,
                    include_augmented=True,
                )
                self._get_dataset(
                    val_dataset_path,
                    include_subjects=val_subjects,
                    include_augmented=False,
                )
                datesets_path[fold] = {
                    "train": train_dataset_path,
                    "val": val_dataset_path,
                }
                print(f"Dataset prepared for fold: {fold}\n")
                # Save dataset paths as pkl file
            dataset_save_path = f"{save_path}/cv_{n_folds}_datasets.pkl"

        with open(dataset_save_path, "wb") as f:
            pickle.dump(datesets_path, f)
        print(f"Datasets saved at: {dataset_save_path}")
        return dataset_save_path

    def get_data_loaders(
        self,
        datasets_path,
        dataset_type=None,
        train_only=False,
        val_only=False,
        group_labels=None,
    ):
        with open(datasets_path, "rb") as f:
            datasets_path = pickle.load(f)

        dataloaders = {}
        input_dims = {}

        if dataset_type == None or dataset_type == "losocv":
            # LOSOCV case
            for i, subject_id in enumerate(self.subjects):
                subject_id = int(float(subject_id))
                if train_only:
                    train_dataset = SensorDataset(
                        datasets_path[subject_id]["train"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    train_loader = DataLoader(train_dataset, **self.params)
                    dataloaders[subject_id] = {"train": train_loader}
                    if i == 0:
                        input_dims = train_dataset.get_dims()
                elif val_only:
                    val_dataset = SensorDataset(
                        datasets_path[subject_id]["val"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    val_loader = DataLoader(val_dataset, **self.params)
                    dataloaders[subject_id] = {"val": val_loader}
                    if i == 0:
                        input_dims = val_dataset.get_dims()
                else:
                    train_dataset = SensorDataset(
                        datasets_path[subject_id]["train"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    val_dataset = SensorDataset(
                        datasets_path[subject_id]["val"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    train_loader = DataLoader(train_dataset, **self.params)
                    val_loader = DataLoader(val_dataset, **self.params)
                    dataloaders[subject_id] = {"train": train_loader, "val": val_loader}
                    if i == 0:
                        input_dims = train_dataset.get_dims()
        else:
            # N-fold CV case
            for fold in datasets_path:
                if train_only:
                    train_dataset = SensorDataset(
                        datasets_path[fold]["train"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    train_loader = DataLoader(train_dataset, **self.params)
                    dataloaders[fold] = {"train": train_loader}
                    if fold == 0:
                        input_dims = train_dataset.get_dims()
                elif val_only:
                    val_dataset = SensorDataset(
                        datasets_path[fold]["val"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    val_loader = DataLoader(val_dataset, **self.params)
                    dataloaders[fold] = {"val": val_loader}
                    if fold == 0:
                        input_dims = val_dataset.get_dims()
                else:
                    train_dataset = SensorDataset(
                        datasets_path[fold]["train"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    val_dataset = SensorDataset(
                        datasets_path[fold]["val"],
                        self.dataset_config["include_sensors"],
                        group_labels=group_labels,
                    )
                    train_loader = DataLoader(train_dataset, **self.params)
                    val_loader = DataLoader(val_dataset, **self.params)
                    dataloaders[fold] = {"train": train_loader, "val": val_loader}
                    if fold == 0:
                        input_dims = train_dataset.get_dims()

        return dataloaders, input_dims
