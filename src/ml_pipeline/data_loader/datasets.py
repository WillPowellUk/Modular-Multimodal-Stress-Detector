import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold


class AugmentedDataset(Dataset):
    def __init__(self, features_path, **kwargs):
        self.features_path = features_path
        self.labels = kwargs.get("labels", [])
        self.exclude_subjects = kwargs.get("exclude_subjects", [])
        self.include_subjects = kwargs.get("include_subjects", [])
        self.include_augmented = kwargs.get("include_augmented", True)
        self.include_sensors = kwargs.get("include_sensors", [])
        self.include_features = kwargs.get("include_features", [])

    def preprocess_and_save(self, output_path):
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        warned_sensors = set()
        warned_features = set()

        with h5py.File(self.features_path, "r") as hdf5_file:
            with h5py.File(output_path, "w") as new_hdf5_file:
                sample_idx = 0
                for subject in hdf5_file.keys():
                    subject_id = int(subject.split("_")[1])

                    if self.exclude_subjects and subject_id in self.exclude_subjects:
                        continue

                    if (
                        self.include_subjects
                        and subject_id not in self.include_subjects
                    ):
                        continue

                    for aug in hdf5_file[subject].keys():
                        is_augmented = aug.split("_")[1] == "True"

                        if not self.include_augmented and is_augmented:
                            continue

                        for label in hdf5_file[subject][aug].keys():
                            if label not in self.labels:
                                continue
                            for batch in hdf5_file[subject][aug][label].keys():
                                data = []
                                for sensor in hdf5_file[subject][aug][label][
                                    batch
                                ].keys():
                                    if sensor not in self.include_sensors:
                                        if sensor not in warned_sensors:
                                            warned_sensors.add(sensor)
                                            print(
                                                f"Sensor '{sensor}' is not being used."
                                            )
                                        continue

                                    for feature in hdf5_file[subject][aug][label][
                                        batch
                                    ][sensor].keys():
                                        if feature not in self.include_features:
                                            if feature not in warned_features:
                                                warned_features.add(feature)
                                                print(
                                                    f"Feature: {feature} for sensor {sensor} not in include_features list"
                                                )
                                            continue
                                        data.append(
                                            hdf5_file[subject][aug][label][batch][
                                                sensor
                                            ][feature][:]
                                        )

                                # Save the preprocessed sample
                                data_label = np.concatenate(
                                    (np.array(data).flatten(), np.array([float(label)]))
                                )
                                new_hdf5_file.create_dataset(
                                    f"data_label_{sample_idx}", data=data_label
                                )
                                sample_idx += 1

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pass


class PerSensorDataset(Dataset):
    def __init__(self, features_path, **kwargs):
        self.features_path = features_path
        self.labels = kwargs.get("labels", [])
        self.exclude_subjects = kwargs.get("exclude_subjects", [])
        self.include_subjects = kwargs.get("include_subjects", [])
        self.include_augmented = kwargs.get("include_augmented", True)
        self.include_sensors = kwargs.get("include_sensors", [])
        self.include_features = kwargs.get("include_features", [])

    def preprocess_and_save(self, output_path):
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        warned_sensors = set()
        warned_features = set()
        sample_idx = 0
        with h5py.File(self.features_path, "r") as hdf5_file:
            with h5py.File(output_path, "w") as new_hdf5_file:
                for subject in hdf5_file.keys():
                    subject_id = int(subject.split("_")[1])

                    if self.exclude_subjects and subject_id in self.exclude_subjects:
                        continue

                    if self.include_subjects and subject_id not in self.include_subjects:
                        continue

                    for aug in hdf5_file[subject].keys():
                        is_augmented = aug.split("_")[1] == "True"

                        if not self.include_augmented and is_augmented:
                            continue

                        for label in hdf5_file[subject][aug].keys():
                            if label not in self.labels:
                                continue

                            for b, batch in enumerate(
                                hdf5_file[subject][aug][label].keys()
                            ):
                                batch_group = new_hdf5_file.require_group(
                                    str(sample_idx)
                                )
                                for sensor in hdf5_file[subject][aug][label][
                                    batch
                                ].keys():
                                    if sensor not in self.include_sensors:
                                        if sensor not in warned_sensors:
                                            warned_sensors.add(sensor)
                                            print(
                                                f"Sensor '{sensor}' is not being used."
                                            )
                                        continue
                                    sensor_group = batch_group.require_group(sensor)
                                    data = []
                                    for feature in hdf5_file[subject][aug][label][
                                        batch
                                    ][sensor].keys():
                                        if feature not in self.include_features:
                                            if feature not in warned_features:
                                                warned_features.add(feature)
                                                print(
                                                    f"Feature: {feature} for sensor {sensor} not in include_features list"
                                                )
                                            continue
                                        data.append(
                                            hdf5_file[subject][aug][label][batch][
                                                sensor
                                            ][feature][:]
                                        )

                                    # Save the preprocessed sample
                                    sensor_group.create_dataset(
                                        f"data", data=np.array(data)
                                    )
                                    sensor_group.create_dataset(
                                        f"label", data=float(label)
                                    )
                                sample_idx += 1
                            print(
                                f"Processed batches for subject {subject_id} and label {label} and aug {aug}"
                            )

    def preprocess_and_save_cross_validation(self, n_splits, save_paths_template):
        # Check if the save paths template is valid
        if not isinstance(save_paths_template, str):
            raise ValueError(
                "Save paths template should be a string with placeholders for split index and type."
            )

        # Initialize datasets
        all_data = []

        warned_sensors = set()
        warned_features = set()

        with h5py.File(self.features_path, "r") as hdf5_file:
            for subject in hdf5_file.keys():
                subject_id = int(subject.split("_")[1])

                if self.exclude_subjects and subject_id in self.exclude_subjects:
                    continue

                if self.include_subjects and subject_id not in self.include_subjects:
                    continue

                for aug in hdf5_file[subject].keys():
                    is_augmented = aug.split("_")[1] == "True"

                    for label in hdf5_file[subject][aug].keys():
                        if label not in self.labels:
                            continue

                        for b, batch in enumerate(
                            hdf5_file[subject][aug][label].keys()
                        ):
                            sample = {"is_augmented": is_augmented}
                            for sensor in hdf5_file[subject][aug][label][batch].keys():
                                if sensor not in self.include_sensors:
                                    if sensor not in warned_sensors:
                                        warned_sensors.add(sensor)
                                        print(f"Sensor '{sensor}' is not being used.")
                                    continue
                                data = []
                                for feature in hdf5_file[subject][aug][label][batch][
                                    sensor
                                ].keys():
                                    if feature not in self.include_features:
                                        if feature not in warned_features:
                                            warned_features.add(feature)
                                            print(
                                                f"Feature: {feature} for sensor {sensor} not in include_features list"
                                            )
                                        continue
                                    data.append(
                                        hdf5_file[subject][aug][label][batch][sensor][
                                            feature
                                        ][:]
                                    )

                                # Save the preprocessed sample
                                sample[sensor] = {
                                    "data": np.array(data),
                                    "label": float(label),
                                }
                            all_data.append(sample)
                        print(
                            f"Processed batches for subject {subject_id} and label {label} and aug {aug}"
                        )

        # Perform K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42069)
        cross_validation_paths = []

        for fold_index, (train_index, val_index) in enumerate(kf.split(all_data)):
            train_data = [all_data[i] for i in train_index]
            val_data = [all_data[i] for i in val_index]

            # Filter the augmented data for validation set
            val_data = [sample for sample in val_data if not sample["is_augmented"]]

            save_paths = {
                "train": save_paths_template.format(split=fold_index, type="train"),
                "val": save_paths_template.format(split=fold_index, type="val"),
            }

            for path in save_paths.values():
                directory = os.path.dirname(path)
                if not os.path.exists(directory):
                    os.makedirs(directory)

            # Save data to hdf5 files
            for name, dataset in zip(["train", "val"], [train_data, val_data]):
                with h5py.File(save_paths[name], "w") as new_hdf5_file:
                    for sample_idx, sample in enumerate(dataset):
                        batch_group = new_hdf5_file.require_group(str(sample_idx))
                        for sensor, sensor_data in sample.items():
                            if sensor == "is_augmented":
                                continue
                            sensor_group = batch_group.require_group(sensor)
                            sensor_group.create_dataset(
                                "data", data=sensor_data["data"]
                            )
                            sensor_group.create_dataset(
                                "label", data=sensor_data["label"]
                            )
                    print(f"Saved {name} data to {save_paths[name]}")

            cross_validation_paths.append(save_paths)

        return cross_validation_paths

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pass


class SensorDataset(Dataset):
    def __init__(self, features_path, include_sensors, group_labels=None):
        self.features_path = features_path
        self.include_sensors = include_sensors
        self.group_labels = group_labels
        with h5py.File(self.features_path, "r") as hdf5_file:
            self.data_keys = list(hdf5_file.keys())
        self.data_keys = sorted(self.data_keys, key=int)
        self.dataset_length = len(self.data_keys)

    def get_dims(self):
        with h5py.File(self.features_path, "r") as hdf5_file:
            for key in hdf5_file.keys():
                data_dict = {}
                for sensor in self.include_sensors:
                    data_dict[sensor] = torch.tensor(
                        hdf5_file[key][sensor]["data"][:], dtype=torch.float32
                    )
                break

        return {sensor: data_dict[sensor].shape[0] for sensor in self.include_sensors}

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        with h5py.File(self.features_path, "r") as hdf5_file:
            sample_key = self.data_keys[idx]
            data_dict = {}
            for sensor in self.include_sensors:
                data_dict[sensor] = torch.tensor(
                    hdf5_file[sample_key][sensor][f"data"][:], dtype=torch.float32
                )
            label = torch.tensor(
                int(hdf5_file[sample_key][sensor][f"label"][()]), dtype=torch.long
            )

        # Merge labels if necessary
        if self.group_labels is not None:
            for group, labels in self.group_labels.items():
                if label.item() in labels:
                    label = torch.tensor(group, dtype=torch.long)
                    break

        return data_dict, label