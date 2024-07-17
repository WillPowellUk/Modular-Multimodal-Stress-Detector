import concurrent.futures
from multiprocessing import cpu_count
import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import gc
import h5py
from sklearn.preprocessing import StandardScaler

from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key, print2, load_json
from .eda_feature_extractor import EDAFeatureExtractor
from .bvp_feature_extractor import BVPFeatureExtractor
from .acc_feature_extractor import AccFeatureExtractor
from .ecg_feature_extractor import ECGFeatureExtractor
from .emg_feature_extractor import EMGFeatureExtractor
from .resp_feature_extractor import RespFeatureExtractor
from .temp_feature_extractor import TempFeatureExtractor

LOG_FILE_PATH = "manual_fe.log"


class ManualFE:
    def __init__(self, batches, save_path: str, config_path: str):
        self.batches = batches
        self.save_path = save_path
        self.config = load_json(config_path)
        self.features = get_active_key(config_path, "features", recursive=True)
        self.sensors = get_active_key(config_path, "sensors")
        self.sampling_rate = get_max_sampling_rate(config_path)

        # if LOG_FILE_PATH exists, delete it
        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)

        # Ignore runtime warning for mean of empty slice
        warnings.filterwarnings("ignore", message="Mean of empty slice")

    def extract_features_from_batch(self, batch):
        features_dict = {}

        for sensor in self.sensors:
            match sensor:
                case "resp":
                    resp_features = RespFeatureExtractor(
                        batch["resp"], self.sampling_rate
                    ).extract_features()
                    features_dict["resp"] = resp_features

        return features_dict

    def extract_features_from_split(self, split):
        features_dict = {}
        features_dict["sid"] = split["sid"].iloc[0]
        features_dict["is_augmented"] = split["is_augmented"].iloc[0]

        for sensor in self.sensors:
            match sensor:
                case "w_eda":
                    eda_features = EDAFeatureExtractor(
                        split["w_eda"], self.sampling_rate
                    ).extract_features()
                    features_dict["w_eda"] = eda_features
                case "bvp":
                    bvp_features = BVPFeatureExtractor(
                        split["bvp"], self.sampling_rate
                    ).extract_features()
                    features_dict["bvp"] = bvp_features
                case "w_temp":
                    temp_features = TempFeatureExtractor(
                        split["w_temp"], self.sampling_rate
                    ).extract_features()
                    features_dict["w_temp"] = temp_features
                case "eda":
                    eda_features = EDAFeatureExtractor(
                        split["eda"], self.sampling_rate
                    ).extract_features()
                    features_dict["eda"] = eda_features
                case "ecg":
                    ecg_features = ECGFeatureExtractor(
                        split["ecg"], self.sampling_rate
                    ).extract_features()
                    features_dict["ecg"] = ecg_features
                case "emg":
                    emg_features = EMGFeatureExtractor(
                        split["emg"], self.sampling_rate
                    ).extract_features()
                    features_dict["emg"] = emg_features
                case "resp":
                    if self.slow_feautres_flag:
                        continue
                    resp_features = RespFeatureExtractor(
                        split["resp"], self.sampling_rate
                    ).extract_features()
                    features_dict["resp"] = resp_features
                case "temp":
                    temp_features = TempFeatureExtractor(
                        split["temp"], self.sampling_rate
                    ).extract_features()
                    features_dict["temp"] = temp_features
                case "w_acc":
                    acc_df = pd.DataFrame(
                        {
                            "x": split["w_acc_x"],
                            "y": split["w_acc_y"],
                            "z": split["w_acc_z"],
                        }
                    )
                    acc_features = AccFeatureExtractor(
                        acc_df, self.sampling_rate
                    ).extract_features()
                    features_dict["w_acc"] = acc_features
                case "acc":
                    acc_df = pd.DataFrame(
                        {"x": split["acc1"], "y": split["acc2"], "z": split["acc3"]}
                    )
                    acc_features = AccFeatureExtractor(
                        acc_df, self.sampling_rate
                    ).extract_features()
                    features_dict["acc"] = acc_features

        features_dict["label"] = split["label"].iloc[0]

        return features_dict

    def save_to_hdf5(self, all_batches_features):
        print2(LOG_FILE_PATH, f"Saving features to {self.save_path}...")
        with h5py.File(self.save_path, "w") as hdf5_file:
            for b, batch in enumerate(all_batches_features):
                # Extract the details of the first element
                first_element = batch[0]
                expected_sid = first_element["sid"]
                expected_is_augmented = first_element["is_augmented"]
                expected_label = first_element["label"]

                # Check if all elements have the same sid, is_augmented, and label
                for element in batch:
                    if (
                        element["sid"] != expected_sid
                        or element["is_augmented"] != expected_is_augmented
                        or element["label"] != expected_label
                    ):
                        print2(
                            LOG_FILE_PATH,
                            "Not all elements in the batch have the same 'sid', 'is_augmented', and 'label'",
                        )
                        continue

                # Extract and format details of the first element
                sid = str(int(batch[0]["sid"]))
                is_augmented = (
                    "augmented_True" if batch[0]["is_augmented"] else "augmented_False"
                )
                label = str(batch[0]["label"])

                # Remove 'sid', 'is_augmented', and 'label' from all elements in the batch
                for element in batch:
                    element.pop("sid")
                    element.pop("is_augmented")
                    element.pop("label")

                subject_group = hdf5_file.require_group(f"subject_{sid}")
                augmented_group = subject_group.require_group(is_augmented)
                label_group = augmented_group.require_group(label)
                batch_group = label_group.require_group(str(b))

                for i, split in enumerate(batch):
                    for j, (sensor_name, feature_data) in enumerate(split.items()):
                        sensor_group = batch_group.require_group(sensor_name)

                        if isinstance(feature_data, pd.DataFrame):
                            for k, column in enumerate(feature_data.columns):
                                if column in sensor_group:
                                    dataset = sensor_group[column]
                                    dataset.resize(
                                        (dataset.shape[0] + feature_data.shape[0],)
                                    )
                                    dataset[-feature_data.shape[0] :] = feature_data[
                                        column
                                    ].values
                                else:
                                    sensor_group.create_dataset(
                                        column,
                                        data=feature_data[column].values,
                                        maxshape=(None,),
                                    )
                        else:
                            raise ValueError(
                                f"Unknown feature data type: {type(feature_data)}"
                            )
        print2(LOG_FILE_PATH, "Features saved successfully")

    def impute_and_normalize_features(self, all_batches_features):
        print2(LOG_FILE_PATH, "Scaling and imputing missing values...")

        # Collecting all features for scaling and imputation
        feature_data = {}
        for batch_idx, batch in enumerate(all_batches_features):
            print2(
                LOG_FILE_PATH,
                f"Processing batch {batch_idx + 1} of {len(all_batches_features)}...",
            )
            for minibatch_idx, minibatch in enumerate(batch):
                print2(
                    LOG_FILE_PATH,
                    f"Processing minibatch {minibatch_idx + 1} of {len(batch)}...",
                )
                for key, value in minibatch.items():
                    if key in ["sid", "label", "is_augmented"]:
                        continue
                    if key not in feature_data:
                        feature_data[key] = []
                    feature_data[key].append(value)

        scaler = StandardScaler()
        for key, data_list in feature_data.items():
            print2(LOG_FILE_PATH, f"Scaling and imputing feature: {key}...")
            if all(isinstance(d, pd.DataFrame) for d in data_list):
                combined_df = pd.concat(data_list)

                combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Normalize and impute features in place
                normalized_df = pd.DataFrame(
                    scaler.fit_transform(combined_df), columns=combined_df.columns
                )
                mean_values = normalized_df.mean()
                mean_values.fillna(0, inplace=True)

                del combined_df  # Free up memory
                gc.collect()  # Explicit garbage collection

                start_idx = 0
                for i, df in enumerate(data_list):
                    end_idx = start_idx + len(df)
                    data_list[i] = normalized_df.iloc[start_idx:end_idx].copy()
                    data_list[i].fillna(mean_values, inplace=True)
                    start_idx = end_idx
                    print2(
                        LOG_FILE_PATH,
                        f"Processed segment {i + 1} of {len(data_list)} for feature: {key}",
                    )

                del normalized_df  # Free up memory
                gc.collect()  # Explicit garbage collection
            else:
                raise ValueError(f"Unknown feature data type in {data_list}")

        # Replace scaled and imputed features back into all_batches_features
        for batch_idx, batch in enumerate(all_batches_features):
            print2(
                LOG_FILE_PATH,
                f"Replacing scaled features in batch {batch_idx + 1} of {len(all_batches_features)}...",
            )
            for minibatch_idx, minibatch in enumerate(batch):
                print2(
                    LOG_FILE_PATH,
                    f"Replacing scaled features in minibatch {minibatch_idx + 1} of {len(batch)}...",
                )
                for key in minibatch.keys():
                    if key in ["sid", "label", "is_augmented"]:
                        continue
                    minibatch[key] = feature_data[key].pop(0)

        print2(LOG_FILE_PATH, "Scaling and imputation complete")
        return all_batches_features
    
    def impute_missing_features(self):
        features_dict = {}

        for sensor in self.sensors:
            match sensor:
                case "resp":
                    features = self.config['features']['resp']
                    # Create a dictionary with features as keys and np.nan as values
                    sensor_features = {feature: np.nan for feature in features}
                    features_dict[sensor] = sensor_features

        return features_dict

    def extract_features(self):
        print(f"Extracting Features. Writing to log file: {LOG_FILE_PATH}...")
        print2(LOG_FILE_PATH, "Extracting features...")
        warnings.warn_explicit = warnings.warn = lambda *_, **__: None
        warnings.filterwarnings("ignore")

        all_batches_features = []
        total_batches = len(self.batches)
        start_time = time.time()

        slow_buffer = pd.DataFrame()
        for i, batch in enumerate(self.batches):
            try:
                if i % 100 == 0 and i != 0:
                    break
                    elapsed_time = time.time() - start_time
                    average_time_per_batch = elapsed_time / (i + 1)
                    remaining_batches = total_batches - (i + 1)
                    eta = average_time_per_batch * remaining_batches
                    hours = math.floor(eta / 3600)
                    minutes = math.floor((eta % 3600) / 60)
                    seconds = eta % 60

                    # print2 LOG_FILE_PATH, the formatted string
                    print2(
                        LOG_FILE_PATH,
                        f"Extracting features from batch {i+1}/{total_batches} | ETA: {hours}h {minutes}m {seconds:.2f}s",
                    )

                slow_features_length_s = 40 # 10s 

                # Complete slow signals such as resp from batch and copy features to each split
                if len(batch[0]) / self.sampling_rate < slow_features_length_s:
                    self.slow_feautres_flag = True
                    # Concatenate DataFrames together
                    batch_concat = pd.concat(batch, axis=0)
                    batch_concat.reset_index(drop=True, inplace=True)

                    # Fill up buffer
                    slow_buffer = pd.concat([slow_buffer, batch_concat], axis=0)

                    # If buffer is not full, impute missing values
                    if len(slow_buffer) / self.sampling_rate < slow_features_length_s:
                        # Impute missing contents
                        slow_features = self.impute_missing_features()

                    # If buffer is full, extract features from buffer and move buffer forward
                    else:
                        # Extract features
                        slow_features = self.extract_features_from_batch(slow_buffer)

                        # Calculate the number of samples to remove to maintain the buffer size
                        max_buffer_length = slow_features_length_s * self.sampling_rate
                        excess_samples = len(slow_buffer) - max_buffer_length

                        # Remove the oldest data points
                        if excess_samples > 0:
                            slow_buffer = slow_buffer.iloc[excess_samples:].reset_index(drop=True)

                else:
                    self.slow_feautres_flag = False

                batch_features = []
                for split in batch:
                    split_features = self.extract_features_from_split(split)
                    if self.slow_feautres_flag:
                        split_features.update(slow_features)
                    batch_features.append(split_features)
                all_batches_features.append(batch_features)
            except Exception as e:
                print2(LOG_FILE_PATH, f"Error processing batch {i}. Error: {e}")

        all_batches_features = self.impute_and_normalize_features(all_batches_features)

        # Ensure the directory exists
        dir_name = os.path.dirname(self.save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Save the features to HDF5
        self.save_to_hdf5(all_batches_features)

    def extract_features_concurrently(self):
        warnings.warn_explicit = warnings.warn = lambda *_, **__: None
        warnings.filterwarnings("ignore")

        all_batches_features = []
        total_batches = len(self.batches)
        start_time = time.time()

        def process_batch(i, batch):
            batch_features = []
            for split in batch:
                split_features = self.extract_features_from_split(split)
                batch_features.append(split_features)
            return i, batch_features

        # Limit to the first 201 batches for testing
        test_batches = self.batches[:201]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_batch = {
                executor.submit(process_batch, i, batch): i
                for i, batch in enumerate(test_batches)
            }

            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_batch)
            ):
                index, batch_features = future.result()
                all_batches_features.append((index, batch_features))

                if i % 100 == 0 and i != 0:
                    elapsed_time = time.time() - start_time
                    average_time_per_batch = elapsed_time / (i + 1)
                    remaining_batches = len(test_batches) - (i + 1)
                    eta = average_time_per_batch * remaining_batches
                    hours = math.floor(eta / 3600)
                    minutes = math.floor((eta % 3600) / 60)
                    seconds = eta % 60

                    # print2 LOG_FILE_PATH, the formatted string
                    print2(
                        LOG_FILE_PATH,
                        f"Extracting features from batch {i+1}/{len(test_batches)} | ETA: {hours}h {minutes}m {seconds:.2f}s",
                    )

        all_batches_features.sort(key=lambda x: x[0])  # Sort by original index
        all_batches_features = [
            batch[1] for batch in all_batches_features
        ]  # Remove index

        all_batches_features = self.impute_and_normalize_features(all_batches_features)

        # Ensure the directory exists
        dir_name = os.path.dirname(self.save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Save the features to HDF5
        self.save_to_hdf5(all_batches_features)

    @staticmethod
    def process_batch(i, batch, extractor):
        batch_start_time = time.time()
        batch_features = []
        for split in batch:
            split_features = extractor.extract_features_from_split(split)
            batch_features.append(split_features)
        batch_end_time = time.time()
        print2(
            LOG_FILE_PATH,
            f"Batch {i} processed in {batch_end_time - batch_start_time:.2f} seconds",
        )
        return i, batch_features

    def extract_features_parallel(self):
        warnings.warn_explicit = warnings.warn = lambda *_, **__: None
        warnings.filterwarnings("ignore")

        all_batches_features = []
        total_batches = len(self.batches)
        start_time = time.time()

        # Limit to the first 201 batches for testing
        test_batches = self.batches[:201]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_count()
        ) as executor:
            future_to_batch = {
                executor.submit(ManualFE.process_batch, i, batch, self): i
                for i, batch in enumerate(test_batches)
            }

            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_batch)
            ):
                index, batch_features = future.result()
                all_batches_features.append((index, batch_features))

                if i % 100 == 0 and i != 0:
                    elapsed_time = time.time() - start_time
                    average_time_per_batch = elapsed_time / (i + 1)
                    remaining_batches = len(test_batches) - (i + 1)
                    eta = average_time_per_batch * remaining_batches
                    hours = math.floor(eta / 3600)
                    minutes = math.floor((eta % 3600) / 60)
                    seconds = eta % 60

                    # print2 LOG_FILE_PATH, the formatted string
                    print2(
                        LOG_FILE_PATH,
                        f"Extracting features from batch {i+1}/{len(test_batches)} | ETA: {hours}h {minutes}m {seconds:.2f}s",
                    )

        all_batches_features.sort(key=lambda x: x[0])  # Sort by original index
        all_batches_features = [
            batch[1] for batch in all_batches_features
        ]  # Remove index

        all_batches_features = self.impute_and_normalize_features(all_batches_features)

        # Ensure the directory exists
        dir_name = os.path.dirname(self.save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Save the features to HDF5
        self.save_to_hdf5(all_batches_features)
        total_end_time = time.time()
        print2(
            LOG_FILE_PATH,
            f"Total time taken: {total_end_time - start_time:.2f} seconds",
        )
