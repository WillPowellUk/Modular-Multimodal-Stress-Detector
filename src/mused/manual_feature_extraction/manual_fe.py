import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import gc
import h5py
from sklearn.preprocessing import StandardScaler
import pickle
from src.ml_pipeline.utils.utils import (
    get_sampling_frequency,
    get_enabled_sensors,
    print2,
    load_json,
)
from src.ml_pipeline.feature_extraction.manual.eda_feature_extractor import EDAFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.bvp_feature_extractor import BVPFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.acc_feature_extractor import AccFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.ecg_feature_extractor import ECGFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.ibi_feature_extractor import IBIFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.emg_feature_extractor import EMGFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.resp_feature_extractor import RespFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.temp_feature_extractor import TempFeatureExtractor
from src.ml_pipeline.feature_extraction.manual.fnirs_feature_extractor import FNIRSFeatureExtractor


class ManualFeatureExtractor:
    def __init__(self, augmented_pkl_path, config_path, output_pkl_path):
        self.augmented_pkl_path = augmented_pkl_path
        self.config_path = config_path
        self.output_pkl_path = output_pkl_path
        self.data = pd.read_pickle(self.augmented_pkl_path)
        self.config = load_json(self.config_path)
        self.log_file = "mused_manual_feature_extraction.log"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def extract_features(self, split_length):
        print2(self.log_file, "Starting feature extraction...")

        # Mapping between sources, sensors, and their corresponding feature extractors
        feature_extractor_mapping = {
            'quattrocento': {
                'emg_upper_trapezius': EMGFeatureExtractor,
                'emg_mastoid': EMGFeatureExtractor,
            },
            'polar': {
                'acc': AccFeatureExtractor,
                'ecg': ECGFeatureExtractor,
                'ibi': IBIFeatureExtractor,
            },
            'empatica': {
                'acc': AccFeatureExtractor,
                'bvp': BVPFeatureExtractor,
                'temp': TempFeatureExtractor,
                'eda': EDAFeatureExtractor,
            },
            'myndsens': {
                'fnirs': FNIRSFeatureExtractor,
            }
        }

        # Iterate through each source and sensor
        for source in self.data:
            print2(self.log_file, f"Processing source: {source}...")

            for sensor in self.data[source]:
                print2(self.log_file, f"Processing sensor: {sensor}...")

                # Get the dataframe for the current source and sensor
                df = self.data[source][sensor]

                # Get the sampling frequency for this source and sensor
                sampling_frequency = get_sampling_frequency(self.config_path, source, sensor)

                if sampling_frequency == 130 or sampling_frequency == 2048 or sampling_frequency == 1:
                    continue

                # Calculate the number of samples per split based on the sampling frequency
                samples_per_split = split_length * sampling_frequency

                # Determine the appropriate feature extractor for this source and sensor
                feature_extractor_class = feature_extractor_mapping.get(source, {}).get(sensor)

                if feature_extractor_class is None:
                    print2(self.log_file, f"No feature extractor available for {source} - {sensor}. Skipping.")
                    continue

                # Initialize a list to store feature dataframes
                features_list = []

                # Split data into segments based on samples_per_split
                for i, start_idx in enumerate(range(0, len(df), samples_per_split)):
                    end_idx = start_idx + samples_per_split
                    segment = df.iloc[start_idx:end_idx].copy()

                    # Ensure segment length is equal to samples_per_split
                    if len(segment) != samples_per_split:
                        print2(self.log_file, f"Skipping incomplete segment at index {start_idx} for {sensor}.")
                        continue

                    # Initialize the feature extractor with the segment and sampling rate
                    feature_extractor = feature_extractor_class(segment.drop(columns=['Label', 'is_augmented']), sampling_rate=sampling_frequency)

                    # Extract features for the segment
                    try:
                        extracted_features = feature_extractor.extract_features()
                    except Exception as e:
                        print2(self.log_file, f"Error extracting features for segment {start_idx} - {end_idx} for {sensor}: {e}")
                        continue
                    if extracted_features is None:
                        print2(self.log_file, f"Skipping segment {start_idx} - {end_idx} for {sensor}.")
                        continue

                    # Preserve supplementary columns
                    extracted_features["Label"] = segment["Label"].iloc[0]  # Assumes label is consistent in segment
                    extracted_features["is_augmented"] = segment["is_augmented"].iloc[0]

                    # Append the extracted features to the list
                    features_list.append(extracted_features)

                    # Log progress
                    if i % 100 == 0:
                        print2(self.log_file, f"Extracted features {start_idx/len(df)*100}% for {sensor}.")

                # Combine all feature dataframes for the current sensor
                if features_list:
                    combined_features_df = pd.concat(features_list, ignore_index=True)

                    # Impute missing values with the mean of each column
                    combined_features_df.fillna(combined_features_df.mean(), inplace=True)

                    # Remove columns where the mean is NaN
                    means = combined_features_df.mean()
                    columns_to_drop = means[means.isna()].index
                    if len(columns_to_drop) > 0:
                        combined_features_df.drop(columns=columns_to_drop, inplace=True)
                        print2(self.log_file, f"Dropped columns with NaN mean for sensor: {sensor} - {list(columns_to_drop)}")

                    self.data[source][sensor] = combined_features_df
                    print2(self.log_file, f"Combined and imputed features for sensor: {sensor}.")

                else:
                    print2(self.log_file, f"No features extracted for sensor: {sensor}.")

        # Save the extracted features to the output pickle file
        with open(self.output_pkl_path, "wb") as file:
            pickle.dump(self.data, file, protocol=pickle.HIGHEST_PROTOCOL)

        print2(self.log_file, f"Feature extraction complete. Features saved to {self.output_pkl_path}.")


def main():
    for subject_id in range(1,2):
        augmented_pkl_path = f"src/mused/dataset/S{subject_id}/S{subject_id}_augmented.pkl"
        config_path = "config_files/dataset/mused_configuration.json"
        output_pkl_path = f"src/mused/dataset/S{subject_id}/S{subject_id}_features.pkl"

        feature_extractor = ManualFeatureExtractor(
            augmented_pkl_path, config_path, output_pkl_path
        )

        WINDOW_LENGTH = 30
        SPLIT_LENGTH = int(WINDOW_LENGTH / 6) # this will sub-split the data 6 times each of 5 seconds

        feature_extractor.extract_features(WINDOW_LENGTH)

if __name__ == "__main__":
    main()