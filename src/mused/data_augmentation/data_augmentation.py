from math import floor
import numpy as np
import pandas as pd
import pickle
import os

from src.ml_pipeline.utils.utils import get_sampling_frequency

class DataAugmenter:
    def __init__(self, pkl_file, config_path, pkl_output_path):
        self.config_path = config_path  
        self.pkl_output_path = pkl_output_path
        with open(pkl_file, "rb") as file:
            self.pkl_file = pickle.load(file)

    def augment_data(self, window_size=60, sliding_length=5):
        print("Segmenting data...")

        # Iterate through each source and sensor
        for source in self.pkl_file:
            for sensor in self.pkl_file[source]:
                # Get the dataframe for the current source and sensor
                df = self.pkl_file[source][sensor]
                df['is_augmented'] = False

                # Get the sampling frequency for this source and sensor
                sampling_frequency = get_sampling_frequency(self.config_path, source, sensor)

                # Calculate the number of samples per window and sliding step
                window_step = window_size * sampling_frequency
                sliding_step = sliding_length * sampling_frequency

                # Number of iterations to create synthetic samples
                num_of_iterations = floor(window_size / sliding_length)

                # Initialize a list to hold all segments (both original and augmented)
                segments = []

                # Initial indices for window slicing
                start_idx = 0
                end_idx = window_step

                # Iterating through the data to create non-augmented and augmented segments
                for i in range(num_of_iterations):
                    while end_idx <= len(df):
                        # Extract a segment of the data
                        segment = df.iloc[start_idx:end_idx].copy()
                        
                        # Check if segment has a consistent label
                        if segment["Label"].nunique() == 1:
                            # Determine if the segment is augmented or not
                            segment.loc[:, "is_augmented"] = (
                                False if start_idx % window_step == 0 else True
                            )

                            # Only add segments of the correct length
                            if len(segment) == window_step:
                                segments.append(segment)

                        # Move window forward
                        start_idx += window_step
                        end_idx += window_step
                    
                    # Adjust start and end indices for the next iteration
                    start_idx = (i + 1) * sliding_step
                    end_idx = start_idx + window_step

                # Print information about segmentation
                print(f"Source: {source}, Sensor: {sensor}")
                # print(f"Number of segments: {len(segments)}")
                # print(f"Synthetic to non-synthetic ratio: {num_of_iterations-1}:{1}")

                # Combine all segments into a single DataFrame
                combined_df = pd.concat(segments, ignore_index=True)

                # Append the combined dataframe back to the original pkl_file structure
                self.pkl_file[source][sensor] = pd.concat([self.pkl_file[source][sensor], combined_df], ignore_index=True)

                # Save the augmented data to a new pickle file
                with open(self.pkl_output_path, "wb") as file:
                    pickle.dump(self.pkl_file, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Data augmentation complete. Saved to {self.pkl_output_path}.")


if __name__ == "__main__":
    for subject_id in range(1, 19):
        print(f"Subject {subject_id}")
        pkl_path = f"src/mused/dataset/S{subject_id}/S{subject_id}_cleaned.pkl"
        pkl_output_path = f"src/mused/dataset/S{subject_id}/S{subject_id}_augmented.pkl"

        config_path = "config_files/dataset/mused_configuration.json"
        data_augmenter = DataAugmenter(pkl_path, config_path, pkl_output_path)

        WINDOW_LENGTH = 30
        SPLIT_LENGTH = int(WINDOW_LENGTH / 6) # this will sub-split the data 6 times each of 5 seconds
        SLIDING_LENGTH = SPLIT_LENGTH # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples

        segments = data_augmenter.augment_data(WINDOW_LENGTH, SLIDING_LENGTH)