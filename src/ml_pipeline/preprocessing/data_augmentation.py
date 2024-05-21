import pandas as pd
import numpy as np
import pickle
import os
import re
import json

class DataAugmentation:
    def __init__(self, data_path: str, output_path: str, config_path: str, window_size=60, sliding_length=5):
        self.df = pd.read_pickle(data_path)
        self.output_path = output_path
        self.window_size = window_size
        self.sliding_length = sliding_length
        
        # Load the JSON data from the file
        with open(config_path, 'r') as file:
            self.sampling_rates = json.load(file)
        
        # Create a list to store the paths of the individual segmented files
        self.segment_files = []

    def segment(self):
        print("Starting segmentation process...")
        acc_columns = [col for col in self.df.columns if re.search(r'acc', col)]
        for key in acc_columns:
            print(f"Segmenting {key}...")
            segmented_windows = self._segment_signal(self.df, key, self.sampling_rates['acc'])
            self.save_segments(key, segmented_windows)
            print(f"Segmentation of {key} completed.")
            
        for key in self.sampling_rates:
            if key == 'ecg' or key == 'emg' or key == 'temp' or key == 'eda':
                continue
            
            if key in self.df.columns:
                print(f"Segmenting {key}...")
                segmented_windows = self._segment_signal(self.df, key, self.sampling_rates[key])
                self.save_segments(key, segmented_windows)
                print(f"Segmentation of {key} completed.")
        print("Segmentation process completed.")

    def _segment_signal(self, df, column_name, sampling_rate):
        window_size_samples = int(self.window_size * sampling_rate)
        sliding_length_samples = int(self.sliding_length * sampling_rate)
    
        windows = []
        total_segments = (len(df) - window_size_samples) // sliding_length_samples + 1
        print(f"Total segments to be created for {column_name}: {total_segments}")
        
        for i in range(0, len(df) - window_size_samples + 1, sliding_length_samples):
            window = df.iloc[i:i + window_size_samples].copy()
            if len(window) == window_size_samples:
                sid = window['sid'].iloc[0] if 'sid' in df.columns else None
                label = window['label'].iloc[0] if 'label' in df.columns else None
                window = window[[column_name]]  # Keep only the relevant signal column
                if sid is not None:
                    window['sid'] = sid
                if label is not None:
                    window['label'] = label
                windows.append(window)
                
            if len(windows) % 200 == 0:  # Print progress every 200 windows
                print(f"Processed {len(windows)} / {total_segments} segments for {column_name}")
        
        return windows

    def save_segments(self, key, windows):
        print(f"Saving segmented windows for {key}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        output_file_path = self.output_path.replace('.pkl', f'_{key}_windows.pkl')
        with open(output_file_path, 'wb') as f:
            pickle.dump(windows, f)
        print(f"Segmented windows for {key} saved to {output_file_path}")
        self.segment_files.append(output_file_path)

    def merge_segments(self):
        print("Merging segmented windows...")
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_ecg_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_eda_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_emg_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_acc1_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_acc2_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_acc3_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_temp_windows.pkl')
        self.segment_files.append('src/wesad/WESAD/augmented/chest_augmented_resp_windows.pkl')

        # Function to read DataFrames from each file in batches
        def read_dataframes_in_batches(file_paths, batch_size=10):
            for file_path in file_paths:
                print(f"Processing file: {file_path}")
                with open(file_path, 'rb') as f:
                    dataframes = pickle.load(f)
                    for start in range(0, len(dataframes), batch_size):
                        yield dataframes[start:start + batch_size]

        # Open the output file in write mode and initialize the pickle writer
        with open(self.output_path, 'wb') as f:
            first_batch = True
            total_dfs = 0

            for batch in read_dataframes_in_batches(self.segment_files):
                if first_batch:
                    # For the first batch, we need to start the pickle
                    pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
                    first_batch = False
                else:
                    # For subsequent batches, we append to the existing pickle
                    for df in batch:
                        f.write(pickle.dumps([df], protocol=pickle.HIGHEST_PROTOCOL)[2:])
                total_dfs += len(batch)
                print(f"Total DataFrames merged: {total_dfs}")

        print(f"All segmented windows merged and saved to {self.output_path}")


if __name__ == '__main__':
    print("Augmentation for chest data started.")
    augmenter = DataAugmentation('src/wesad/WESAD/cleaned/chest_preprocessed.pkl', 'src/wesad/WESAD/augmented/chest_augmented.pkl', 'src/wesad/wesad_configuration.json', window_size=60, sliding_length=5)
    # augmenter.segment()
    augmenter.merge_segments()
    print("Augmentation for chest data completed.")
    
    # print("Augmentation for wrist data started.")
    # augmenter = DataAugmentation('src/wesad/WESAD/cleaned/wrist_preprocessed.pkl', 'src/wesad/WESAD/augmented/wrist_augmented.pkl', 'src/wesad/wesad_configuration.json', window_size=60, sliding_length=5)
    # augmenter.segment()
    # augmenter.merge_segments()
    # print("Augmentation for wrist data completed.")
