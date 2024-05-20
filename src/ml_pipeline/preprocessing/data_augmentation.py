import pandas as pd
import numpy as np
import pickle
import os
import json

class DataAugmentation:
    def __init__(self, data_path: str, output_dir: str, config_path: str, window_size=60, sliding_length=5):
        self.df = pd.read_pickle(data_path)
        self.output_dir = output_dir
        self.window_size = window_size
        self.sliding_length = sliding_length
        
        # Load the JSON data from the file
        with open(config_path, 'r') as file:
            self.sampling_rates = json.load(file)

    def segment(self):
        windows = {}
        
        for key in self.sampling_rates:
            if key in self.df.columns:
                print(f"Segmenting {key}...")
                windows[key] = self._segment_signal(self.df, key, self.sampling_rates[key])
                print(f"Segmentation of {key} completed.")

        self.save_segments(windows, self.output_dir)
        
        return windows

    def _segment_signal(self, df, column_name, sampling_rate):
        window_size_samples = int(self.window_size * sampling_rate)
        sliding_length_samples = int(self.sliding_length * sampling_rate)
    
        windows = []
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
        
        return windows

    def save_segments(self, segmented_windows, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        for key, windows in segmented_windows.items():
            output_path = os.path.join(output_dir, f"segmented_{key}_windows.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(windows, f)
            print(f"Segmented windows for {key} saved to {output_path}")