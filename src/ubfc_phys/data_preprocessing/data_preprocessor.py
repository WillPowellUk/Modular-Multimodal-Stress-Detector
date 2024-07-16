import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle

class UBFCPhysDataPreprocessor:
    def __init__(self, directory):
        self.directory = directory
        self.data = pd.DataFrame()

    def load_data(self):
        for sid in range(1, 57):
            for task in ['T1', 'T2', 'T3']:
                bvp_file = os.path.join(self.directory, f"bvp_s{sid}_{task}.csv")
                eda_file = os.path.join(self.directory, f"eda_s{sid}_{task}.csv")
                
                if os.path.exists(bvp_file) and os.path.exists(eda_file):
                    bvp_data = pd.read_csv(bvp_file, header=None)
                    eda_data = pd.read_csv(eda_file, header=None)
                    
                    # Interpolate EDA signal to 64 Hz
                    eda_interpolated = self.interpolate_signal(eda_data[0], original_freq=4, target_freq=64)
                    
                    # Create the combined dataframe for this participant and task
                    combined_df = pd.DataFrame({
                        'sid': float(sid),
                        'bvp': bvp_data[0],
                        'eda': eda_interpolated,
                        'label': float(task[1])
                    })
                    
                    # Append to the main dataframe
                    self.data = pd.concat([self.data, combined_df], ignore_index=True)
    
    def interpolate_signal(self, signal, original_freq, target_freq):
        original_time = np.arange(0, len(signal)) / original_freq
        target_time = np.arange(0, len(signal) * (target_freq / original_freq)) / target_freq
        interpolator = interp1d(original_time, signal, kind='linear', fill_value='extrapolate')
        interpolated_signal = interpolator(target_time)
        return interpolated_signal

    def get_data(self):
        return self.data
    
    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.data, file)
        print(f"Data saved to {file_path}")

# Usage example:
preprocessor = UBFCPhysDataPreprocessor('../UBFC-PHYS')
preprocessor.load_data()
data = preprocessor.get_data()
preprocessor.save_to_pickle('../UBFC-PHYS/raw/merged.pkl')
print(data)
