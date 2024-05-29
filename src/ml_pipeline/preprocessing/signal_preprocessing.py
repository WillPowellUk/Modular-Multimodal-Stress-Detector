import pandas as pd
import numpy as np
import os
import re
import pickle
from src.ml_pipeline.utils.utils import get_max_sampling_rate, get_active_key
from src.ml_pipeline.preprocessing.acc_preprocessing import AccPreprocessing
from src.ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from src.ml_pipeline.preprocessing.bvp_preprocessing import BVPPreprocessing
from src.ml_pipeline.preprocessing.eda_preprocessing import EDAPreprocessing
from src.ml_pipeline.preprocessing.temp_preprocessing import TempPreprocessing
from src.ml_pipeline.preprocessing.resp_preprocessing import RespPreprocessing
from src.ml_pipeline.preprocessing.emg_preprocessing import EMGPreprocessing

class SignalPreprocessor:
    def __init__(self, data_path: str, output_dir:str, config_path: str, wrist=False):
        self.data_path = data_path
        self.output_dir = output_dir
        self.wrist = wrist
        self.config_path = config_path
        
        self.df = pd.read_pickle(self.data_path)

        self.sampling_rate = get_max_sampling_rate(config_path)

    def preprocess_signals(self):
        print("Starting signal preprocessing...")

        sensors = get_active_key(self.config_path, 'sensors')

        # Chest ECG Preprocessing
        if 'ecg' in sensors:
            print("Processing Chest ECG...")
            ecg_processor = ECGPreprocessing(self.df, fs=self.sampling_rate)
            self.df = ecg_processor.process(use_neurokit=True, plot=False)
            print("Chest ECG processing completed.")

        # Chest EMG Preprocessing
        if 'emg' in sensors:
            print("Processing Chest EMG...")
            emg_processor = EMGPreprocessing(self.df, fs=self.sampling_rate)
            self.df = emg_processor.process()
            print("Chest EMG processing completed.")

        # Chest EDA Preprocessing
        if 'eda' in sensors:
            print("Processing Chest EDA...")
            eda_processor = EDAPreprocessing(self.df, fs=self.sampling_rate)
            self.df = eda_processor.process()
            print("Chest EDA processing completed.")

        # Chest TEMP Preprocessing
        if 'temp' in sensors:
            print("Processing Chest TEMP...")
            temp_processor = TempPreprocessing(self.df, window_size=11, poly_order=3)
            self.df = temp_processor.process()
            print("Chest TEMP processing completed.")

        # Chest RESP Preprocessing
        if 'resp' in sensors:
            print("Processing Chest RESP...")
            resp_processor = RespPreprocessing(self.df, fs=self.sampling_rate)
            self.df = resp_processor.process()
            print("Chest RESP processing completed.")

        # Chest ACC Preprocessing
        if any(re.search(r'(?<!w_)acc', sensor) for sensor in sensors):
            print("Processing Chest ACC...")
            acc_processor = AccPreprocessing(self.df, window_size=31, poly_order=5)
            self.df = acc_processor.process()
            print("Chest ACC processing completed.")

        # Wrist ACC Preprocessing
        if any(re.search(r'w_acc', sensor) for sensor in sensors):
            print("Processing Wrist ACC...")
            acc_processor_wrist = AccPreprocessing(self.df, wrist=True)
            self.df = acc_processor_wrist.process()
            print("Wrist ACC processing completed.")

        # Wrist BVP Preprocessing
        if 'bvp' in sensors:
            print("Processing Wrist BVP...")
            bvp_processor = BVPPreprocessing(self.df, fs=self.sampling_rate)
            self.df = bvp_processor.process()
            print("Wrist BVP processing completed.")

        # Wrist EDA Preprocessing
        if 'w_eda' in sensors:
            print("Processing Wrist EDA...")
            eda_processor_wrist = EDAPreprocessing(self.df, fs=self.sampling_rate, lp_order=6, lp_cutoff=1.0, wrist=True)
            self.df = eda_processor_wrist.process()
            print("Wrist EDA processing completed.")

        # Wrist TEMP Preprocessing
        if 'w_temp' in sensors:
            print("Processing Wrist TEMP...")
            temp_processor_wrist = TempPreprocessing(self.df, window_size=11, poly_order=3, wrist=True)
            self.df = temp_processor_wrist.process()
            print("Wrist TEMP processing completed.")

        # Save preprocessed data
        print("Saving preprocessed data...")
        suffix = "wrist" if self.wrist else "chest"
        self.save_preprocessed_data(self.df, f"{suffix}_preprocessed.pkl")
        print("Preprocessed data saved successfully.")

    def save_preprocessed_data(self, data, filename):
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        
        with open(self.output_dir, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {self.output_dir}")
