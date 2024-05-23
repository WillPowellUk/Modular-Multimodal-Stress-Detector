import pandas as pd
import numpy as np
import os
import json
import pickle
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
        
        self.df = pd.read_pickle(self.data_path)

        # Load the JSON data from the file
        with open(config_path, 'r') as file:
            self.sampling_rates = json.load(file)

    def preprocess_signals(self):
        print("Starting signal preprocessing...")

        if self.wrist:
            self._preprocess_wrist_signals()
        else:
            self._preprocess_chest_signals()

        # Save preprocessed data
        print("Saving preprocessed data...")
        suffix = "wrist" if self.wrist else "chest"
        self.save_preprocessed_data(self.df, f"{suffix}_preprocessed.pkl")
        print("Preprocessed data saved successfully.")

    def _preprocess_chest_signals(self):
        # Chest ECG Preprocessing
        if 'ecg' in self.sampling_rates:
            print("Processing Chest ECG...")
            ecg_processor = ECGPreprocessing(self.df, fs=self.sampling_rates['ecg'])
            self.df = ecg_processor.process(use_neurokit=True, plot=False)
            print("Chest ECG processing completed.")

        # Chest EMG Preprocessing
        if 'emg' in self.sampling_rates:
            print("Processing Chest EMG...")
            emg_processor = EMGPreprocessing(self.df, fs=self.sampling_rates['emg'])
            self.df = emg_processor.process()
            print("Chest EMG processing completed.")

        # Chest EDA Preprocessing
        if 'eda' in self.sampling_rates:
            print("Processing Chest EDA...")
            eda_processor = EDAPreprocessing(self.df, fs=self.sampling_rates['eda'])
            self.df = eda_processor.process()
            print("Chest EDA processing completed.")

        # Chest TEMP Preprocessing
        if 'temp' in self.sampling_rates:
            print("Processing Chest TEMP...")
            temp_processor = TempPreprocessing(self.df, window_size=11, poly_order=3)
            self.df = temp_processor.process()
            print("Chest TEMP processing completed.")

        # Chest RESP Preprocessing
        if 'resp' in self.sampling_rates:
            print("Processing Chest RESP...")
            resp_processor = RespPreprocessing(self.df, fs=self.sampling_rates['resp'])
            self.df = resp_processor.process()
            print("Chest RESP processing completed.")

        # Chest ACC Preprocessing
        if 'acc' in self.sampling_rates:
            print("Processing Chest ACC...")
            acc_processor = AccPreprocessing(self.df, window_size=31, poly_order=5)
            self.df = acc_processor.process()
            print("Chest ACC processing completed.")

    def _preprocess_wrist_signals(self):
        # Wrist ACC Preprocessing
        if 'w_acc' in self.sampling_rates:
            print("Processing Wrist ACC...")
            acc_processor_wrist = AccPreprocessing(self.df, wrist=True)
            self.df = acc_processor_wrist.process()
            print("Wrist ACC processing completed.")

        # Wrist BVP Preprocessing
        if 'bvp' in self.sampling_rates:
            print("Processing Wrist BVP...")
            bvp_processor = BVPPreprocessing(self.df, fs=self.sampling_rates['bvp'])
            self.df = bvp_processor.process()
            print("Wrist BVP processing completed.")

        # Wrist EDA Preprocessing
        if 'w_eda' in self.sampling_rates:
            print("Processing Wrist EDA...")
            eda_processor_wrist = EDAPreprocessing(self.df, fs=self.sampling_rates['w_eda'], lp_order=6, lp_cutoff=1.0, wrist=True)
            self.df = eda_processor_wrist.process()
            print("Wrist EDA processing completed.")

        # Wrist TEMP Preprocessing
        if 'w_temp' in self.sampling_rates:
            print("Processing Wrist TEMP...")
            temp_processor_wrist = TempPreprocessing(self.df, window_size=11, poly_order=3, wrist=True)
            self.df = temp_processor_wrist.process()
            print("Wrist TEMP processing completed.")

    def save_preprocessed_data(self, data, filename):
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        
        with open(self.output_dir, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {self.output_dir}")
