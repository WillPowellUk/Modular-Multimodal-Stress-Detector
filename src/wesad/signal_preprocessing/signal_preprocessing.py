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
    DATA_PATH = 'src/wesad/WESAD/'
    CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned/')
    
    def __init__(self, chest_path="src/wesad/WESAD/raw/merged_chest.pkl", 
                 wrist_path="src/wesad/WESAD/raw/merged_wrist.pkl",
                 config_path="src/wesad/wesad_configuration.json",
                 output_dir=CLEANED_PATH):
        
        # Create the cleaned directory if it does not exist
        if not os.path.exists(self.CLEANED_PATH):
            os.makedirs(self.CLEANED_PATH)
        
        self.chest_path = chest_path
        self.wrist_path = wrist_path
        self.output_dir = output_dir
        
        self.df_c = pd.read_pickle(self.chest_path)
        self.df_w = pd.read_pickle(self.wrist_path)

        # Load the JSON data from the file
        with open(config_path, 'r') as file:
            self.sampling_rates = json.load(file)


    def preprocess_signals(self):
        print("Starting signal preprocessing...")

        # Chest ECG Preprocessing
        print("Processing Chest ECG...")
        ecg_processor = ECGPreprocessing(self.df_c, fs=self.sampling_rates['ecg'])
        self.df_c = ecg_processor.process()
        print("Chest ECG processing completed.")

        # Chest EMG Preprocessing
        print("Processing Chest EMG...")
        emg_processor = EMGPreprocessing(self.df_c, fs=self.sampling_rates['emg'])
        self.df_c = emg_processor.process()
        print("Chest EMG processing completed.")

        # Chest EDA Preprocessing
        print("Processing Chest EDA...")
        eda_processor = EDAPreprocessing(self.df_c, fs=self.sampling_rates['eda'])
        self.df_c = eda_processor.process()
        print("Chest EDA processing completed.")

        # Chest TEMP Preprocessing
        print("Processing Chest TEMP...")
        temp_processor = TempPreprocessing(self.df_c, window_size=11, poly_order=3)
        self.df_c = temp_processor.process()
        print("Chest TEMP processing completed.")

        # Chest RESP Preprocessing
        print("Processing Chest RESP...")
        resp_processor = RespPreprocessing(self.df_c, fs=self.sampling_rates['resp'])
        self.df_c = resp_processor.process()
        print("Chest RESP processing completed.")

        # Chest ACC Preprocessing
        print("Processing Chest ACC...")
        acc_processor = AccPreprocessing(self.df_c, window_size=31, poly_order=5)
        self.df_c = acc_processor.process()
        print("Chest ACC processing completed.")

        # Wrist ACC Preprocessing
        print("Processing Wrist ACC...")
        acc_processor_wrist = AccPreprocessing(self.df_w, wrist=True)
        self.df_w = acc_processor_wrist.process()
        print("Wrist ACC processing completed.")

        # Wrist BVP Preprocessing
        print("Processing Wrist BVP...")
        bvp_processor = BVPPreprocessing(self.df_w, fs=self.sampling_rates['bvp'])
        self.df_w = bvp_processor.process()
        print("Wrist BVP processing completed.")

        # Wrist EDA Preprocessing
        print("Processing Wrist EDA...")
        eda_processor_wrist = EDAPreprocessing(self.df_w, fs=self.sampling_rates['eda'], wrist=True)
        self.df_w = eda_processor_wrist.process()
        print("Wrist EDA processing completed.")

        # Wrist TEMP Preprocessing
        print("Processing Wrist TEMP...")
        temp_processor_wrist = TempPreprocessing(self.df_w, window_size=11, poly_order=3, wrist=True)
        self.df_w = temp_processor_wrist.process()
        print("Wrist TEMP processing completed.")

        # Save preprocessed data
        print("Saving preprocessed data...")
        self.save_preprocessed_data(self.df_c, f"chest_preprocessed.pkl")
        self.save_preprocessed_data(self.df_w, f"wrist_preprocessed.pkl")
        print("Preprocessed data saved successfully.")

    def save_preprocessed_data(self, data, filename):
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {output_path}")
