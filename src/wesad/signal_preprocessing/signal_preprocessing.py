import pandas as pd
import numpy as np
import os
import pickle
from src.ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from src.ml_pipeline.preprocessing.bvp_preprocessing import BVPPreprocessing
from src.ml_pipeline.preprocessing.eda_preprocessing import EDAPreprocessing
from src.ml_pipeline.preprocessing.temp_preprocessing import TempPreprocessing
from src.ml_pipeline.preprocessing.resp_preprocessing import RespPreprocessing
from src.ml_pipeline.preprocessing.emg_preprocessing import EMGPreprocessing
from src.ml_pipeline.preprocessing.acc_preprocessing import ACCPreprocessing

class SignalPreprocessor:
    DATA_PATH = 'src/wesad/WESAD/'
    CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned/')
    
    def __init__(self, chest_path="wesad/WESAD/raw/merged_chest_fltr.pkl", 
                 bvp_path="wesad/WESAD/raw/subj_merged_bvp_w.pkl",
                 eda_temp_path="wesad/WESAD/raw/subj_merged_eda_temp_w.pkl",
                 output_dir=CLEANED_PATH):
        
        # Create the cleaned directory if it does not exist
        if not os.path.exists(self.CLEANED_PATH):
            os.makedirs(self.CLEANED_PATH)
        
        self.chest_path = chest_path
        self.bvp_path = bvp_path
        self.eda_temp_path = eda_temp_path
        self.output_dir = output_dir
        
        self.df_c = pd.read_pickle(self.chest_path)
        self.df_w1 = pd.read_pickle(self.bvp_path)
        self.df_w2 = pd.read_pickle(self.eda_temp_path)

    def preprocess_signals(self):
        print("Starting signal preprocessing...")

        # Chest ECG Preprocessing
        print("Processing Chest ECG...")
        ecg_processor = ECGPreprocessing(self.df_c)
        self.df_c = ecg_processor.process()
        print("Chest ECG processing completed.")

        # Chest EMG Preprocessing
        print("Processing Chest EMG...")
        emg_processor = EMGPreprocessing(self.df_c)
        self.df_c = emg_processor.process()
        print("Chest EMG processing completed.")

        # Chest EDA Preprocessing
        print("Processing Chest EDA...")
        eda_processor = EDAPreprocessing(self.df_c)
        self.df_c = eda_processor.process()
        print("Chest EDA processing completed.")

        # Chest TEMP Preprocessing
        print("Processing Chest TEMP...")
        temp_processor = TempPreprocessing(self.df_c, window_size=11, poly_order=3)
        self.df_c = temp_processor.process()
        print("Chest TEMP processing completed.")

        # Chest RESP Preprocessing
        print("Processing Chest RESP...")
        resp_processor = RespPreprocessing(self.df_c)
        self.df_c = resp_processor.process()
        print("Chest RESP processing completed.")

        # Chest ACC Preprocessing
        print("Processing Chest ACC...")
        acc_processor = ACCPreprocessing(self.df_c, window_size=31, poly_order=5)
        self.df_c = acc_processor.process()
        print("Chest ACC processing completed.")

        # Wrist BVP Preprocessing
        print("Processing Wrist BVP...")
        bvp_processor = BVPPreprocessing(self.df_w1)
        self.df_w1 = bvp_processor.process()
        print("Wrist BVP processing completed.")

        # Wrist TEMP Preprocessing
        print("Processing Wrist TEMP...")
        temp_processor_wrist = TempPreprocessing(self.df_w2, window_size=11, poly_order=3)
        self.df_w2 = temp_processor_wrist.process()
        print("Wrist TEMP processing completed.")

        # Wrist ACC Preprocessing
        print("Processing Wrist ACC...")
        acc_processor_wrist = ACCPreprocessing(self.df_w2, window_size=31, poly_order=5)
        self.df_w2 = acc_processor_wrist.process()
        print("Wrist ACC processing completed.")

        # Save preprocessed data
        print("Saving preprocessed data...")
        self.save_preprocessed_data(self.df_c, "chest_preprocessed.pkl")
        self.save_preprocessed_data(self.df_w1, "bvp_preprocessed.pkl")
        self.save_preprocessed_data(self.df_w2, "wrist_temp_acc_preprocessed.pkl")
        print("Preprocessed data saved successfully.")

    def save_preprocessed_data(self, data, filename):
        output_path = f"{self.output_dir}{filename}"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {output_path}")
