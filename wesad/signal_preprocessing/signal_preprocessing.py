import pandas as pd
import numpy as np
import os
import pickle
from ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from ml_pipeline.preprocessing.bvp_preprocessing import BVPPreprocessing
from ml_pipeline.preprocessing.eda_preprocessing import EDAPreprocessing
from ml_pipeline.preprocessing.temp_preprocessing import TempPreprocessing
from ml_pipeline.preprocessing.resp_preprocessing import RespPreprocessing
from ml_pipeline.preprocessing.emg_preprocessing import EMGPreprocessing
from ml_pipeline.preprocessing.acc_preprocessing import ACCPreprocessing

class SignalPreprocessor:
    DATA_PATH = 'wesad/WESAD/'
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
        # Chest ECG Preprocessing
        ecg_processor = ECGPreprocessing(self.df_c)
        self.df_c = ecg_processor.process()

        # Chest EMG Preprocessing
        emg_processor = EMGPreprocessing(self.df_c)
        self.df_c = emg_processor.process()

        # Chest EDA Preprocessing
        eda_processor = EDAPreprocessing(self.df_c)
        self.df_c = eda_processor.process()

        # Chest TEMP Preprocessing
        temp_processor = TempPreprocessing(self.df_c, window_size=11, poly_order=3)
        self.df_c = temp_processor.process()

        # Chest RESP Preprocessing
        resp_processor = RespPreprocessing(self.df_c)
        self.df_c = resp_processor.process()

        # Chest ACC Preprocessing
        acc_processor = ACCPreprocessing(self.df_c, window_size=31, poly_order=5)
        self.df_c = acc_processor.process()

        # Wrist BVP Preprocessing
        bvp_processor = BVPPreprocessing(self.df_w1)
        self.df_w1 = bvp_processor.process()me

        # Wrist TEMP Preprocessing
        temp_processor_wrist = TempPreprocessing(self.df_w2, window_size=11, poly_order=3)
        self.df_w2 = temp_processor_wrist.process()

        # Wrist ACC Preprocessing
        acc_processor_wrist = ACCPreprocessing(self.df_w2, window_size=31, poly_order=5)
        self.df_w2 = acc_processor_wrist.process()

        # Save preprocessed data
        self.save_preprocessed_data(self.df_c, f"{self.output_dir}chest_preprocessed.pkl")
        self.save_preprocessed_data(self.df_w1, f"bvp_preprocessed.pkl")
        self.save_preprocessed_data(self.df_w2, "wrist_temp_acc_preprocessed.pkl")

    def save_preprocessed_data(self, data, filename):
        output_path = f"{self.output_dir}{filename}"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
