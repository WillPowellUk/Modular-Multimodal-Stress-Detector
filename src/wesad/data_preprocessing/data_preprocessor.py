import pandas as pd
import numpy as np
from scipy import signal, stats
import sys
import gc
import os

class DataPreprocessor:
    DATA_PATH = 'src/wesad/WESAD/'
    RAW_PATH = os.path.join(DATA_PATH, 'raw/')
    CHEST_COLUMNS = ['sid', 'acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']
    ALL_COLUMNS = ['sid', 'c_acc_x', 'c_acc_y', 'c_acc_z', 'ecg', 'emg', 'c_eda', 'c_temp', 'resp', 'w_acc_x', 'w_acc_y', 'w_acc_z', 'bvp', 'w_eda', 'w_temp', 'label']
    IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    SF_BVP = 64
    SF_EDA = 4
    SF_TEMP = 4
    SF_ACC = 32
    SF_CHEST = 700

    def __init__(self):
        # Create the Raw directory if it does not exist
        if not os.path.exists(self.RAW_PATH):
            os.makedirs(self.RAW_PATH)

    def pkl_to_np_wrist(self, filename, subject_id):
        print(f"Processing wrist data for subject {subject_id}")
        try:
            unpickled_df = pd.read_pickle(filename)
        except FileNotFoundError:
            print(f"File {filename} not found. Have you ran the download_database.sh script and are you running from the root directory?")
            sys.exit(1)

        wrist_acc = unpickled_df["signal"]["wrist"]["ACC"]
        wrist_bvp = unpickled_df["signal"]["wrist"]["BVP"]
        wrist_eda = unpickled_df["signal"]["wrist"]["EDA"]
        wrist_temp = unpickled_df["signal"]["wrist"]["TEMP"]
        lbl = unpickled_df["label"].reshape(unpickled_df["label"].shape[0], 1)

        n_wrist_acc = len(wrist_acc)
        n_wrist_bvp = len(wrist_bvp)
        n_wrist_eda = len(wrist_eda)
        n_wrist_temp = len(wrist_temp)

        sid_acc = np.repeat(subject_id, n_wrist_acc).reshape((n_wrist_acc, 1))
        batch_size = self.SF_CHEST / self.SF_ACC
        lbl_m = np.zeros((n_wrist_acc, 1))
        for i in range(n_wrist_acc):
            lbl_m[i] = (stats.mode(lbl[round(i * batch_size): round((i + 1) * batch_size) - 1]))[0].squeeze()
        lbl_acc = lbl_m

        sid_bvp = np.repeat(subject_id, n_wrist_bvp).reshape((n_wrist_bvp, 1))
        batch_size = self.SF_CHEST / self.SF_BVP
        lbl_m = np.zeros((n_wrist_bvp, 1))
        for i in range(n_wrist_bvp):
            lbl_m[i] = (stats.mode(lbl[round(i * batch_size): round((i + 1) * batch_size) - 1]))[0].squeeze()
        lbl_bvp = lbl_m

        sid_eda_temp = np.repeat(subject_id, n_wrist_eda).reshape((n_wrist_eda, 1))
        batch_size = self.SF_CHEST / self.SF_EDA
        lbl_m = np.zeros((n_wrist_eda, 1))
        for i in range(n_wrist_eda):
            lbl_m[i] = (stats.mode(lbl[round(i * batch_size): round((i + 1) * batch_size) - 1]))[0].squeeze()
        lbl_eda_temp = lbl_m

        data1 = np.concatenate((sid_acc, wrist_acc, lbl_acc), axis=1)
        data2 = np.concatenate((sid_bvp, wrist_bvp, lbl_bvp), axis=1)
        data3 = np.concatenate((sid_eda_temp, wrist_eda, wrist_temp, lbl_eda_temp), axis=1)

        print(f"Finished processing wrist data for subject {subject_id}")
        return data1, data2, data3
    
    def merge_wrist_data(self):
        print("Merging wrist data...")
        combined_data = []

        for i, sid in enumerate(self.IDS):
            file = os.path.join(self.DATA_PATH, f'S{sid}', f'S{sid}.pkl')
            print(f"Processing file: {file}")
            subj1, subj2, subj3 = self.pkl_to_np_wrist(file, sid)

            # Find the minimum length among the three arrays
            min_length = min(subj1.shape[0], subj2.shape[0], subj3.shape[0])

            # Truncate the arrays to the minimum length
            subj1 = subj1[:min_length, :]
            subj2 = subj2[:min_length, :]
            subj3 = subj3[:min_length, :]

            # Combine the data for this subject, ensuring 'label' is the last column
            combined_subj = np.hstack((subj1[:, :-1], subj2[:, 1:-1], subj3[:, 1:-1], subj1[:, -1:]))
            combined_data.append(combined_subj)

        # Concatenate all subjects' data
        combined_data = np.vstack(combined_data)

        # Define columns
        all_columns = ['sid', 'w_acc_x', 'w_acc_y', 'w_acc_z', 'bvp', 'w_eda', 'w_temp', 'label']

        # Save the combined DataFrame to a single pickle file
        fn_merged = os.path.join(self.RAW_PATH, 'merged_wrist.pkl')
        pd.DataFrame(combined_data, columns=all_columns).to_pickle(fn_merged)

        print("Finished merging wrist data")

    
    def pkl_to_np_chest(self, filename, subject_id):
        print(f"Processing chest data for subject {subject_id}")
        unpickled_df = pd.read_pickle(filename)
        chest_acc = unpickled_df["signal"]["chest"]["ACC"]
        chest_ecg = unpickled_df["signal"]["chest"]["ECG"]
        chest_emg = unpickled_df["signal"]["chest"]["EMG"]
        chest_eda = unpickled_df["signal"]["chest"]["EDA"]
        chest_temp = unpickled_df["signal"]["chest"]["Temp"]
        chest_resp = unpickled_df["signal"]["chest"]["Resp"]
        lbl = unpickled_df["label"].reshape(unpickled_df["label"].shape[0], 1)
        sid = np.full((lbl.shape[0], 1), subject_id)
        chest_all = np.concatenate((sid, chest_acc, chest_ecg, chest_emg, chest_eda, chest_temp, chest_resp, lbl), axis=1)
        print(f"Finished processing chest data for subject {subject_id}")
        return chest_all
    
    def merge_chest_data(self):
        print("Merging chest data...")

        # Define a memory-mapped file for merged data
        merged_data_filename = os.path.join(self.RAW_PATH, 'merged_chest.dat')
        dtype = np.float32  # Assuming data type is float32; change if necessary

        # Initialize size estimation
        initial_file = self.DATA_PATH + 'S' + str(self.IDS[0]) + '/S' + str(self.IDS[0]) + '.pkl'
        initial_data = self.pkl_to_np_chest(initial_file, self.IDS[0])
        num_columns = initial_data.shape[1]
        
        # Estimate total number of rows (if possible)
        total_rows = sum(self.pkl_to_np_chest(self.DATA_PATH + 'S' + str(sid) + '/S' + str(sid) + '.pkl', sid).shape[0] for sid in self.IDS)
        
        # Create a memory-mapped file with estimated size
        merged_data = np.memmap(merged_data_filename, dtype=dtype, mode='w+', shape=(total_rows, num_columns))
        
        current_row = 0
        
        try:
            for i, sid in enumerate(self.IDS):
                file = self.DATA_PATH + 'S' + str(sid) + '/S' + str(sid) + '.pkl'
                print(f"Processing file: {file}")
                subj_data = self.pkl_to_np_chest(file, sid)
                
                rows = subj_data.shape[0]
                merged_data[current_row:current_row + rows] = subj_data
                current_row += rows

                # Free memory
                del subj_data
                gc.collect()

            # Flush data to disk
            merged_data.flush()

            # Save as pickle file
            final_df = pd.DataFrame(np.array(merged_data), columns=self.CHEST_COLUMNS)
            final_df.to_pickle(os.path.join(self.RAW_PATH, 'merged_chest_temp.pkl'))

            print("Finished merging chest data")

        finally:
            del merged_data
            gc.collect()
            if os.path.exists(merged_data_filename):
                os.remove(merged_data_filename)

    def filter_chest_data(self):
        print("Filtering chest data...")
        df = pd.read_pickle(os.path.join(self.RAW_PATH, "merged_chest_temp.pkl"))
        df_fltr = df[df["label"].isin([1, 2, 3])]
        df_fltr = df_fltr[df_fltr["temp"] > 0]
        pd.DataFrame(df_fltr, columns=self.CHEST_COLUMNS).to_pickle(os.path.join(self.RAW_PATH, "merged_chest.pkl"))
        os.remove(os.path.join(self.RAW_PATH, "merged_chest_temp.pkl"))
        print("Finished filtering chest data")

    def preprocess(self):
        print("Starting preprocessing...")
        # self.merge_wrist_data()
        self.merge_chest_data()
        self.filter_chest_data()
        print("Preprocessing completed")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess()
