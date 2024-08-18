import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import sys
import gc
import os


class WESADDataPreprocessor:
    CHEST_COLUMNS = [
        "sid",
        "acc1",
        "acc2",
        "acc3",
        "ecg",
        "emg",
        "eda",
        "temp",
        "resp",
        "label",
    ]
    ALL_COLUMNS = [
        "sid",
        "c_acc_x",
        "c_acc_y",
        "c_acc_z",
        "ecg",
        "emg",
        "c_eda",
        "c_temp",
        "resp",
        "w_acc_x",
        "w_acc_y",
        "w_acc_z",
        "bvp",
        "w_eda",
        "w_temp",
        "label",
    ]
    SF_BVP = 64
    SF_EDA = 4
    SF_TEMP = 4
    SF_ACC = 32
    SF_WRIST = max(SF_BVP, SF_EDA, SF_TEMP)
    SF_CHEST = 700

    def __init__(
        self,
        dataset_path="src/wesad/WESAD/",
        IDs=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
    ):
        self.IDs = IDs
        self.dataset_path = dataset_path
        self.raw_path = os.path.join(dataset_path, "raw/")

        # Ensure the directories exists
        dir_name = os.path.dirname(self.dataset_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        dir_name = os.path.dirname(self.raw_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def pkl_to_np_wrist(self, filename, subject_id):
        print(f"Processing wrist data for subject {subject_id}")
        try:
            unpickled_df = pd.read_pickle(filename)
        except FileNotFoundError:
            print(
                f"File {filename} not found. Have you ran the download_database.sh script and are you running from the root directory?"
            )
            sys.exit(1)

        # Assuming unpickled_df is already loaded and subject_id is defined
        wrist_acc = unpickled_df["signal"]["wrist"]["ACC"]
        wrist_bvp = unpickled_df["signal"]["wrist"]["BVP"]
        wrist_eda = unpickled_df["signal"]["wrist"]["EDA"]
        wrist_temp = unpickled_df["signal"]["wrist"]["TEMP"]
        lbl = unpickled_df["label"].reshape(unpickled_df["label"].shape[0], 1)

        n_wrist_acc = len(wrist_acc)
        n_wrist_bvp = len(wrist_bvp)
        n_wrist_eda = len(wrist_eda)
        n_wrist_temp = len(wrist_temp)

        # sid / label for wrist data will be bvp as all data will be resampled to bvp since it has the highest sampling rate
        sid_bvp = np.repeat(subject_id, n_wrist_bvp).reshape((n_wrist_bvp, 1))

        # Ensure lbl is one-dimensional for interpolation
        lbl = lbl.flatten()

        # Create time arrays for original and target sampling rates
        time_original = np.linspace(0, len(lbl) / self.SF_CHEST, len(lbl))
        time_target = np.linspace(0, len(lbl) / self.SF_CHEST, n_wrist_bvp)

        # Create the interpolator
        interpolator = interp1d(
            time_original, lbl, kind="nearest", fill_value="extrapolate"
        )

        # Interpolate the labels to the BVP sampling frequency
        lbl_bvp = interpolator(time_target).reshape(-1, 1)

        # resample to bvp sampling rate
        resampled_wrist_acc = np.zeros((n_wrist_bvp, 3))

        # Resample to bvp sampling rate
        for i in range(3):
            resampled_wrist_acc[:, i] = signal.resample(wrist_acc[:, i], n_wrist_bvp)

        wrist_acc = resampled_wrist_acc
        wrist_eda = signal.resample(wrist_eda, n_wrist_bvp)
        wrist_temp = signal.resample(wrist_temp, n_wrist_bvp)

        print(f"Finished processing wrist data for subject {subject_id}")
        return sid_bvp, wrist_acc, wrist_bvp, wrist_eda, wrist_temp, lbl_bvp

    def merge_wrist_data(self):
        print("Merging wrist data...")
        combined_data = []

        for i, sid in enumerate(self.IDs):
            file = os.path.join(self.dataset_path, f"S{sid}", f"S{sid}.pkl")
            print(f"Processing file: {file}")
            (
                sid,
                wrist_acc,
                wrist_bvp,
                wrist_eda,
                wrist_temp,
                lbl,
            ) = self.pkl_to_np_wrist(file, sid)

            # Combine the data for this subject, ensuring 'label' is the last column
            combined_subj = np.hstack(
                (sid, wrist_acc, wrist_bvp, wrist_eda, wrist_temp, lbl)
            )
            combined_data.append(combined_subj)

        # Concatenate all subjects' data
        combined_data = np.vstack(combined_data)

        # Define columns
        all_columns = [
            "sid",
            "w_acc_x",
            "w_acc_y",
            "w_acc_z",
            "bvp",
            "w_eda",
            "w_temp",
            "label",
        ]

        # Save the combined DataFrame to a single pickle file
        fn_merged = os.path.join(self.raw_path, "merged_wrist.pkl")
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
        chest_all = np.concatenate(
            (
                sid,
                chest_acc,
                chest_ecg,
                chest_emg,
                chest_eda,
                chest_temp,
                chest_resp,
                lbl,
            ),
            axis=1,
        )
        print(f"Finished processing chest data for subject {subject_id}")
        return chest_all

    def merge_chest_data(self):
        print("Merging chest data...")

        # Define a memory-mapped file for merged data
        merged_data_filename = os.path.join(self.dataset_path, "merged_chest.dat")
        dtype = np.float32  # Assuming data type is float32; change if necessary

        # Initialize size estimation
        initial_file = (
            self.dataset_path
            + "S"
            + str(self.IDs[0])
            + "/S"
            + str(self.IDs[0])
            + ".pkl"
        )
        initial_data = self.pkl_to_np_chest(initial_file, self.IDs[0])
        num_columns = initial_data.shape[1]

        # Estimate total number of rows (if possible)
        total_rows = sum(
            self.pkl_to_np_chest(
                self.dataset_path + "S" + str(sid) + "/S" + str(sid) + ".pkl", sid
            ).shape[0]
            for sid in self.IDs
        )

        # Create a memory-mapped file with estimated size
        merged_data = np.memmap(
            merged_data_filename,
            dtype=dtype,
            mode="w+",
            shape=(total_rows, num_columns),
        )

        current_row = 0

        try:
            for i, sid in enumerate(self.IDs):
                file = self.dataset_path + "S" + str(sid) + "/S" + str(sid) + ".pkl"
                print(f"Processing file: {file}")
                subj_data = self.pkl_to_np_chest(file, sid)

                rows = subj_data.shape[0]
                merged_data[current_row : current_row + rows] = subj_data
                current_row += rows

                # Free memory
                del subj_data
                gc.collect()

            # Flush data to disk
            merged_data.flush()

            # Save as pickle file
            final_df = pd.DataFrame(np.array(merged_data), columns=self.CHEST_COLUMNS)
            final_df.to_pickle(os.path.join(self.raw_path, "merged_chest_temp.pkl"))

            print("Finished merging chest data")

        finally:
            del merged_data
            gc.collect()
            if os.path.exists(merged_data_filename):
                os.remove(merged_data_filename)

    def filter_chest_data(self):
        print("Filtering chest data...")
        df = pd.read_pickle(os.path.join(self.raw_path, "merged_chest_temp.pkl"))
        df_fltr = df[df["label"].isin([1, 2, 3])]
        df_fltr = df_fltr[df_fltr["temp"] > 0]
        pd.DataFrame(df_fltr, columns=self.CHEST_COLUMNS).to_pickle(
            os.path.join(self.raw_path, "merged_chest.pkl")
        )
        os.remove(os.path.join(self.raw_path, "merged_chest_temp.pkl"))
        print("Finished filtering chest data")

    def preprocess(self):
        print("Starting preprocessing...")
        self.merge_wrist_data()
        self.merge_chest_data()
        self.filter_chest_data()
        print("Preprocessing completed")


if __name__ == "__main__":
    preprocessor = WESADDataPreprocessor()
    preprocessor.preprocess()
