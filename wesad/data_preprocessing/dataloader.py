import pandas as pd
import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

class WesadPreprocessor:
    def __init__(self, window_size=0.25, chest_path="wesad/WESAD/merged_chest_fltr.pkl", 
                 bvp_path="wesad/WESAD/subj_merged_bvp_w.pkl",
                 eda_temp_path="wesad/WESAD/subj_merged_eda_temp_w.pkl",
                 feat_sf700=['ecg', 'emg', 'eda', 'temp', 'resp'],
                 feat_sf64=['bvp'],
                 feat_sf4=['w_eda', 'w_temp'], 
                 sf_chest=700, sf_BVP=64, sf_EDA=4, sf_TEMP=4):
        
        self.chest_path = chest_path
        self.bvp_path = bvp_path
        self.eda_temp_path = eda_temp_path
        self.feat_sf700 = feat_sf700
        self.feat_sf64 = feat_sf64
        self.feat_sf4 = feat_sf4
        self.sf_chest = sf_chest
        self.sf_BVP = sf_BVP
        self.sf_EDA = sf_EDA
        self.sf_TEMP = sf_TEMP
        self.window_size = window_size

        self.df_c = pd.read_pickle(self.chest_path)
        self.df_w1 = pd.read_pickle(self.bvp_path)
        self.df_w2 = pd.read_pickle(self.eda_temp_path)
        self.df_w1 = self.df_w1[self.df_w1["label"].isin([1, 2, 3])]
        self.df_w2 = self.df_w2[self.df_w2["label"].isin([1, 2, 3])]

        self.batch_size = int(self.sf_chest * self.window_size)
        self.batch_size_bvp = int(self.sf_BVP * self.window_size)
        self.batch_size_eda = int(self.sf_EDA * self.window_size)
        self.batch_size_temp = int(self.sf_TEMP * self.window_size)

        self.ids = self.df_c["sid"].unique().astype(int)
        self.K = len(self.df_c["label"].unique())

    def one_hot_enc(self, r, k):
        new_r = np.zeros((r.shape[0], k))
        for i, val in enumerate(r):
            new_r[i, val - 1] = 1
        return new_r

    def get_data(self, test_id):
        cnt = 0
        scaler = StandardScaler()

        for j in self.ids:
            df_s = self.df_c[self.df_c["sid"] == j]
            n = (len(df_s) // self.batch_size) * self.batch_size
            df_s = df_s[:n]
            s = scaler.fit_transform(df_s[self.feat_sf700].values)
            s = s.reshape(int(s.shape[0] / self.batch_size), s.shape[1], self.batch_size)

            lbl_m = np.zeros((s.shape[0], 1))
            lbl = df_s["label"].values.astype(int)
            for i in range(s.shape[0]):
                lbl_m[i] = int((stats.mode(lbl[i * self.batch_size: (i + 1) * self.batch_size - 1]))[0].squeeze())
            y_k = lbl_m.astype(int)
            s_y = self.one_hot_enc(lbl_m.astype(int), self.K).astype(int)

            if j == test_id:
                x_test = s
                y_test = s_y
                yk_test = y_k
            else:
                if cnt:
                    merged = np.concatenate((merged, s), axis=0)
                    merged_y = np.concatenate((merged_y, s_y), axis=0)
                    merged_yk = np.concatenate((merged_yk, y_k), axis=0)
                else:
                    merged = s
                    merged_y = s_y
                    merged_yk = y_k
                cnt += 1

        return merged, merged_y, x_test, y_test, merged_yk, yk_test

    def preprocess_data(self):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        test_data_merged = []
        test_labels_merged = []

        for test_id in self.ids:
            merged, merged_y, x_test, y_test, merged_yk, yk_test = self.get_data(test_id)
            train_data.append((merged, merged_y))
            test_data.append((x_test, y_test))
            test_data_merged.append(merged_yk)
            test_labels_merged.append(yk_test)
        
        return train_data, test_data, test_data_merged, test_labels_merged

    def save_preprocessed_data(self, filepath):
        train_data, test_data, test_data_merged, test_labels_merged = self.preprocess_data()
        with open(filepath, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'test_data': test_data,
                'test_data_merged': test_data_merged,
                'test_labels_merged': test_labels_merged
            }, f)

    def load_preprocessed_data(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data['train_data'], data['test_data'], data['test_data_merged'], data['test_labels_merged']

class WesadDataset(Dataset):
    def __init__(self, train_data, test_data, transform=None):
        self.train_data = train_data
        self.test_data = test_data
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        train = self.train_data[idx]
        test = self.test_data[idx]
        
        if self.transform:
            train = self.transform(train)
            test = self.transform(test)
        
        return train, test

def perform_loocv(preprocessor, use_preprocessed=False, preprocessed_filepath=None):
    if use_preprocessed and preprocessed_filepath:
        train_data, test_data, _, _ = preprocessor.load_preprocessed_data(preprocessed_filepath)
    else:
        train_data, test_data, _, _ = preprocessor.preprocess_data()
        if preprocessed_filepath:
            preprocessor.save_preprocessed_data(preprocessed_filepath)
    
    results = []
    for idx in range(len(preprocessor.ids)):
        train_data_split = train_data[idx]
        test_data_split = test_data[idx]
        
        wesad_dataset = WesadDataset([train_data_split], [test_data_split])
        dataloader = DataLoader(wesad_dataset, batch_size=1, shuffle=False)

        for train, test in dataloader:
            # Perform training on train data
            # Train the model here and validate on test data
            results.append((train, test))
            # Add model training and evaluation logic here
            
    return results

# Usage example
preprocessor = WesadPreprocessor()
preprocessed_filepath = 'preprocessed_wesad_data.pkl'

# First time processing and saving the data
perform_loocv(preprocessor, use_preprocessed=False, preprocessed_filepath=preprocessed_filepath)

# Later on, loading the preprocessed data for LOOCV
results = perform_loocv(preprocessor, use_preprocessed=True, preprocessed_filepath=preprocessed_filepath)
