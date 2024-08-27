import pandas as pd
import numpy as np
import os
import pickle

def create_subject_pickle_file(subject_id, dir):
    # Define sampling frequencies
    EMG_SAMPLING_FREQUENCY = 2048
    ECG_SAMPLING_FREQUENCY = 130
    POLAR_HR_SAMPLING_FREQUENCY = 1
    POLAR_ACC_SAMPLING_FREQUENCY = 32
    FNIRS_SAMPLING_FREQUENCY = 10
    EMPATICA_ACC_SAMPLING_FREQUENCY = 32
    EMPATICA_BVP_SAMPLING_FREQUENCY = 64
    EMPATICA_HR_SAMPLING_FREQUENCY = 1
    EMPATICA_TEMP_SAMPLING_FREQUENCY = 4
    EMPATICA_EDA_SAMPLING_FREQUENCY = 4

    # Function to create labels synchronized with data
    def create_labels(time_points, phases):
        labels = np.zeros(len(time_points))
        for i, (start, end, label) in enumerate(phases):
            mask = (time_points >= start) & (time_points < end)
            labels[mask] = label
        return labels

    # Read phases and adjust to new times
    # Create labels
    label_description = ['Baseline sit', 'Baseline stand', 'Anticipation', 'Entry Judges', 'Interview', 'Arithmetic', 'Goodbye']
    label_ids = [0, 1, 2, 3, 4, 4, 5]
    phases_df = pd.read_csv(f'{dir}/Phases.csv', delimiter=';')
    phases = list(zip(phases_df['StartTime'][:-1], phases_df['StartTime'][1:], label_ids))

    # Initialize the dictionary to store all data
    data_dict = {
        'polar': {},
        'empatica': {},
        'quattrocento': {},
        'myndsens': {}
    }

    # Polar Data
    # ECG
    ecg_df = pd.read_csv(f'{dir}/polar/ECG.csv')
    ecg_time_points = np.arange(0, len(ecg_df) / ECG_SAMPLING_FREQUENCY, 1 / ECG_SAMPLING_FREQUENCY)
    ecg_df['Label'] = create_labels(ecg_time_points, phases)
    data_dict['polar']['ecg'] = ecg_df

    # IBI
    ibi_df = pd.read_csv(f'{dir}/polar/IBI.csv')
    ibi_time_points = np.arange(0, len(ibi_df) / POLAR_HR_SAMPLING_FREQUENCY, 1 / POLAR_HR_SAMPLING_FREQUENCY)
    ibi_df['Label'] = create_labels(ibi_time_points, phases)
    data_dict['polar']['ibi'] = ibi_df

    # ACC
    acc_df = pd.read_csv(f'{dir}/polar/ACC.csv')
    acc_time_points = np.arange(0, len(acc_df) / POLAR_ACC_SAMPLING_FREQUENCY, 1 / POLAR_ACC_SAMPLING_FREQUENCY)
    acc_df['Label'] = create_labels(acc_time_points, phases)
    data_dict['polar']['acc'] = acc_df

    # Empatica Data
    # ACC
    empatica_acc_df = pd.read_csv(f'{dir}/empatica/ACC.csv')
    acc_time_points = np.arange(0, len(empatica_acc_df) / EMPATICA_ACC_SAMPLING_FREQUENCY, 1 / EMPATICA_ACC_SAMPLING_FREQUENCY)
    empatica_acc_df['Label'] = create_labels(acc_time_points, phases)
    data_dict['empatica']['acc'] = empatica_acc_df

    # BVP
    empatica_bvp_df = pd.read_csv(f'{dir}/empatica/BVP.csv')
    bvp_time_points = np.arange(0, len(empatica_bvp_df) / EMPATICA_BVP_SAMPLING_FREQUENCY, 1 / EMPATICA_BVP_SAMPLING_FREQUENCY)
    empatica_bvp_df['Label'] = create_labels(bvp_time_points, phases)
    data_dict['empatica']['bvp'] = empatica_bvp_df

    # HR
    empatica_hr_df = pd.read_csv(f'{dir}/empatica/HR.csv')
    hr_time_points = np.arange(0, len(empatica_hr_df) / EMPATICA_HR_SAMPLING_FREQUENCY, 1 / EMPATICA_HR_SAMPLING_FREQUENCY)
    empatica_hr_df['Label'] = create_labels(hr_time_points, phases)
    data_dict['empatica']['hr'] = empatica_hr_df

    # TEMP
    empatica_temp_df = pd.read_csv(f'{dir}/empatica/TEMP.csv')
    temp_time_points = np.arange(0, len(empatica_temp_df) / EMPATICA_TEMP_SAMPLING_FREQUENCY, 1 / EMPATICA_TEMP_SAMPLING_FREQUENCY)
    empatica_temp_df['Label'] = create_labels(temp_time_points, phases)
    data_dict['empatica']['temp'] = empatica_temp_df

    # EDA
    empatica_eda_df = pd.read_csv(f'{dir}/empatica/EDA.csv')
    eda_time_points = np.arange(0, len(empatica_eda_df) / EMPATICA_EDA_SAMPLING_FREQUENCY, 1 / EMPATICA_EDA_SAMPLING_FREQUENCY)
    empatica_eda_df['Label'] = create_labels(eda_time_points, phases)
    data_dict['empatica']['eda'] = empatica_eda_df

    # Quattrocento Data
    # EMG
    emg_df = pd.read_csv(f'{dir}/quattrocento/EMG.csv', delimiter=';')
    emg_time_points = np.arange(0, len(emg_df) / EMG_SAMPLING_FREQUENCY, 1 / EMG_SAMPLING_FREQUENCY)
    emg_df['Label'] = create_labels(emg_time_points, phases)
    upper_trapezius_df = emg_df[['Upper Trapezius', 'Label']].copy()
    mastoid_df = emg_df[['Mastoid', 'Label']].copy()
    data_dict['quattrocento']['emg_upper_trapezius'] = upper_trapezius_df
    data_dict['quattrocento']['emg_mastoid'] = mastoid_df

    # Myndsens Data
    # FNIRS
    fnirs_df = pd.read_csv(f'{dir}/myndsens/FNIRS.csv')
    fnirs_time_points = np.arange(0, len(fnirs_df) / FNIRS_SAMPLING_FREQUENCY, 1 / FNIRS_SAMPLING_FREQUENCY)
    fnirs_df['Label'] = create_labels(fnirs_time_points, phases)
    data_dict['myndsens']['fnirs'] = fnirs_df

    # Save the dictionary to a pickle file
    with open(f'src/mused/dataset/S{subject_id}/S{subject_id}.pkl', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Data saved successfully in 'src/mused/dataset/S{subject_id}/S{subject_id}.pkl'")

def preprocess_dataset():
    for subject_id in range(1, 19):
            dir = f'src/mused/dataset/S{subject_id}'
            print(f"Creating pickle for subject {subject_id}")
            create_subject_pickle_file(subject_id, dir)