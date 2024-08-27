import pandas as pd
import numpy as np
import pyxdf
import os

total_duration = 0
total_duration_baseline = 0
total_duration_baseline_sit = 0
total_duration_baseline_stand = 0
total_duration_anticipation = 0
total_duration_stress = 0
total_arithmetic = 0
total_interview = 0

for subject_id in range(1, 19):
    crop_times = f"data_collection/crop_times/S{subject_id}.txt"
    times_df = pd.read_csv(crop_times, sep=',', header=None)

    EMG_crop_timestamp = times_df[1][0]
    ECG_crop_timestamp = times_df[1][1]
    FNIRS_crop_timestamp = times_df[1][2]
    EMPATICA_crop_timestamp = times_df[1][3]


    subject_dir = f'data_collection/recordings/S{subject_id}'

    # extract timings
    timings = pd.read_csv(f'{subject_dir}/Phases.csv', sep=';')

    baseline_sit_timestamp = timings.loc[timings['PhaseName'] == 'Baseline sit', 'StartTime'].values[0]
    baseline_stand_timestamp = timings.loc[timings['PhaseName'] == 'Baseline stand', 'StartTime'].values[0] - baseline_sit_timestamp
    anticiaption_timestamp = timings.loc[timings['PhaseName'] == 'TSST1 Interview Instruction', 'StartTime'].values[0] - baseline_sit_timestamp
    interview_timestamp = timings.loc[timings['PhaseName'] == 'TSST1 Job Interview', 'StartTime'].values[0] - baseline_sit_timestamp
    arthmetic_timestamp = timings.loc[timings['PhaseName'] == 'TSST2 Subtraction', 'StartTime'].values[0] - baseline_sit_timestamp
    goodbye_timestamp = timings.loc[timings['PhaseName'] == 'Goodbye', 'StartTime'].values[0] - baseline_sit_timestamp
    baseline_sit_timestamp = 0

    print(f"baseline_sit_timestamp = {baseline_sit_timestamp}")
    print(f"baseline_stand_timestamp = {baseline_stand_timestamp}")
    print(f"anticiaption_timestamp = {anticiaption_timestamp}")
    print(f"interview_timestamp = {interview_timestamp}")
    print(f"arthmetic_timestamp = {arthmetic_timestamp}")
    print(f"goodbye_timestamp = {goodbye_timestamp}")

    duration = goodbye_timestamp - baseline_sit_timestamp
    duration_baseline_sit = baseline_stand_timestamp - baseline_sit_timestamp
    duration_baseline_stand = anticiaption_timestamp - baseline_stand_timestamp
    duration_anticipation = interview_timestamp - anticiaption_timestamp
    duration_interview = arthmetic_timestamp - interview_timestamp
    duration_arithmetic = goodbye_timestamp - arthmetic_timestamp

    duration_baseline = anticiaption_timestamp - baseline_sit_timestamp
    duration_stress = goodbye_timestamp - anticiaption_timestamp
    print(f'duration_experiment = {duration}')
    print(f'duration_non_stress = {duration_baseline}')
    print(f'duration_stress = {duration_stress}')

    total_duration += duration
    total_duration_baseline += duration_baseline
    total_duration_stress += duration_stress
    total_duration_baseline_sit += duration_baseline_sit
    total_duration_baseline_stand += duration_baseline_stand
    total_duration_anticipation += duration_anticipation
    total_interview += duration_interview
    total_arithmetic += duration_arithmetic

    # Crop EMG
    EMG_SAMPLING_FREQUENCY = 2048  # in Hz
    emg_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/quattrocento/EMG.csv', delimiter=';')
    emg_df = emg_df.iloc[:, 1:]
    emg_segment = emg_df.values[int(EMG_crop_timestamp * EMG_SAMPLING_FREQUENCY):]
    emg_segment = emg_segment[:int(goodbye_timestamp*EMG_SAMPLING_FREQUENCY)]
    emg_df = pd.DataFrame(emg_segment)
    emg_df.columns = ['Upper Trapezius', 'Mastoid']
    if not os.path.exists(f'src/mused/dataset/S{subject_id}/quattrocento/'):
        os.makedirs(f'src/mused/dataset/S{subject_id}/quattrocento/')
    emg_df.to_csv(f'src/mused/dataset/S{subject_id}/quattrocento/EMG.csv', index=False, sep=';')

    # Crop Polar HR
    rr_intervals = pd.read_csv(f'data_collection/recordings/S{subject_id}/polar/HR.csv', header=None)
    rr_intervals = rr_intervals[0].values/1000
    cum_intervals = np.cumsum(rr_intervals)
    rr_intervals = cum_intervals[int(ECG_crop_timestamp * 1):]
    rr_intervals = rr_intervals[:int(goodbye_timestamp*1)]
    rr_intervals = np.diff(rr_intervals)
    rr_intervals = np.round(rr_intervals, 5)
    if not os.path.exists(f'src/mused/dataset/S{subject_id}/polar/'):
        os.makedirs(f'src/mused/dataset/S{subject_id}/polar/')
    rr_intervals_df = pd.DataFrame(rr_intervals)
    rr_intervals_df.to_csv(f'src/mused/dataset/S{subject_id}/polar/IBI.csv', index=False, header=['IBI'])

    # Crop Polar ECG
    ECG_SAMPLING_FREQUENCY = 130  # in Hz
    emg_data = np.loadtxt(f'data_collection/recordings/S{subject_id}/polar/ECG.csv', delimiter=',')
    ecg_segment = emg_data[int(ECG_crop_timestamp * ECG_SAMPLING_FREQUENCY):]
    ecg_segment = ecg_segment[:int(goodbye_timestamp*ECG_SAMPLING_FREQUENCY)]
    ecg_segment = ecg_segment.astype(int)
    ecg_df = pd.DataFrame(ecg_segment, columns=['ECG'])
    ecg_df.to_csv(f'src/mused/dataset/S{subject_id}/polar/ECG.csv', index=False)

    # Crop Polar ACC
    POLAR_ACC_SAMPLING_FREQUENCY = 25  # in Hz
    acc_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/polar/ACC.csv')
    acc_df.columns = ['x', 'y', 'z']
    start_index = int(ECG_crop_timestamp * POLAR_ACC_SAMPLING_FREQUENCY)
    end_index = int(goodbye_timestamp * POLAR_ACC_SAMPLING_FREQUENCY)
    acc_df = acc_df.iloc[start_index:]
    acc_df = acc_df[:end_index].reset_index(drop=True)
    acc_df.to_csv(f'src/mused/dataset/S{subject_id}/polar/ACC.csv', index=False, header=['x', 'y', 'z'])

    # Crop FNIRS
    FNIRS_SAMPLING_FREQUENCY = 10  # in Hz
    XDF_FILE = f'data_collection/recordings/S{subject_id}/myndsens/FNIRS.xdf'
    data, header = pyxdf.load_xdf(XDF_FILE)
    headers = ['O2Hb', 'HHb', 'Brain oxy', 'Brain state']
    fnirs_df = pd.DataFrame(data[0]['time_series'], columns=headers)
    fnirs_df.insert(0, 'Timestamp', data[0]['time_stamps'])
    start_time = fnirs_df['Timestamp'].min()
    end_time = fnirs_df['Timestamp'].max()
    new_time_points = np.arange(start_time, end_time, 1 / FNIRS_SAMPLING_FREQUENCY)
    fnirs_df_interpolated = pd.DataFrame({
        'O2Hb': np.interp(new_time_points, fnirs_df['Timestamp'], fnirs_df['O2Hb']),
        'HHb': np.interp(new_time_points, fnirs_df['Timestamp'], fnirs_df['HHb']),
        'Brain oxy': np.interp(new_time_points, fnirs_df['Timestamp'], fnirs_df['Brain oxy'])
    })
    fnirs_df_interpolated = fnirs_df_interpolated[int(FNIRS_crop_timestamp * FNIRS_SAMPLING_FREQUENCY):]
    fnirs_df_interpolated = fnirs_df_interpolated[:int(goodbye_timestamp * FNIRS_SAMPLING_FREQUENCY):].reset_index(drop=True)
    output_dir = f'src/mused/dataset/S{subject_id}/myndsens/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fnirs_df_interpolated.to_csv(f'{output_dir}/FNIRS.csv', index=False)

    # Crop EMPATICA ACC 
    EMPATICA_ACC_SAMPLING_FREQUENCY = 32  # in Hz
    acc_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/empatica/ACC.csv')
    acc_df.columns = ['x', 'y', 'z']
    start_index = int(EMPATICA_crop_timestamp * EMPATICA_ACC_SAMPLING_FREQUENCY)
    end_index = int(goodbye_timestamp * EMPATICA_ACC_SAMPLING_FREQUENCY)
    acc_df = acc_df.iloc[start_index:]
    acc_df = acc_df[:end_index]
    if not os.path.exists(f'src/mused/dataset/S{subject_id}/empatica/'):
        os.makedirs(f'src/mused/dataset/S{subject_id}/empatica/')
    acc_df.to_csv(f'src/mused/dataset/S{subject_id}/empatica/ACC.csv', index=False, header=['x', 'y', 'z'])

    # Crop EMPATICA BVP
    EMPATICA_BVP_SAMPLING_FREQUENCY = 64  # in Hz
    acc_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/empatica/BVP.csv')
    start_index = int(EMPATICA_crop_timestamp * EMPATICA_BVP_SAMPLING_FREQUENCY)
    end_index = int(goodbye_timestamp * EMPATICA_BVP_SAMPLING_FREQUENCY)
    acc_df = acc_df.iloc[start_index:]
    acc_df = acc_df[:end_index]
    acc_df.to_csv(f'src/mused/dataset/S{subject_id}/empatica/BVP.csv', index=False, header=['BVP'])

    # Crop EMPATICA HR
    EMPATICA_HR_SAMPLING_FREQUENCY = 1  # in Hz
    hr_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/empatica/HR.csv')
    start_index = int(EMPATICA_crop_timestamp * EMPATICA_HR_SAMPLING_FREQUENCY)
    end_index = int(goodbye_timestamp * EMPATICA_HR_SAMPLING_FREQUENCY)
    hr_df = hr_df.iloc[start_index:]
    hr_df = hr_df[:end_index]
    hr_df.to_csv(f'src/mused/dataset/S{subject_id}/empatica/HR.csv', index=False, header=['HR'])

    # Crop EMPATICA IBI
    rr_intervals = pd.read_csv(f'data_collection/recordings/S{subject_id}/empatica/IBI.csv')
    cropped_rr_intervals = rr_intervals[(rr_intervals.iloc[:, 0] >= EMPATICA_crop_timestamp) & (rr_intervals.iloc[:, 0] <= goodbye_timestamp + EMPATICA_crop_timestamp)]
    cropped_rr_intervals.iloc[:, 0] -= EMPATICA_crop_timestamp
    cropped_rr_intervals.columns = ['Timestamp', 'IBI']
    cropped_rr_intervals.to_csv(f'src/mused/dataset/S{subject_id}/empatica/IBI.csv', index=False)

    # Crop EMPATICA TEMP
    EMPATICA_TEMP_SAMPLING_FREQUENCY = 4  # in Hz
    temp_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/empatica/TEMP.csv')
    start_index = int(EMPATICA_crop_timestamp * EMPATICA_TEMP_SAMPLING_FREQUENCY)
    end_index = int(goodbye_timestamp * EMPATICA_TEMP_SAMPLING_FREQUENCY)
    temp_df = temp_df.iloc[start_index:]
    temp_df = temp_df[:end_index]
    temp_df.to_csv(f'src/mused/dataset/S{subject_id}/empatica/TEMP.csv', index=False, header=['TEMP'])

    # Crop EMPATICA EDA
    EMPATICA_EDA_SAMPLING_FREQUENCY = 4  # in Hz
    eda_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/empatica/EDA.csv')
    start_index = int(EMPATICA_crop_timestamp * EMPATICA_EDA_SAMPLING_FREQUENCY)
    end_index = int(goodbye_timestamp * EMPATICA_EDA_SAMPLING_FREQUENCY)
    eda_df = eda_df.iloc[start_index:]
    eda_df = eda_df[:end_index]
    eda_df.to_csv(f'src/mused/dataset/S{subject_id}/empatica/EDA.csv', index=False, header=['EDA'])

    # Shift
    phases_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/Phases.csv', delimiter=';')

    # Calculate the offset (the second StartTime value)
    phases_offset = phases_df['StartTime'].iloc[1]

    # Subtract the offset from all StartTime values in the Phases DataFrame
    phases_df['StartTime'] = phases_df['StartTime'] - phases_offset

    # Save the modified Phases DataFrame to a new CSV file
    phases_df.to_csv(f'src/mused/dataset/S{subject_id}/Phases.csv', index=False, sep=';')

    # Read the Actions.csv file
    actions_df = pd.read_csv(f'data_collection/recordings/S{subject_id}/Actions.csv', delimiter=';')

    # Subtract the same offset from all StartTime values in the Actions DataFrame
    actions_df['StartTime'] = actions_df['StartTime'] - phases_offset

    # Save the modified Actions DataFrame to a new CSV file
    actions_df.to_csv(f'src/mused/dataset/S{subject_id}/Actions.csv', index=False, sep=';')

    # Copy other files
    import shutil
    shutil.copy(f'data_collection/recordings/S{subject_id}/readme.txt', f'src/mused/dataset/S{subject_id}/readme.txt')
    shutil.copy(f'data_collection/recordings/S{subject_id}/empatica/info.txt', f'src/mused/dataset/S{subject_id}/empatica/info.txt')
    shutil.copy(f'data_collection/recordings/S{subject_id}/empatica/tags.csv', f'src/mused/dataset/S{subject_id}/empatica/tags.csv')
    if not os.path.exists(f'src/mused/dataset/S{subject_id}/questionnaires/'):
        os.makedirs(f'src/mused/dataset/S{subject_id}/questionnaires/')
    shutil.copy(f'data_collection/recordings/S{subject_id}/questionnaires/panas_post.csv', f'src/mused/dataset/S{subject_id}/questionnaires/panas_post.csv')
    shutil.copy(f'data_collection/recordings/S{subject_id}/questionnaires/panas_pre.csv', f'src/mused/dataset/S{subject_id}/questionnaires/panas_pre.csv')
    shutil.copy(f'data_collection/recordings/S{subject_id}/questionnaires/sssq_post.csv', f'src/mused/dataset/S{subject_id}/questionnaires/sssq_post.csv')
    shutil.copy(f'data_collection/recordings/S{subject_id}/questionnaires/stai_pre.csv', f'src/mused/dataset/S{subject_id}/questionnaires/stai_pre.csv')
    shutil.copy(f'data_collection/recordings/S{subject_id}/questionnaires/stai_post.csv', f'src/mused/dataset/S{subject_id}/questionnaires/stai_post.csv')


print(f"total_duration = {total_duration}")
print(f"total_duration_baseline = {total_duration_baseline}")
print(f"total_duration_stress = {total_duration_stress}")

print(f"total_duration_baseline_sit = {total_duration_baseline_sit}")
print(f"total_duration_baseline_stand = {total_duration_baseline_stand}")
print(f"total_duration_anticipation = {total_duration_anticipation}")
print(f"total_interview = {total_interview}")
print(f"total_arithmetic = {total_arithmetic}")
