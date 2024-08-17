import pandas as pd
import numpy as np
from scipy.integrate import simpson


subject_id = 1
EMG_crop_timestamp = 0
ECG_crop_timestamp = 0
FNIRS_crop_timestamp = 0
EMPATICA_crop_timestamp = 0


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
duration_baseline = anticiaption_timestamp - baseline_sit_timestamp
duration_stress = goodbye_timestamp - anticiaption_timestamp
print(f'duration_experiment = {duration}')
print(f'duration_non_stress = {duration_baseline}')
print(f'duration_stress = {duration_stress}')

# Crop EMG
