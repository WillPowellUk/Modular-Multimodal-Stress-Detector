import pandas as pd
import numpy as np
from scipy.integrate import simpson


subject_id = 1
EMG_crop_timestamp = 0
ECG_crop_timestamp = 0
FNIRS_crop_timestamp = 0
_crop_timestamp = 0


subject_dir = f'data_collection/recordings/S{subject_id}'

# extract timings
timings = pd.read_csv(f'{subject_dir}/Phases.csv', sep=';')
# Extract the Baseline sit, anticipation and goodbye times
baseline_sit_timestamp = timings.loc[timings['PhaseName'] == 'Baseline sit', 'StartTime'].values[0]
anticiaption_timestamp = timings.loc[timings['PhaseName'] == 'TSST1 Interview Instruction', 'StartTime'].values[0]
goodbye_timestamp = timings.loc[timings['PhaseName'] == 'Goodbye', 'StartTime'].values[0]

print(f"Baseline sit time: {baseline_sit_timestamp}")
print(f"TSST1 Interview Instruction time: {baseline_sit_timestamp}")
print(f"Goodbye time: {goodbye_timestamp}")
duration = goodbye_timestamp - baseline_sit_timestamp
duration_baseline = anticiaption_timestamp - baseline_sit_timestamp
duration_stress = goodbye_timestamp - anticiaption_timestamp
print(f'Duration of experiment: {duration}s')
print(f'Duration of non-stress: {duration_baseline}s')
print(f'Duration of stress: {duration_stress}s')

# Crop EMG
