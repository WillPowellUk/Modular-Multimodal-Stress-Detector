import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subject_id = 3

# Load the CSV files
emg_df1 = pd.read_csv(rf'data_collection\recordings\S{subject_id}\quattrocento\EMG-1.csv', delimiter=';')
emg_df2 = pd.read_csv(rf'data_collection\recordings\S{subject_id}\quattrocento\EMG-2.csv', delimiter=';')

# Concatenate the two DataFrames vertically
emg_new = pd.concat([emg_df1, emg_df2], axis=0, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
emg_new.to_csv(rf'data_collection\recordings\S{subject_id}\quattrocento\EMG-NEW.csv', index=False, sep=';')

print("EMG-1 and EMG-2 have been concatenated and saved to EMG-NEW.csv")
