import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subject_id = 1
emg_df = pd.read_csv(rf'data_collection\recordings\S{subject_id}\quattrocento\EMG.csv', delimiter=';')
emg_df.columns = ['Time', 'Upper Trapezius', 'Mastoid']

# Sampling rate
sample_rate = 2048  # in Hz

# Calculate time vector
time = emg_df.index / sample_rate

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Plot Upper Trapezius data
axs[0].plot(time, emg_df['Upper Trapezius'], label='Upper Trapezius')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('EMG Signal')
axs[0].set_title('EMG Signal - Upper Trapezius')
axs[0].grid(True)
for marker in [40.3, 390.9, 1000.3]:
    axs[0].axvline(x=marker, color='r', linestyle='--', label=f'Marker at {marker}s')

# Plot Mastoid data
axs[1].plot(time, emg_df['Mastoid'], label='Mastoid')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('EMG Signal')
axs[1].set_title('EMG Signal - Mastoid')
axs[1].grid(True)
for marker in [40.3, 390.9, 1000.3]:
    axs[1].axvline(x=marker, color='r', linestyle='--', label=f'Marker at {marker}s')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
