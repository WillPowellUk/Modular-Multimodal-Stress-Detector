import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subject_id = 1
acc_df = pd.read_csv(rf'data_collection\recordings\S{subject_id}\empatica\ACC.csv')

acc_df.columns = ['x', 'y', 'z']

# Sampling rate
sample_rate = 32  # in Hz

# Calculate time vector
time = acc_df.index / sample_rate

# Create a figure with three subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plot x-axis data
axs[0].plot(time, acc_df['x'], label='x-axis')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('ACC Signal')
axs[0].set_title('ACC Signal - X Axis')
axs[0].grid(True)
for marker in [40.3, 390.9, 1000.3]:
    axs[0].axvline(x=marker, color='r', linestyle='--', label=f'Marker at {marker}s')

# Plot y-axis data
axs[1].plot(time, acc_df['y'], label='y-axis')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('ACC Signal')
axs[1].set_title('ACC Signal - Y Axis')
axs[1].grid(True)
for marker in [40.3, 390.9, 1000.3]:
    axs[1].axvline(x=marker, color='r', linestyle='--', label=f'Marker at {marker}s')

# Plot z-axis data
axs[2].plot(time, acc_df['z'], label='z-axis')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('ACC Signal')
axs[2].set_title('ACC Signal - Z Axis')
axs[2].grid(True)
for marker in [40.3, 390.9, 1000.3]:
    axs[2].axvline(x=marker, color='r', linestyle='--', label=f'Marker at {marker}s')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
