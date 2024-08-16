import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

emg_df = pd.read_csv(r'data_collection\recordings\S2\quattrocento\EMG.csv', delimiter=';')
emg_df.columns = ['Upper Trapezius', 'Mastoid']

# Sampling rate
sample_rate = 2048  # in Hz

# Calculate time vector
time = emg_df.index / sample_rate

# Plot the EMG signal
plt.figure(figsize=(12, 6))
plt.plot(time, emg_df['Upper Trapezius'], label='Upper Trapezius')
plt.plot(time, emg_df['Mastoid'], label='Mastoid')

# Add vertical markers at the specified timeframes
markers = [40.3, 390.9, 1000.3]
for marker in markers:
    plt.axvline(x=marker, color='r', linestyle='--', label=f'Marker at {marker}s')

# Add labels and legend
plt.xlabel('Time (seconds)')
plt.ylabel('EMG Signal')
plt.title('EMG Signal with Vertical Markers')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
