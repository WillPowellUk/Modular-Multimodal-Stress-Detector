import pandas as pd
import neurokit2 as nk

# Read the ECG data from the CSV file
ecg_data = pd.read_csv(r"data_collection\recordings\S13\ECG.csv")

# Assuming the ECG data is in a column named 'ECG'
# If the column has a different name, replace 'ECG' with the appropriate column name
ecg_signal = ecg_data.values.flatten()

# Process the ECG signal to extract the R-peaks
signals, info = nk.ecg_process(ecg_signal, sampling_rate=130)  # Adjust sampling_rate if necessary

# Extract the R-peaks from the processed signal
rpeaks = info["ECG_R_Peaks"]

print("RR intervals have been computed and saved to RR.csv")
