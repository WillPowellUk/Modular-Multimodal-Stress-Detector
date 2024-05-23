# import pandas as pd
# import matplotlib.pyplot as plt

# # File paths
# file_path_raw = 'src/wesad/WESAD/raw/merged_chest.pkl'
# file_path_cleaned = 'src/wesad/WESAD/cleaned/chest_preprocessed.pkl'

# # Read data from pickle files
# df_raw = pd.read_pickle(file_path_raw)
# df_cleaned = pd.read_pickle(file_path_cleaned)

# # Plotting the raw data
# plt.figure(figsize=(10, 6))
# plt.plot(df_raw['ecg'][:70000])
# plt.title('Raw Chest Data')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend(loc='best')
# plt.show(block=False)

# # Plotting the cleaned data
# plt.figure(figsize=(10, 6))
# plt.plot(df_cleaned['ecg'][:70000])
# plt.title('Cleaned Chest Data')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.legend(loc='best')
# plt.show()


from src.ml_pipeline.preprocessing import SignalPreprocessor

# preprocess the chest data
signal_preprocessor = SignalPreprocessor('src/wesad/WESAD/raw/merged_chest.pkl', 'src/wesad/WESAD/cleaned/chest_preprocessed.pkl', 'src/wesad/wesad_configuration.json')
signal_preprocessor.preprocess_signals()

# preprocess the wrist data
signal_preprocessor = SignalPreprocessor('src/wesad/WESAD/raw/merged_wrist.pkl', 'src/wesad/WESAD/cleaned/wrist_preprocessed.pkl', 'src/wesad/wesad_configuration.json', wrist=True)
signal_preprocessor.preprocess_signals()