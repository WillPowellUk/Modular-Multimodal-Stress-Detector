import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os

two_level_label_mapping = {
    1: 0,  # Merging original labels 1 and 2 into class 0
    2: 0,
    3: 1,  # Merging original labels 3, 4, and 5 into class 1
    4: 1,
    5: 1
}

def load_data(subject_ids, base_path, selected_sensors):
    """
    Load data from the specified pickle files and filter based on selected sensors.
    
    Parameters:
    - subject_ids: list of integers representing subject IDs.
    - base_path: the base path where the data files are located.
    - selected_sensors: dictionary specifying which sensors and signals to load.

    Returns:
    - data: dictionary with subject_id as keys and filtered data as values.
    - feature_dict: dictionary with sensor names as keys and corresponding signal feature names as values.
    """
    data = {}
    feature_dict = {}

    for subject_id in subject_ids:
        file_path = os.path.join(base_path, f"S{subject_id}", f"S{subject_id}_features.pkl")

        with open(file_path, 'rb') as f:
            pkl_data = pickle.load(f)

        # Load all data without filtering by is_augmented
        collected_data = []
        for sensor, signals in selected_sensors.items():
            # Initialize the dictionary for the sensor if it doesn't exist
            if sensor not in feature_dict:
                feature_dict[sensor] = {}  # Initialize an empty dictionary for this sensor
            
            for signal in signals:
                if sensor in pkl_data and signal in pkl_data[sensor]:
                    df = pkl_data[sensor][signal].copy()

                    # Identify non-feature columns to be ignored in the renaming
                    non_feature_columns = ['Label', 'is_augmented']

                    # Update feature names by appending sensor and signal names, excluding non-feature columns
                    new_column_names = {col: f"{sensor}_{signal}_{col}" for col in df.columns if col not in non_feature_columns}
                    df.rename(columns=new_column_names, inplace=True)

                    # Assign updated features to feature_dict[sensor] using the correct signal key
                    feature_dict[sensor][signal] = [new_column_names.get(col, col) for col in df.columns if col not in non_feature_columns]
                    collected_data.append(df)

        # Concatenate all dataframes for a single subject after cropping to minimum length
        if collected_data:
            # Find the minimum number of rows across all dataframes
            min_rows = min(df.shape[0] for df in collected_data)
            
            # Trim each dataframe to the minimum number of rows
            trimmed_data = [df.iloc[:min_rows, :] for df in collected_data]
            
            # Concatenate trimmed dataframes along columns
            subject_data = pd.concat(trimmed_data, axis=1)

            # Normalize the subject data, excluding 'Label' and 'is_augmented' columns
            feature_columns = [col for col in subject_data.columns if col not in ['Label', 'is_augmented']]
            normalized_data = subject_data[feature_columns].copy()

            # Add back non-feature columns without normalization
            normalized_data['Label'] = subject_data['Label'].iloc[:, 0].values
            normalized_data['is_augmented'] = subject_data['is_augmented'].iloc[:, 0].values
            
            data[subject_id] = normalized_data

    return data, feature_dict


def plot_features(subject_data, feature_dict, label_mapping):
    """
    Plot box plots for each feature grouped by sensor and split into 'Non-stress' and 'Stress' categories.
    
    Parameters:
    - subject_data: dictionary with subject_id as keys and data as values.
    - feature_dict: dictionary with device, sensor, and feature names.
    - label_mapping: dictionary for converting original labels to binary stress/non-stress labels.
    """

    # Prepare a list to store all features data
    feature_data = pd.concat(subject_data.values(), ignore_index=True)

    # Map the original labels to binary labels (0 for non-stress, 1 for stress)
    feature_data['BinaryLabel'] = feature_data['Label'].map(label_mapping)

    # Iterate through each device and sensor in feature_dict
    for device, sensors in feature_dict.items():
        for sensor, features in sensors.items():
            # Create a new figure for each sensor
            num_features = len(features)
            fig, axs = plt.subplots(1, num_features, figsize=(5 * num_features, 6))

            # Ensure axs is a list even when there's only one plot
            if num_features == 1:
                axs = [axs]

            # Extract data for the current sensor's features
            for j, feature in enumerate(features):
                sns.boxplot(
                    x='BinaryLabel', 
                    y=feature, 
                    data=feature_data, 
                    ax=axs[j], 
                    showfliers=False  # This removes the data point circles (outliers)
                )
                axs[j].set_title(f"Feature: {feature}")
                axs[j].set_ylabel("Values")
                axs[j].set_xlabel("")  # Remove the x label
                axs[j].set_xticks([0, 1])
                axs[j].set_xticklabels(['No-Stress', 'Stress'])  # Replace 0 and 1 with 'No-Stress' and 'Stress'

            # Set the overall title for each sensor plot
            plt.suptitle(f"Device: {device}, Sensor: {sensor}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()



def main():
    # Define all sensor groups
    selected_sensors = {
        "empatica": ["bvp", "eda", "temp", "acc"],
        "polar": ["ecg", "ibi", "acc"],
        "quattrocento": ["emg_upper_trapezius", "emg_mastoid"],
        "myndsens": ["fnirs"]
    }

    base_path = "src/mused/dataset/dataset_features"

    # Load and normalize data
    subjects = range(1, 19)
    subject_data, feature_dict = load_data(subjects, base_path, selected_sensors)

    feature_dict = {
    'empatica': {
        'bvp': ['empatica_bvp_HRV_MeanNN', 'empatica_bvp_HRV_SDNN', 'empatica_bvp_HRV_RMSSD', 'empatica_bvp_HRV_LnHF'],
        'eda': ['empatica_eda_mean_SCL', 'empatica_eda_num_SCR', 'empatica_eda_sum_amp_SCR'],
        'temp': ['empatica_temp_mean_temp', 'empatica_temp_std_temp'],
        'acc': ['empatica_acc_mean_acc_all']
    },
    'polar': {
        'ecg': ['polar_ecg_HRV_MeanNN', 'polar_ecg_HRV_SDNN', 'polar_ecg_HRV_RMSSD', 'polar_ecg_HRV_LnHF']
    }
}


    print("Plotting feature data")

    # Call the plot function
    plot_features(subject_data, feature_dict, two_level_label_mapping)


if __name__ == "__main__":
    main()
