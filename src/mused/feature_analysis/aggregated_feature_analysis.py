import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
from src.mused.dataloader.dataloader import normalize_data

# Define label mappings for both classifications
two_level_label_mapping = {
    1: 0,  # Merging original labels 1 and 2 into class 0
    2: 0,
    3: 1,  # Merging original labels 3, 4, and 5 into class 1
    4: 1,
    5: 1
}

four_level_label_mapping = {
    1: 0, 
    2: 0,
    3: 1,
    4: 2,
    5: 3
}

def feature_importance(subject_data, feature_dict):
    # Initialize lists to store data
    X_four_class = []
    y_four_class = []
    X_binary_class = []
    y_binary_class = []

    # Iterate through each subject's data
    for subject_id, df in subject_data.items():
        # Map labels for four-level classification
        df['FourClassLabel'] = df['Label'].map(four_level_label_mapping)
        # Map labels for binary classification
        df['BinaryClassLabel'] = df['Label'].map(two_level_label_mapping)
        
        # Drop rows 'is_augmented' is True
        df = df[df['is_augmented'] == False]
        df = df.drop(columns=['is_augmented'])

        # Split features and labels for both tasks
        features = df.drop(columns=['Label', 'FourClassLabel', 'BinaryClassLabel'])
        
        # Convert all column names to strings to avoid mixed types
        features.columns = features.columns.astype(str)

        # Append to lists for model training
        X_four_class.append(features)
        y_four_class.append(df['FourClassLabel'])
        
        X_binary_class.append(features)
        y_binary_class.append(df['BinaryClassLabel'])

    # Concatenate all subjects' data for training
    X_four_class = pd.concat(X_four_class, ignore_index=True)
    y_four_class = pd.concat(y_four_class, ignore_index=True)
    X_binary_class = pd.concat(X_binary_class, ignore_index=True)
    y_binary_class = pd.concat(y_binary_class, ignore_index=True)

    # Convert all column names to strings for concatenated dataframes
    X_four_class.columns = X_four_class.columns.astype(str)
    X_binary_class.columns = X_binary_class.columns.astype(str)

    # Train RandomForestClassifier for four-class classification
    rf_four_class = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_four_class.fit(X_four_class, y_four_class)

    # Train RandomForestClassifier for binary classification
    rf_binary_class = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_binary_class.fit(X_binary_class, y_binary_class)

    # Extract feature importance for four-class classification
    feature_importances_four_class = rf_four_class.feature_importances_

    # Extract feature importance for binary classification
    feature_importances_binary_class = rf_binary_class.feature_importances_

    # Create DataFrames to display the results
    features_list = X_four_class.columns

    feature_importance_df_four_class = pd.DataFrame({
        'Feature': features_list,
        'Importance': feature_importances_four_class
    }).sort_values(by='Importance', ascending=False)

    feature_importance_df_binary_class = pd.DataFrame({
        'Feature': features_list,
        'Importance': feature_importances_binary_class
    }).sort_values(by='Importance', ascending=False)

    # Display the feature importance for four-class classification
    print("Feature Importance for Four-Class Classification:")
    print(feature_importance_df_four_class)

    # Display the feature importance for binary classification
    print("\nFeature Importance for Binary Classification:")
    print(feature_importance_df_binary_class)

    # Save the feature importance DataFrames to CSV files
    feature_importance_df_four_class.to_csv('feature_importance_four_class.csv', index=False)
    feature_importance_df_binary_class.to_csv('feature_importance_binary_class.csv', index=False)

    # Function to create a sensor mapping dictionary
    def create_sensor_mapping(feature_dict):
        sensor_mapping = {}
        for device, sensors in feature_dict.items():
            for sensor, features in sensors.items():
                for feature in features:
                    sensor_mapping[feature] = f"{device}_{sensor}"
        return sensor_mapping

    # Create a sensor mapping dictionary from the feature_dict
    sensor_mapping = create_sensor_mapping(feature_dict)

    # Function to compute the total importance per sensor
    def compute_total_importance(feature_importance_df, sensor_mapping):
        # Add a 'Sensor' column to the DataFrame based on the feature names
        feature_importance_df['Sensor'] = feature_importance_df['Feature'].map(sensor_mapping)
        
        # Group by 'Sensor' and sum the 'Importance' values
        sensor_importance = feature_importance_df.groupby('Sensor')['Importance'].sum().reset_index()
        
        # Rename columns for clarity
        sensor_importance.columns = ['Sensor', 'Total Importance']
        
        return sensor_importance

    # Compute the total importance per sensor for both DataFrames
    sensor_importance_four_class = compute_total_importance(feature_importance_df_four_class, sensor_mapping)
    sensor_importance_binary_class = compute_total_importance(feature_importance_df_binary_class, sensor_mapping)

    # Display the results
    print("Total Importance per Sensor for Four-Class Classification:")
    print(sensor_importance_four_class)

    print("\nTotal Importance per Sensor for Binary Classification:")
    print(sensor_importance_binary_class)

    # Save the results to CSV files
    sensor_importance_four_class.to_csv('sensor_importance_four_class.csv', index=False)
    sensor_importance_binary_class.to_csv('sensor_importance_binary_class.csv', index=False)

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
            normalized_data = normalize_data(subject_data[feature_columns])

            # Add back non-feature columns without normalization
            normalized_data['Label'] = subject_data['Label'].iloc[:, 0].values
            normalized_data['is_augmented'] = subject_data['is_augmented'].iloc[:, 0].values
            
            data[subject_id] = normalized_data

    return data, feature_dict



def main():

    # Define all sensor groups
    selected_sensors = {
        "empatica": ["bvp", "eda", "temp", "acc"],
        "polar": ["ecg", "ibi", "acc"],
        "quattrocento": ["emg_upper_trapezius", "emg_mastoid"],
        "myndsens": ["fnirs"]
    }

    # Create different configurations of sensors
    configurations = [
        {"description": "all_sensors", "sensors": selected_sensors},
    ]

    base_path = "src/mused/dataset/dataset_features"

    # Load and normalize data
    subjects = range(1, 19)
    subject_data, feature_dict = load_data(subjects, base_path, selected_sensors)

    print("Running feature analysis")

    feature_importance(subject_data, feature_dict)


if __name__ == "__main__":
    main()
