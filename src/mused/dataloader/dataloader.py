import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
import pandas as pd

class SensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def normalize_data(data):
    """
    Normalize each feature in the DataFrame to have zero mean and unit variance.
    
    Parameters:
    - data: pandas DataFrame containing the data to be normalized.

    Returns:
    - normalized_data: pandas DataFrame with normalized features.
    """
    # Compute mean and standard deviation for each column
    mean = data.mean()
    std = data.std()

    # Normalize data
    normalized_data = (data - mean) / std
    return normalized_data

def load_data(subject_ids, base_path, selected_sensors):
    """
    Load data from the specified pickle files and filter based on selected sensors.
    
    Parameters:
    - subject_ids: list of integers representing subject IDs.
    - base_path: the base path where the data files are located.
    - selected_sensors: dictionary specifying which sensors and signals to load.

    Returns:
    - data: dictionary with subject_id as keys and filtered data as values.
    """
    data = {}

    for subject_id in subject_ids:
        file_path = os.path.join(base_path, f"S{subject_id}", f"S{subject_id}_features.pkl")

        with open(file_path, 'rb') as f:
            pkl_data = pickle.load(f)

        # Load all data without filtering by is_augmented
        collected_data = []
        for sensor, signals in selected_sensors.items():
            for signal in signals:
                if sensor in pkl_data and signal in pkl_data[sensor]:
                    df = pkl_data[sensor][signal]
                    collected_data.append(df)

        # Concatenate all dataframes for a single subject after cropping to minimum length
        if collected_data:
            # Find the minimum number of rows across all dataframes
            min_rows = min(df.shape[0] for df in collected_data)
            
            # Trim each dataframe to the minimum number of rows
            trimmed_data = [df.iloc[:min_rows, :] for df in collected_data]
            
            # Concatenate trimmed dataframes along columns
            subject_data = pd.concat(trimmed_data, axis=1)

            # Check for duplicate column names and remove duplicates if necessary
            subject_data = subject_data.loc[:, ~subject_data.columns.duplicated()]
            
            # Normalize the subject data
            normalized_data = normalize_data(subject_data.drop(columns=['Label', 'is_augmented']))

            # Add back non-feature columns
            normalized_data['Label'] = subject_data['Label'].values
            normalized_data['is_augmented'] = subject_data['is_augmented'].values
            
            data[subject_id] = normalized_data

    return data

def create_dataloader(subject_data, test_subject_id, batch_size=32):
    """
    Create DataLoader for training and testing data using LOSO cross-validation.
    
    Parameters:
    - subject_data: dictionary containing data for each subject.
    - test_subject_id: integer representing the subject ID for the test set.
    - batch_size: integer specifying batch size for the DataLoader.

    Returns:
    - train_loader: DataLoader for the training set.
    - test_loader: DataLoader for the test set.
    """
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for subject_id, data in subject_data.items():
        # Separate features and labels
        if subject_id == test_subject_id:
            # Filter out rows where 'is_augmented' is True
            data = data[data['is_augmented'] == False]
        else:
            # For the train set, retain all rows including where is_augmented is True
            pass

        features = data.drop(columns=['Label', 'is_augmented'])
        labels = data['Label'].values

        if subject_id == test_subject_id:
            test_data.append(features)  # Append DataFrame directly
            test_labels.extend(labels)
        else:
            train_data.append(features)  # Append DataFrame directly
            train_labels.extend(labels)

    # Ensure all dataframes have the same columns and trim them if necessary
    common_columns = set(train_data[0].columns)
    for df in train_data[1:]:
        common_columns.intersection_update(df.columns)
    for df in test_data:
        common_columns.intersection_update(df.columns)

    common_columns = list(common_columns)  # Convert set to list
    train_data = [df.loc[:, common_columns] for df in train_data]
    test_data = [df.loc[:, common_columns] for df in test_data]

    # Concatenate the list of DataFrames into a single DataFrame
    train_data = pd.concat(train_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)

    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data.values, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data.values, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create DataLoader
    train_dataset = SensorDataset(train_data, train_labels)
    test_dataset = SensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    subjects = range(1, 4)  # Example: 3 subjects
    base_path = "src/mused/dataset"
    selected_sensors = {
        "polar": ["acc", "ecg"],
        "empatica": ["bvp"],
        "myndsens": ["fnirs"]
    }
    
    # Load and normalize data
    subject_data = load_data(subjects, base_path, selected_sensors)

    # Leave-One-Subject-Out Cross-Validation
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(subjects):
        train_subjects = [subjects[i] for i in train_index]
        test_subject = subjects[test_index[0]]
        
        # Create dataloaders for the current split
        train_loader, test_loader = create_dataloader(subject_data, test_subject)

        # Example of using the DataLoader in training loop
        for epoch in range(1):
            for batch_data, batch_labels in train_loader:
                # Training loop logic here
                pass

            # Test loop logic here
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    # Testing loop logic here
                    pass

if __name__ == "__main__":
    main()
