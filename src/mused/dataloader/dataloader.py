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

        # Filter the data based on selected sensors
        filtered_data = []
        for sensor, signals in selected_sensors.items():
            for signal in signals:
                if sensor in pkl_data and signal in pkl_data[sensor]:
                    df = pkl_data[sensor][signal]
                    # Consider only the rows where is_augmented is False
                    df_filtered = df[df['is_augmented'] == False]
                    filtered_data.append(df_filtered)

        # Concatenate all dataframes for a single subject
        if filtered_data:
            subject_data = pd.concat(filtered_data, axis=1)
            data[subject_id] = subject_data

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
        features = data.drop(columns=['Label', 'is_augmented'])
        labels = data['Label'].values

        if subject_id == test_subject_id:
            test_data.append(features.values)
            test_labels.extend(labels)
        else:
            train_data.append(features.values)
            train_labels.extend(labels)

    # Convert to PyTorch tensors
    train_data = torch.tensor(pd.concat(train_data).values, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(pd.concat(test_data).values, dtype=torch.float32)
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
    
    # Load and filter data
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
