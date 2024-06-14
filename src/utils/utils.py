import pickle
import os

def save_var(variable, filename):
    """
    Save a variable to a file using pickle, creating parent directories if they do not exist.
    
    Args:
    variable: The variable to save.
    filename: The name of the file where the variable will be saved.
    """
    print(f"Saving variable to {filename}")
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the variable to the file
    with open(filename, 'wb') as file:
        pickle.dump(variable, file)
    print(f"Variable saved to {filename}")

def load_var(filename):
    """
    Load a variable from a file using pickle.
    
    Args:
    filename: The name of the file to load the variable from.
    
    Returns:
    The variable loaded from the file.
    """
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
    print(f"Variable loaded from {filename}")
    return variable
