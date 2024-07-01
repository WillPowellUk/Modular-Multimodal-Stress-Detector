import pickle
import os


def save_var(variable, filename, var_name=None):
    """
    Save a variable to a file using pickle, creating parent directories if they do not exist.

    Args:
    variable: The variable to save.
    filename: The name of the file where the variable will be saved.
    var_name: The name of the variable (as a string) to be printed (optional).
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the variable to the file
    with open(filename, "wb") as file:
        pickle.dump(variable, file)

    if var_name:
        print(f"Variable '{var_name}' saved to {filename}")
    else:
        print(f"Variable saved to {filename}")


def load_var(filename):
    """
    Load a variable from a file using pickle.

    Args:
    filename: The name of the file to load the variable from.

    Returns:
    The variable loaded from the file.
    """
    with open(filename, "rb") as file:
        variable = pickle.load(file)
    print(f"Variable loaded from {filename}")
    return variable
