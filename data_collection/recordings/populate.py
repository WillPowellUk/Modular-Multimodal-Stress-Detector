import os
import shutil

# Function to create directories and copy files
def setup_directories(base_path, start, end, files_to_copy):
    for i in range(start, end + 1):
        dir_name = f"S{i}"
        dir_path = os.path.join(base_path, dir_name)
        
        # Create main directory SX
        os.makedirs(dir_path, exist_ok=True)
        
        # Create subdirectories
        subdirectories = ['empatica', 'myndsens', 'polar', 'quattrocento', 'questionnaires']
        for subdirectory in subdirectories:
            os.makedirs(os.path.join(dir_path, subdirectory), exist_ok=True)
        
        # Copy specified files to the questionnaires folder
        for file_name in files_to_copy:
            shutil.copy(os.path.join(base_path, file_name), os.path.join(dir_path, 'questionnaires'))
        
        # Copy the readme.txt file to the SX directory
        shutil.copy(os.path.join(base_path, 'readme.txt'), dir_path)

# Define the base directory (replace with your actual base directory)
base_directory = ""

# List of files to copy to the questionnaires folder
files_to_copy = ['panas_post.csv', 'panas_pre.csv', 'sssq_post.csv', 'stai_pre.csv', 'stai_post.csv']

# Create directories from S3 to S20 and copy the files
setup_directories(base_directory, 3, 20, files_to_copy)

print("Directories and files have been set up successfully.")
