import os

# Define the base directory containing the folders S1 to S20
base_dir = ''  # Replace with the path to your directory

# List all the folders (S1 to S20) in the base directory
folders = [f"S{i}" for i in range(1, 21)]

# Define the files that need to be deleted
files_to_delete = ["panas.csv", "sssq.csv", "stai.csv", "timings.csv"]

# Loop through each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.exists(folder_path):
        # Loop through each file in the list
        for file_name in files_to_delete:
            file_to_delete = os.path.join(folder_path, file_name)
            
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                print(f"Deleted {file_to_delete}")
            else:
                print(f"File {file_to_delete} does not exist in {folder_path}")
    else:
        print(f"Folder {folder_path} does not exist")

print("Deletion operation completed.")
