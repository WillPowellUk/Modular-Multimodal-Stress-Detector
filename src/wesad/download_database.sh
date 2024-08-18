#!/bin/bash

# Define the URL of the WESAD database
WESAD_URL="https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

# Define the name of the zip file
ZIP_FILE="wesad/wesad.zip"

# Download the WESAD database
echo "Downloading WESAD database..."
wget -O $ZIP_FILE $WESAD_URL

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful. Unzipping the file..."
    
    # Unzip the file
    unzip $ZIP_FILE
    
    # Check if the unzip was successful
    if [ $? -eq 0 ]; then
        echo "Unzip successful. Deleting the zip file..."
        
        # Delete the zip file
        rm $ZIP_FILE
        
        if [ $? -eq 0 ]; then
            echo "Zip file deleted successfully."
        else
            echo "Error: Failed to delete the zip file."
        fi
    else
        echo "Error: Failed to unzip the file."
    fi
else
    echo "Error: Failed to download the file."
fi
