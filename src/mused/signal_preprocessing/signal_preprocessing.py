import pandas as pd
import numpy as np
import os
import pickle
import inspect
from src.ml_pipeline.utils.utils import get_sampling_frequency
from src.ml_pipeline.preprocessing.acc_preprocessing import AccPreprocessing
from src.ml_pipeline.preprocessing.ecg_preprocessing import ECGPreprocessing
from src.ml_pipeline.preprocessing.bvp_preprocessing import BVPPreprocessing
from src.ml_pipeline.preprocessing.eda_preprocessing import EDAPreprocessing
from src.ml_pipeline.preprocessing.temp_preprocessing import TempPreprocessing
from src.ml_pipeline.preprocessing.resp_preprocessing import RespPreprocessing
from src.ml_pipeline.preprocessing.emg_preprocessing import EMGPreprocessing
from src.ml_pipeline.preprocessing.fnirs_preprocessing import FNIRSPreprocessing


class SignalPreprocessor:
    def __init__(self, data_path: str, config_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        self.config_path = config_path
        self.pkl_file = pd.read_pickle(self.data_path)

    def has_parameter(self, cls, parameter):
        # Get the __init__ method of the class
        init_signature = inspect.signature(cls.__init__)
        # Check if the parameter is in the method's parameters
        return parameter in init_signature.parameters

    def process_data(self, source, sensor, processor_class, output, name=None):
        """
        Process data from a specified source, sensor, and name using a given processor class.

        Parameters:
        - source: str, the source key (e.g., 'quattrocento')
        - sensor: str, the sensor key (e.g., 'emg_upper_trapezius')
        - processor_class: class, a class with a .process() method to process the EMG data
        - output: dict, the dictionary where the processed output will be stored
        - name: str, the name of the data (e.g., 'Upper Trapezius'), defaults to None

        Returns:
        - None: the function modifies the output dictionary in place
        """
        print(f"Processing {sensor}...")

        # Retrieve the data from the pkl_file using the given parameters
        if name == None:
            data = self.pkl_file[source][sensor]
        else:
            data = self.pkl_file[source][sensor][name]
        
        # Get the sampling frequency from the configuration file
        fs = get_sampling_frequency(self.config_path, source, sensor)
        
        # Initialize the processing object using the provided processor class
        if self.has_parameter(processor_class, 'fs'):
            processor = processor_class(data, fs=fs)
        else:
            processor = processor_class(data)
        
        # Process the data using the .process() method of the processor object
        processed_data = processor.process()

        # Create dataframe
        processed_data = pd.DataFrame(processed_data)
        if name is not None:
            processed_data.columns = [name]

        # Add label back to each one
        processed_data['Label'] = self.pkl_file[source][sensor]['Label']
        
        # Store the processed data in the output dictionary
        self.populate_output(output, source, sensor, processed_data)
        
        print(f"{sensor} processing completed.")

    def preprocess_signals(self):
        print("Starting signal preprocessing...")

        output = {}

        # Quattrocento Data
        self.process_data('quattrocento', 'emg_upper_trapezius', EMGPreprocessing, output, name='Upper Trapezius')
        self.process_data('quattrocento', 'emg_mastoid', EMGPreprocessing, output, name='Mastoid')

        # Polar Data
        self.process_data('polar', 'acc', AccPreprocessing, output)
        self.process_data('polar', 'ecg', ECGPreprocessing, output, name='ECG')
        self.populate_output(output, 'polar', 'ibi', self.pkl_file['polar']['ibi'])

        # Empatica Data
        self.process_data('empatica', 'acc', AccPreprocessing, output)
        self.process_data('empatica', 'bvp', BVPPreprocessing, output, name='BVP')
        self.process_data('empatica', 'temp', TempPreprocessing, output, name='TEMP')
        self.process_data('empatica', 'eda', EDAPreprocessing, output, name='EDA')

        # Myndsens
        self.process_data('myndsens', 'fnirs', FNIRSPreprocessing, output)

        # Save preprocessed data
        print("Saving cleaned data...")
        self.save_preprocessed_data(output, self.output_path)
        print("Preprocessed data saved successfully.")

    def populate_output(self, output, source, sensor, value):
        """
        Populates the nested dictionary structure for output[source][sensor] with the specified value.
        
        Parameters:
        - output (dict): The main dictionary to populate.
        - source (str): The key for the source level.
        - sensor (str): The key for the sensor level.
        - value (any): The value to assign at the output[source][sensor] location.
        
        Returns:
        - None: the function modifies the output dictionary in place
        """

        # Check if the 'source' key exists in the output dictionary
        if source not in output:
            output[source] = {}  # If not, create a new dictionary for the source

        # Check if the 'sensor' key exists under the 'source' key in the output dictionary
        if sensor not in output[source]:
            output[source][sensor] = {}  # If not, create a new dictionary for the sensor

        # Finally, set the value for the 'name' key under 'sensor' and 'source'
        output[source][sensor] = value

        # Return the updated output dictionary
        return output

    def save_preprocessed_data(self, output, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(output, f)
        print(f"Data saved to {output_path}")


if __name__ == "__main__":
    for subject_id in range(1, 4):
        data_path = f"src/mused/dataset/S{subject_id}/S{subject_id}.pkl"
        config_path = "config_files/dataset/mused_configuration.json"
        output_path = f"src/mused/dataset/S{subject_id}/S{subject_id}_cleaned.pkl"
        preprocessor = SignalPreprocessor(data_path, config_path, output_path)
        preprocessor.preprocess_signals()