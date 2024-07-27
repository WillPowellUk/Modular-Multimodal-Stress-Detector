import os
import statistics
import sys
from datetime import datetime
import torch

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the desired directory
target_dir = current_dir
while "src" not in os.listdir(target_dir) and target_dir != os.path.dirname(target_dir):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "src" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError("Could not find 'src' directory in the path hierarchy")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the desired directory
target_dir = current_dir
while "experiments" not in os.listdir(target_dir) and target_dir != os.path.dirname(
    target_dir
):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "experiments" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError("Could not find 'experiments' directory in the path hierarchy")

from src.ml_pipeline.models.attention_models.co_attention_models import MOSCAN
from src.ml_pipeline.utils import print_model_summary
from src.ml_pipeline.utils.utils import load_json
import time

MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"
config = load_json(MOSCAN_CONFIG)
device = 'cpu'
embedding_dim = 16
cache = True

def measure_model(model, inputs, device):
        inference_times = []
        with torch.no_grad():
            for i in range(10):
                model(inputs) # Warm-up
            for i in range(100):
                # Measure inference time for each batch
                if device == "cuda":
                    torch.cuda.synchronize()  # Synchronize CUDA operations before starting the timer
                start_time = time.time()
                model(inputs)
                if device == "cuda":
                    torch.cuda.synchronize()  # Synchronize CUDA operations after model inference
                end_time = time.time()

                inference_times.append(
                    (end_time - start_time) * 1000
                )  # Convert to milliseconds


        average_time = sum(inference_times) / len(inference_times)
        std_deviation = statistics.stdev(inference_times)

        print(f"Average inference time (ms): {average_time:.5f}, Standard deviation (ms): {std_deviation:.5f}")

        print("================================================================")

    

# Testing 1-10 modalities
for n in range(1,11):
    print("\n Number of modalities: ", n)

    input_dims = {}
    active_sensors = []
    inputs = {}
    for i in range(n):
        input_dims[str(i)] = embedding_dim
        active_sensors.append(str(i))
        inputs[str(i)] = torch.randn(1, embedding_dim, 1)

    config['input_dims'] = input_dims
    config['active_sensors'] = active_sensors
    config['device'] = device

    if cache:
        config['seq_length'] = 1
    else:
        config['seq_length'] = 6

    model = MOSCAN(**config)

    measure_model(model, inputs, device)
    print_model_summary(model, input_dims, 1, -1, device=device)

