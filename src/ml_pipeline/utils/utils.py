import json
import itertools
import copy
import tempfile
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def get_max_sampling_rate(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    sampling_rates = config["sampling rate"]
    max_sampling_rate = max(sampling_rates.values())

    return max_sampling_rate


def get_key(config_path, key):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config[key]


def copy_json(src, dst):
    shutil.copy(src, dst)


def get_active_key(config_path, key, recursive=False):
    with open(config_path, "r") as f:
        config = json.load(f)

    if recursive:
        active_keys = []
        for sub_key, sub_value in config.items():
            if isinstance(sub_value, dict):
                if key == sub_key:
                    for feature in config[sub_key]:
                        for sub_feature in config[sub_key][feature]:
                            active_keys.append(sub_feature)
    else:
        active_keys = [val for val, is_active in config[key].items() if is_active]

    return active_keys


def get_values(config_path, key):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config[key]


def load_generalized_model(generalized_model_path, model_class, **model_args):
    model = model_class(**model_args)
    state_dict = torch.load(generalized_model_path)
    model.load_state_dict(state_dict)
    return model


def load_json(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def print_model_summary(model, input_dims, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            input_shapes = []
            for inp in input:
                if isinstance(inp, dict):
                    for key in inp:
                        input_shapes.append(list(inp[key].size()))
                else:
                    input_shapes.append(list(inp.size()))

            summary[m_key]["input_shape"] = input_shapes
            for inp_shape in summary[m_key]["input_shape"]:
                inp_shape[0] = batch_size

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Create input dictionary with tensors of appropriate shape
    x = {}
    for key, dim in input_dims.items():
        x[key] = torch.rand(2, dim, 10).type(dtype)  # assuming Z dimension to be 10

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    with torch.no_grad():
        model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>30}  {:>40} {:>20}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>30}  {:>40} {:>20}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(
        np.prod(list(input_dims.values())) * batch_size * 4.0 / (1024**2.0)
    )
    total_output_size = abs(
        2.0 * total_output * 4.0 / (1024**2.0)
    )  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4.0 / (1024**2.0))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


def print2(filename, text):
    with open(filename, "a") as file:
        file.write(text + "\n")


def print_weights_and_biases(attention):
    query_weights, key_weights, value_weights = attention.in_proj_weight.chunk(3, dim=0)

    print("Query weights shape:", query_weights.shape)
    print("Key weights shape:", key_weights.shape)
    print("Value weights shape:", value_weights.shape)

    query_biases, key_biases, value_biases = attention.in_proj_bias.chunk(3, dim=0)

    print("Query biases shape:", query_biases.shape)
    print("Key biases shape:", key_biases.shape)


def plot_attention(attention, title="", x_label="Source Sequence", y_label="Target Sequence"):
    """
    Plots the attention weights.
    
    Args:
        attention (numpy.ndarray or torch.Tensor): The attention weights. 
            Should be of shape (L, S) for single head or (N, L, S) for batched input
            or (N, num_heads, L, S) for multi-head attention.
        title (str): The title of the plot.
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    
    # If the attention weights are multi-headed or batched, average them
    if attention.ndim == 4:  # (N, num_heads, L, S)
        attention = attention.mean(axis=1)  # Average over heads
    if attention.ndim == 3:  # (N, L, S)
        attention = attention.mean(axis=0)  # Average over batch

    # Determine if annotations are needed
    annot = False
    annot_kws = {}
    if attention.shape[0] <= 10 and attention.shape[1] <= 10:
        annot = True
        annot_kws = {"size": 16, "weight": "bold"}

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='Blues', annot=annot, annot_kws=annot_kws, fmt=".2f")
    plt.title(title)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    # Set x-axis label at the top
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.show()
    plt.savefig("attention_weights.png", dpi=300, format="png", bbox_inches="tight")

class HyperParamsIterator:
    def __init__(self, json_path, hyperparameter_grid):
        self.json_path = json_path
        self.hyperparameter_grid = hyperparameter_grid
        self.grid_keys = list(hyperparameter_grid.keys())
        self.grid_values = list(hyperparameter_grid.values())
        self.combinations = list(itertools.product(*self.grid_values))
        self.temp_files = []

    def __call__(self):
        for combination in self.combinations:
            # Load the original JSON configuration
            with open(self.json_path, 'r') as f:
                config = json.load(f)

            # Update the configuration with the current combination of hyperparameters
            for key, value in zip(self.grid_keys, combination):
                config[key] = value

            # Create a temporary file and save the modified configuration
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            with open(temp_file.name, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Store the path of the temporary file
            temp_file_path = temp_file.name
            self.temp_files.append(temp_file_path)

            # Clean up previous temporary file
            if len(self.temp_files) > 1:
                os.remove(self.temp_files[-2])
                self.temp_files.pop(0)

            # Yield the path to the temporary JSON file
            yield temp_file_path

        # Clean up the last temporary file after the last iteration
        if self.temp_files:
            os.remove(self.temp_files[-1])
            self.temp_files.pop(0)

    def __del__(self):
        # Destructor to clean up any remaining temporary files
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass