import json
import shutil
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

def get_max_sampling_rate(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    sampling_rates = config["sampling rate"]
    max_sampling_rate = max(sampling_rates.values())
    
    return max_sampling_rate

def get_key(config_path, key):
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config[key]

def copy_json(src, dst):
    shutil.copy(src, dst)

def get_active_key(config_path, key, recursive=False):
    with open(config_path, 'r') as f:
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
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config[key]

def load_generalized_model(generalized_model_path, model_class, **model_args):
    model = model_class(**model_args)
    state_dict = torch.load(generalized_model_path)
    model.load_state_dict(state_dict)
    return model

def load_json(config_path):
    with open(config_path, 'r') as f:
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
    assert device in ["cuda", "cpu"], "Input device is not valid, please specify 'cuda' or 'cpu'"

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
    total_input_size = abs(np.prod(list(input_dims.values())) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
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
