import json
import shutil
import torch

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

def load_generalized_model(generalized_model_path, model_class, *model_args):
    model = model_class(*model_args)
    state_dict = torch.load(generalized_model_path)
    model.load_state_dict(state_dict)
    return model
