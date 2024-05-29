import json

def get_max_sampling_rate(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    sampling_rates = config["sampling rate"]
    max_sampling_rate = max(sampling_rates.values())
    
    return max_sampling_rate

def get_active_key(config_path, key):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    active_keys = [val for val, is_active in config[key].items() if is_active]
    return active_keys