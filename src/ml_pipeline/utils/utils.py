import json

def get_max_sampling_rate(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    sampling_rates = config["sampling rate"]
    max_sampling_rate = max(sampling_rates.values())
    
    return max_sampling_rate

def get_active_sensors(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    active_sensors = [sensor for sensor, is_active in config['sensors'].items() if is_active]
    return active_sensors