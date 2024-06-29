from src.ml_pipeline.train import PyTorchTrainer
from src.ml_pipeline.models.attention_models import MARCONet
from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader
from src.ml_pipeline.utils import get_active_key, get_key, load_json, copy_json, get_values
from src.utils import save_var
from datetime import datetime
import torch

WINDOW_LENGTH = 10
SLIDING_LENGTH = 2 # this will create 5 segments per 10 seconds since 10/2 = 5 with 4:1 ratio of synthetic to real samples
SPLIT_LENGTH = WINDOW_LENGTH # this will not sub-split the data

DATASETS_PATH = f'src/wesad/WESAD/datasets/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/losocv_datasets.pkl'
WRIST_CONFIG = 'config_files/dataset/wesad_wrist_configuration.json'
WRIST_FE = f'src/wesad/WESAD/manual_fe/wrist_manual_fe/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/wrist_features.hdf5'
MARCO_CONFIG = 'config_files/model_training/deep/marco_config.json'

dataloader_params = {
    'batch_size': get_values(MARCO_CONFIG, 'batch_size'),
    'shuffle': False,
    'drop_last': True
}

# Load Dataloaders for LOSOCV
losocv_loader = LOSOCVSensorDataLoader(WRIST_FE, WRIST_CONFIG, **dataloader_params)
dataloaders, input_dims = losocv_loader.get_data_loaders(DATASETS_PATH)

# Load Model Parameters
model_config = load_json(MARCO_CONFIG)
model_config = {
    **model_config,
    'input_dims': input_dims
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

results = []
for i, (subject_id, loaders) in enumerate(dataloaders.items()):
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print(f'\nSubject: {subject_id}')
    print(f'Train: {len(train_loader.dataset)}')
    print(f'Val: {len(val_loader.dataset)}')
    print()

    # Initialize model
    model = MARCONet(**model_config)

    # Initialize trainerself.attention(x, x, x)
    trainer = PyTorchTrainer(model, train_loader, val_loader, MARCO_CONFIG, device)
    trainer.save_path = trainer.save_path.format(fold=f'subject_{subject_id}')
    # if i == 0:
    #     trainer.print_model_summary()
    trained_model_ckpt = trainer.train()
    print(f'Model checkpoint saved to: {trained_model_ckpt}\n')

    result = trainer.validate(trained_model_ckpt)
    results.append(result)

# save the results to pkl
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_path = f'src/wesad/WESAD/results/san/wrist_results/{WINDOW_LENGTH}s_{SLIDING_LENGTH}s_{SPLIT_LENGTH}s/{current_time}/generalized'
save_var(results, f'{save_path}/results.pkl', 'Results')
copy_json(MARCO_CONFIG, f'{save_path}/config.json')

from src.ml_pipeline.analysis import ModelResultsAnalysis
from src.utils import load_var

results_path = f'{save_path}/results.pkl'
results = load_var(results_path)

analysis = ModelResultsAnalysis(results)
analysis.analyze_collective()