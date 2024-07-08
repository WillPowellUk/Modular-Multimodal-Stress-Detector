import os
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

from src.ml_pipeline.train import PyTorchTrainer
from src.ml_pipeline.models.attention_models import MOSCAN
from src.ml_pipeline.losses import LossWrapper
from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader, SeqToSeqDataLoader
from src.utils import save_var
from src.ml_pipeline.utils import (
    get_active_key,
    get_key,
    load_json,
    save_json,
    copy_json,
    get_values,
    HyperParamsIterator,
)

# PRE_TRAINED_CKPTS = ["src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/5s_5s_5s/generalized/2024_07_05_13_15_50/fold_0/checkpoint_final.pth"]
PRE_TRAINED_CKPTS = ["src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/checkpoint_final.pth"]

# CONFIG file for the dataset
WRIST_CONFIG = "config_files/dataset/wesad_wrist_bvp_w_eda_configuration.json"

# CONFIG file for the model
MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"

# Set either losocv or kfold
# DATASET_TYPE = "losocv"
DATASET_TYPE = "cv_5"

# Set the sensors to use
active_sensors = get_active_key(WRIST_CONFIG, "sensors")
SENSORS = '_'.join(active_sensors)

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

# Uncomment parameters to use them in a grid search
HYPERPARAMETER_GRID = {
    # "embed_dim": [16],
    # "hidden_dim": [16, 32, 62, 64, 128, 256],
    # "n_head_gen": [2, 4, 8],
    # "dropout": [0.3, 0.5, 0.7],
    # "attention_dropout": [0.3, 0.5, 0.7],
    # "learning_rate": [0.0001, 0.001, 0.01],
    # "batch_size": [8, 16, 32],
    # "epochs": [5, 7, 10],
    "fine_tune_epochs": [1, 3, 5, 10, 20],
    "fine_tune_learning_rate": [0.0005, 0.0001, 0.00005],
    # "early_stopping_patience": [5,8,10,20],
    # "early_stopping_metric": ["loss", "accuracy"],
    "predictor": ["avg_pool"], # ['weighted_avg_pool', "avg_pool", "og"], 
    "kalman": [True]
}

# Grid Search Parameters
hyperparams = HyperParamsIterator(MOSCAN_CONFIG, HYPERPARAMETER_GRID)

for c, current_config in enumerate(hyperparams()):
    print(f"\n\nCurrent Configuration: {c+1} out of {len(hyperparams.combinations)}\n\n")

    non_batched_dataloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
    }
    non_batched_losocv_loader = LOSOCVSensorDataLoader(
        NON_BATCHED_FE, WRIST_CONFIG, **non_batched_dataloader_params
    )
    non_batched_dataloaders, non_batched_input_dims = non_batched_losocv_loader.get_data_loaders(
        NON_BATCHED_DATASETS_PATH, dataset_type=DATASET_TYPE
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    for idx, (subject_id, data_loader) in enumerate(non_batched_dataloaders.items()):
        train_loader_non_batched = data_loader["train"]
        val_loader_non_batched = data_loader["val"]
        if DATASET_TYPE == 'losocv':
            print(f"\nSubject: {subject_id}")
            fold = f"subject_{subject_id}"
        else:
            print(f"\nFold: {idx}")
            fold = f"fold_{idx}"
        print(f"Non-Batched Train Length: {len(train_loader_non_batched.dataset)}")
        print(f"Non-Batched Val Length: {len(val_loader_non_batched.dataset)}")
        print()

        # Load Model Parameters
        model_config = load_json(current_config)
        model_config = {**model_config, }

        # Mix the dataloaders so there are randomised segments whilst ensuring the sequential format is maintained.
        train_loader_non_batched = SeqToSeqDataLoader(train_loader_non_batched, model_config['fine_tune_sequence_length']) 

        # Configure LossWrapper for the model
        loss_wrapper = LossWrapper(model_config["loss_fns"])

        # Load Pre-Trained Model
        pre_trained_ckpt = PRE_TRAINED_CKPTS[idx]
        
        # Modify Current Config with 
        model_config = load_json(current_config)
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_config["save_path"] = f"src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/generalized/{current_time}/{fold}"
        model_config["device"] = str(device)
        model_config["input_dims"] = non_batched_input_dims
        model_config["active_sensors"] = active_sensors
        save_json(model_config, current_config)

        # Initialize model
        model = MOSCAN(**model_config)

        # Initialize trainer
        trainer = PyTorchTrainer(
            model,
            current_config,
        )

        # Validate model on non-batched data
        # print("Validating Pre-Trained Model on Non-Batched Data")
        # model_config["kalman"] = False
        # save_json(model_config, current_config)

        # trainer.model.token_length = get_values(current_config, "token_length")
        # if DATASET_TYPE == 'losocv':
        #     result = trainer.validate(val_loader_non_batched, loss_wrapper, ckpt_path=pre_trained_ckpt, subject_id=subject_id, pre_trained_run=True, check_overlap=True)
        # else: 
        #     result = trainer.validate(val_loader_non_batched, loss_wrapper, ckpt_path=pre_trained_ckpt, subject_id=idx, pre_trained_run=True, check_overlap=True)
        # print(result)
        
        # Fine Tune on non-batched

        # save_json(model_config, current_config)

        print("Fine Tuning Model on Non-Batched Data")
        fine_tune_loss_wrapper = LossWrapper(model_config["fine_tune_loss_fns"])

        trainer.model.token_length = get_values(current_config, "token_length")
        fine_tuned_model_ckpt = trainer.train(train_loader_non_batched, val_loader_non_batched, fine_tune_loss_wrapper, ckpt_path=pre_trained_ckpt, use_wandb=True, name_wandb=f"{model.NAME}_{fold}_fine-tune", use_local_wandb=True, fine_tune=True, val_freq_per_epoch=2)
        print(f"Fine Tuned Model checkpoint saved to: {fine_tuned_model_ckpt}\n")

        # Validate model on non-batched data
        print("Validating Fine Tuned Model on Non-Batched Data")
        trainer.model.token_length = get_values(current_config, "token_length")
        if DATASET_TYPE == 'losocv':
            result = trainer.validate(val_loader_non_batched, fine_tune_loss_wrapper, fine_tuned_model_ckpt, subject_id=subject_id, fine_tune_run=True)
        else: 
            result = trainer.validate(val_loader_non_batched, fine_tune_loss_wrapper, fine_tuned_model_ckpt, subject_id=idx, fine_tune_run=True)
        
        results.append(result)

        del trainer  # delete the trainer object to finish wandb

        if idx == 0:
            break

    # save the results to pkl
    save_path = get_values(current_config, "save_path")
    save_var(results, f"{save_path}/results.pkl", "Results")
    copy_json(current_config, f"{save_path}/config.json")

    from src.ml_pipeline.analysis import ModelResultsAnalysis
    from src.utils import load_var
    import os

    results_path = f"{save_path}/results.pkl"
    results = load_var(results_path)

    analysis = ModelResultsAnalysis(results)
    analysis.analyze_collective(os.path.dirname(results_path))

del hyperparams