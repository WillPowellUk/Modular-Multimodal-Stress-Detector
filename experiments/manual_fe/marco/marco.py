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
while "experiments" not in os.listdir(target_dir) and target_dir != os.path.dirname(target_dir):
    target_dir = os.path.dirname(target_dir)

# Append the target directory to sys.path
if "experiments" in os.listdir(target_dir):
    sys.path.append(target_dir)
else:
    raise ImportError("Could not find 'experiments' directory in the path hierarchy")

from src.ml_pipeline.train import PyTorchTrainer
from src.ml_pipeline.models.attention_models import MARCONet
from src.ml_pipeline.losses import LossWrapper
from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader
from src.utils import save_var
from src.ml_pipeline.utils import (
    get_active_key,
    get_key,
    load_json,
    copy_json,
    get_values,
    HyperParamsIterator
)

# CONFIG file for the model
MARCO_CONFIG = "config_files/model_training/deep/marco_config.json"

# CONFIG file for the dataset
WRIST_CONFIG = "config_files/dataset/wesad_wrist_configuration.json"

# Set the sensors to use
sensors = "all"

# Load Train Dataloaders for LOSOCV
TRAIN_WINDOW_LENGTH = 30
TRAIN_SPLIT_LENGTH = int(
    TRAIN_WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
TRAIN_SLIDING_LENGTH = TRAIN_SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples
TRAIN_WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{TRAIN_WINDOW_LENGTH}s_{TRAIN_SLIDING_LENGTH}s_{TRAIN_SPLIT_LENGTH}s/wrist_features.hdf5"
TRAIN_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{sensors}/{TRAIN_WINDOW_LENGTH}s_{TRAIN_SLIDING_LENGTH}s_{TRAIN_SPLIT_LENGTH}s/losocv_datasets.pkl"

# Load Val Dataloaders for LOSOCV
VAL_WINDOW_LENGTH = 5
VAL_SLIDING_LENGTH = VAL_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
VAL_SPLIT_LENGTH = VAL_WINDOW_LENGTH  # this will not sub-split the data
VAL_WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{VAL_WINDOW_LENGTH}s_{VAL_SLIDING_LENGTH}s_{VAL_SPLIT_LENGTH}s/wrist_features.hdf5"
VAL_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{sensors}/{VAL_WINDOW_LENGTH}s_{VAL_SLIDING_LENGTH}s_{VAL_SPLIT_LENGTH}s/losocv_datasets.pkl"

HYPERPARAMETER_GRID = {
    "embed_dim": [16],
    # "hidden_dim": [16, 32, 62, 64, 128, 256, 512],
    # "n_head_gen": [2, 4, 8],
    # "dropout": [0.3, 0.5, 0.7],
    # "learning_rate": [0.0001, 0.001, 0.01],
    # "batch_size": [8, 16, 32]
}

# Grid Search Parameters
hyperparams = HyperParamsIterator(MARCO_CONFIG, HYPERPARAMETER_GRID)

for c, current_config in enumerate(hyperparams()):
    print(f"\n\nCurrent Configuration: {c} out of {len(hyperparams.combinations)}\n\n")

    train_dataloader_params = {
        "batch_size": get_values(current_config, "batch_size"),
        "shuffle": True,
        "drop_last": False,
    }
    train_losocv_loader = LOSOCVSensorDataLoader(
        TRAIN_WRIST_FE, WRIST_CONFIG, **train_dataloader_params
    )
    train_dataloaders, train_input_dims = train_losocv_loader.get_data_loaders(
        TRAIN_DATASETS_PATH
    )

    val_dataloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "drop_last": True,
    }
    val_losocv_loader = LOSOCVSensorDataLoader(
        VAL_WRIST_FE, WRIST_CONFIG, **val_dataloader_params
    )
    val_dataloaders, val_input_dims = val_losocv_loader.get_data_loaders(
        VAL_DATASETS_PATH, val_only=True
    )

    assert (
        train_input_dims == val_input_dims
    ), "Input dimensions of train and val dataloaders do not match"

    # Load Model Parameters
    model_config = load_json(current_config)
    model_config = {**model_config, "input_dims": train_input_dims}

    # Configure LossWrapper for the model
    loss_wrapper = LossWrapper(current_config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    for idx, ((subject_id, train_loader), (_, val_loader)) in enumerate(
        zip(train_dataloaders.items(), val_dataloaders.items())
    ):
        train_loader_batched = train_loader["train"]
        val_loader_batched = train_loader["val"]
        val_loader = val_loader["val"]
        print(f"\nSubject: {subject_id}")
        print(f"Batched Train Length: {len(train_loader_batched.dataset)}")
        print(f"Batched Val Length: {len(val_loader_batched.dataset)}")
        print(f"Val Length: {len(val_loader.dataset)}")
        print()

        # Initialize model
        model = MARCONet(**model_config)

        # Initialize trainer
        trainer = PyTorchTrainer(
            model,
            train_loader_batched,
            val_loader_batched,
            loss_wrapper,
            current_config,
            device,
        )
        trainer.save_path = trainer.save_path.format(fold=f"subject_{subject_id}")
        # if idx == 0:
        #     trainer.print_model_summary()

        # Train the model on the batched data without the sliding co-attention buffer
        trainer.model.token_length = 1
        trained_model_ckpt = trainer.train(
            use_wandb=True, name_wandb=f"marco_subject_{subject_id}_embed_{model_config['embed_dim']}"
        )
        print(f"Model checkpoint saved to: {trained_model_ckpt}\n")

        # from src.ml_pipeline.utils import print_weights_and_biases, plot_attention
        # bvp_head_0_attention = trainer.model.self_attention_blocks['bvp'][0].attention
        # print_weights_and_biases(bvp_head_0_attention)
        # plot_attention(bvp_head_0_attention.in_proj_weight)
        # plot_attention(_)

        # Now validate using the sliding co-attention buffer
        trainer.model.token_length = get_values(current_config, "token_length")
        result = trainer.validate(trained_model_ckpt, subject_id, val_loader=val_loader)
        del trainer  # delete the trainer object and finish wandb
        results.append(result)

        if idx + 1 == 3:
            break

    # save the results to pkl
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = f"src/wesad/WESAD/results/marco/wrist_results/{VAL_WINDOW_LENGTH}s_{VAL_SLIDING_LENGTH}s_{VAL_SPLIT_LENGTH}s/{current_time}/generalized"
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