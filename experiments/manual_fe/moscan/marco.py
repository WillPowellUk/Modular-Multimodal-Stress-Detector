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
from src.ml_pipeline.models.attention_models import MARCONet
from src.ml_pipeline.losses import LossWrapper
from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader
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

# CONFIG file for the model
MARCO_CONFIG = "config_files/model_training/deep/marco_config.json"

# CONFIG file for the dataset
# WRIST_CONFIG = "config_files/dataset/wesad_wrist_configuration.json"
WRIST_CONFIG = "config_files/dataset/wesad_wrist_bvp_eda_configuration.json"

# Set either losocv or kfold
# DATASET_TYPE = "losocv"
DATASET_TYPE = "cv_5"

# Set the sensors to use
# SENSORS = "all"
SENSORS = "bvp_eda"

# Optional fine tune on non-batched data
FINE_TUNE = get_values(MARCO_CONFIG, "fine_tune")

# Load Train Dataloaders for LOSOCV
BATCHED_WINDOW_LENGTH = 30
BATCHED_SPLIT_LENGTH = int(
    BATCHED_WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
BATCHED_SLIDING_LENGTH = BATCHED_SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples
BATCHED_WRIST_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

HYPERPARAMETER_GRID = {
    # "embed_dim": [16],
    # "hidden_dim": [16, 32, 62, 64, 128, 256],
    # "n_head_gen": [2, 4, 8],
    "dropout": [0.3, 0.5, 0.7],
    "attention_dropout": [0.3, 0.5, 0.7],
    # "learning_rate": [0.0001, 0.001, 0.01],
    # "batch_size": [8, 16, 32],
    # "epochs": [5, 7, 10],
    # "fine_tune_epochs": [1, 3, 5],
    # "fine_tune_learning_rate": [0.001, 0.0001, 0.00005],
    "early_stopping_patience": [5, 8, 10, 20],
    "early_stopping_metric": ["loss", "accuracy"],
}

# Grid Search Parameters
hyperparams = HyperParamsIterator(MARCO_CONFIG, HYPERPARAMETER_GRID)

for c, current_config in enumerate(hyperparams()):
    print(
        f"\n\nCurrent Configuration: {c+1} out of {len(hyperparams.combinations)}\n\n"
    )

    batched_dataloader_params = {
        "batch_size": get_values(current_config, "batch_size"),
        "shuffle": True,
        "drop_last": False,
    }
    batched_losocv_loader = LOSOCVSensorDataLoader(
        BATCHED_WRIST_FE, WRIST_CONFIG, **batched_dataloader_params
    )
    batched_dataloaders, batched_input_dims = batched_losocv_loader.get_data_loaders(
        BATCHED_DATASETS_PATH, dataset_type=DATASET_TYPE
    )

    non_batched_dataloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
    }
    non_batched_losocv_loader = LOSOCVSensorDataLoader(
        NON_BATCHED_FE, WRIST_CONFIG, **non_batched_dataloader_params
    )
    (
        non_batched_dataloaders,
        non_batched_input_dims,
    ) = non_batched_losocv_loader.get_data_loaders(
        NON_BATCHED_DATASETS_PATH, dataset_type=DATASET_TYPE
    )

    assert (
        batched_input_dims == non_batched_input_dims
    ), "Input dimensions of batched and non-batched dataloaders do not match"

    assert batched_input_dims != 0, "Input dimensions are 0"

    # Load Model Parameters
    model_config = load_json(current_config)
    model_config = {
        **model_config,
        "input_dims": batched_input_dims,
    }

    # Configure LossWrapper for the model
    loss_wrapper = LossWrapper(model_config["loss_fns"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    for idx, ((subject_id, batched_loader), (_, val_loader)) in enumerate(
        zip(batched_dataloaders.items(), non_batched_dataloaders.items())
    ):
        train_loader_batched = batched_loader["train"]
        val_loader_batched = batched_loader["val"]
        train_loader_non_batched = val_loader["train"]
        val_loader_non_batched = val_loader["val"]
        if DATASET_TYPE == "losocv":
            print(f"\nSubject: {subject_id}")
            fold = f"subject_{subject_id}"
        else:
            print(f"\nFold: {idx}")
            fold = f"fold_{idx}"
        print(f"Batched Train Length: {len(train_loader_batched.dataset)}")
        print(f"Batched Val Length: {len(val_loader_batched.dataset)}")
        print(f"Non-Batched Train Length: {len(train_loader_non_batched.dataset)}")
        print(f"Non-Batched Val Length: {len(val_loader_non_batched.dataset)}")
        print()

        # Modify Current Config
        config = load_json(current_config)
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config[
            "save_path"
        ] = f"src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/generalized/{current_time}/{fold}"
        save_json(config, current_config)

        # Initialize model
        model = MARCONet(**model_config)

        # Initialize trainer
        trainer = PyTorchTrainer(
            model,
            current_config,
            device,
        )
        # if idx == 0:
        #     trainer.print_model_summary()

        # Train the model on the batched data without the sliding co-attention buffer
        trainer.model.source_seq_length = 1
        pre_trained_model_ckpt = trainer.train(
            train_loader_batched,
            val_loader_batched,
            loss_wrapper,
            use_wandb=True,
            name_wandb=f"{model.NAME}_{fold}",
        )
        print(f"Pre-Trained Model checkpoint saved to: {pre_trained_model_ckpt}\n")

        # Validate model on non-batched data
        print("Validating Pre-Trained Model on Non-Batched Data")
        trainer.model.source_seq_length = get_values(current_config, "source_seq_length")
        if DATASET_TYPE == "losocv":
            result = trainer.validate(
                val_loader_non_batched,
                loss_wrapper,
                ckpt_path=pre_trained_model_ckpt,
                subject_id=subject_id,
                pre_trained_run=True,
                check_overlap=True,
            )
        else:
            result = trainer.validate(
                val_loader_non_batched,
                loss_wrapper,
                ckpt_path=pre_trained_model_ckpt,
                subject_id=idx,
                pre_trained_run=True,
                check_overlap=True,
            )

        # Fine Tune on non-batched (Optional)
        if FINE_TUNE:
            print("Fine Tuning Model on Non-Batched Data")
            fine_tune_loss_wrapper = LossWrapper(model_config["fine_tune_loss_fns"])

            trainer.model.source_seq_length = get_values(current_config, "source_seq_length")
            fine_tuned_model_ckpt = trainer.train(
                train_loader_non_batched,
                val_loader_non_batched,
                fine_tune_loss_wrapper,
                ckpt_path=pre_trained_model_ckpt,
                use_wandb=True,
                name_wandb=f"{model.NAME}_{fold}",
                fine_tune=True,
            )
            print(f"Fine Tuned Model checkpoint saved to: {fine_tuned_model_ckpt}\n")
            # Validate model on non-batched data
            print("Validating Fine Tuned Model on Non-Batched Data")
            trainer.model.source_seq_length = get_values(current_config, "source_seq_length")
            if DATASET_TYPE == "losocv":
                result = trainer.validate(
                    val_loader_non_batched,
                    fine_tune_loss_wrapper,
                    fine_tuned_model_ckpt,
                    subject_id=subject_id,
                    fine_tune_run=FINE_TUNE,
                    pre_trained_run=not FINE_TUNE,
                )
            else:
                result = trainer.validate(
                    val_loader_non_batched,
                    fine_tune_loss_wrapper,
                    fine_tuned_model_ckpt,
                    subject_id=idx,
                    fine_tune_run=FINE_TUNE,
                    pre_trained_run=not FINE_TUNE,
                )

        results.append(result)

        del trainer  # delete the trainer object to finish wandb

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
