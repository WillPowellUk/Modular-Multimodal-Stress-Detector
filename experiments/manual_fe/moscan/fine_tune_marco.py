import os
from datetime import datetime
import torch
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

PRE_TRAINED_CKPT = "src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/5s_5s_5s/generalized/2024_07_04_11_24_10/fold_0/checkpoint_7.pth"

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

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/wrist_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/wrist/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

HYPERPARAMETER_GRID = {
    "embed_dim": [16],
    # "embed_dim": [16],
    # "hidden_dim": [16, 32, 62, 64, 128, 256],
    # "n_head_gen": [2, 4, 8],
    # "dropout": [0.3],
    # "learning_rate": [0.0001, 0.001, 0.01],
    # "batch_size": [8, 16, 32]
    # "epochs": [5, 7, 10],
    # "fine_tune_epochs": [1, 3, 5],
    # "fine_tune_learning_rate": [0.001, 0.0001, 0.00005],
}

# Grid Search Parameters
hyperparams = HyperParamsIterator(MARCO_CONFIG, HYPERPARAMETER_GRID)

for c, current_config in enumerate(hyperparams()):
    print(
        f"\n\nCurrent Configuration: {c+1} out of {len(hyperparams.combinations)}\n\n"
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

    # Load Model Parameters
    model_config = load_json(current_config)
    model_config = {
        **model_config,
        "input_dims": non_batched_input_dims,
    }

    # Configure LossWrapper for the model
    loss_wrapper = LossWrapper(model_config["loss_fns"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    for idx, (subject_id, val_loader) in enumerate(non_batched_dataloaders.items()):
        train_loader_non_batched = val_loader["train"]
        val_loader_non_batched = val_loader["val"]
        if DATASET_TYPE == "losocv":
            print(f"\nSubject: {subject_id}")
            fold = f"subject_{subject_id}"
        else:
            print(f"\nFold: {idx}")
            fold = f"fold_{idx}"
        print(f"Non-Batched Train Length: {len(train_loader_non_batched.dataset)}")
        print(f"Non-Batched Val Length: {len(val_loader_non_batched.dataset)}")
        print()

        # Modify Current Config
        config = load_json(current_config)
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config[
            "save_path"
        ] = f"src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/generalized/{current_time}/{fold}/fine_tuned"
        save_json(config, current_config)

        # Initialize model
        model = MARCONet(**model_config)

        # Initialize trainer
        trainer = PyTorchTrainer(
            model,
            current_config,
            device,
        )
        fine_tune_loss_wrapper = LossWrapper(model_config["fine_tune_loss_fns"])

        print("Fine Tuning Model on Non-Batched Data")
        trainer.model.source_seq_length = get_values(current_config, "source_seq_length")
        fine_tuned_model_ckpt = trainer.train(
            train_loader_non_batched,
            val_loader_non_batched,
            fine_tune_loss_wrapper,
            ckpt_path=PRE_TRAINED_CKPT,
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

        # if idx == 0:
        #     break

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
