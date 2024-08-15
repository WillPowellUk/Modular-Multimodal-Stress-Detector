from datetime import datetime
import torch
from src.ml_pipeline.train import PyTorchTrainer
from src.ml_pipeline.models.attention_models import MOSCAN
from experiments.manual_fe.moscan.moscan import moscan
from src.ml_pipeline.losses import LossWrapper
from src.ml_pipeline.data_loader import LOSOCVSensorDataLoader
from src.ml_pipeline.utils.utils import load_json, save_json

# Load Train Dataloaders for LOSOCV
TYPE = "wrist"
SENSORS = "bvp_w_eda_temp_w_acc"
DATASET_TYPE = "losocv"
DATASET_CONFIG = f"config_files/dataset/wesad_{TYPE}_all_configuration.json"

BATCHED_WINDOW_LENGTH = 30
BATCHED_SPLIT_LENGTH = int(
    BATCHED_WINDOW_LENGTH / 6
)  # this will sub-split the data 6 times each of 5 seconds
BATCHED_SLIDING_LENGTH = BATCHED_SPLIT_LENGTH  # this will create 6 samples per 30 seconds since 30/5 = 6 with 5:1 ratio of synthetic to real samples
BATCHED_FE = f"src/wesad/WESAD/manual_fe/{TYPE}_manual_fe/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{TYPE}_features.hdf5"
BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/{TYPE}/{SENSORS}/{BATCHED_WINDOW_LENGTH}s_{BATCHED_SLIDING_LENGTH}s_{BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

# Load Val Dataloaders for LOSOCV
NON_BATCHED_WINDOW_LENGTH = 5
NON_BATCHED_SLIDING_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will create no overlap between segments i.e. no augmented / synthetic data.
NON_BATCHED_SPLIT_LENGTH = NON_BATCHED_WINDOW_LENGTH  # this will not sub-split the data
NON_BATCHED_FE = f"src/wesad/WESAD/manual_fe/{TYPE}_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{TYPE}_features.hdf5"
NON_BATCHED_DATASETS_PATH = f"src/wesad/WESAD/datasets/{TYPE}/{SENSORS}/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/{DATASET_TYPE}_datasets.pkl"

MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"
ckpts = [
    "src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/5s_5s_5s/generalized/2024_07_28_16_56_44/subject_2/checkpoint_final.pth"
]
predictors = [
    "hard_voting",
    "weighted_attn_pool",
    "stacked_attn_pool",
    "stacked_max_pool",
    "stacked_avg_pool",
]

for ckpt in ckpts:
    for p in predictors:
        batched_dataloader_params = {
            "batch_size": 32,
            "shuffle": True,
            "drop_last": False,
        }
        batched_losocv_loader = LOSOCVSensorDataLoader(
            BATCHED_FE, DATASET_CONFIG, **batched_dataloader_params
        )
        (
            batched_dataloaders,
            batched_input_dims,
        ) = batched_losocv_loader.get_data_loaders(
            BATCHED_DATASETS_PATH, dataset_type=DATASET_TYPE
        )

        non_batched_dataloader_params = {
            "batch_size": 1,
            "shuffle": False,
            "drop_last": False,
        }
        non_batched_losocv_loader = LOSOCVSensorDataLoader(
            NON_BATCHED_FE, DATASET_CONFIG, **non_batched_dataloader_params
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

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        fold = "0"
        train_loader_batched = batched_dataloaders[0]["train"]
        val_loader_batched = batched_dataloaders[2]["val"]

        # Load Model Parameters
        model_config = load_json(MOSCAN_CONFIG)
        model_config = {
            **model_config,
        }

        # Configure LossWrapper for the model
        loss_wrapper = LossWrapper(model_config["loss_fns"])

        # Modify Current Config with
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_config[
            "save_path"
        ] = f"src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/generalized/{current_time}/{fold}"
        model_config["device"] = str(device)
        model_config["input_dims"] = batched_input_dims
        model_config["active_sensors"] = ["bvp", "w_eda", "w_temp", "w_acc"]

        tmp_json = "tmp/config.json"
        save_json(model_config, tmp_json)

        # Initialize model
        model = MOSCAN(**model_config)

        model.load_state_dict(torch.load(ckpt))

        # Initialize trainer
        trainer = PyTorchTrainer(
            model,
            tmp_json,
        )

        # Train the model on the batched data without the sliding co-attention buffer
        trainer.model.source_seq_length = 1
        pre_trained_model_ckpt = trainer.validate(
            val_loader_batched, loss_wrapper, ckpt_path=ckpt
        )
        print(f"Pre-Trained Model checkpoint saved to: {pre_trained_model_ckpt}\n")
