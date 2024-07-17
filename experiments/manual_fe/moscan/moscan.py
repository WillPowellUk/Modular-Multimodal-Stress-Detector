
def moscan(moscan_model, MOSCAN_CONFIG, DATASET_CONFIG, BATCHED_FE, DATASET_TYPE, BATCHED_DATASETS_PATH, NON_BATCHED_FE, NON_BATCHED_DATASETS_PATH, name=None):
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

    # Set the sensors to use
    active_sensors = get_active_key(DATASET_CONFIG, "sensors")
    SENSORS = '_'.join(active_sensors)

    # Optional fine tune on non-batched data
    FINE_TUNE = get_values(MOSCAN_CONFIG, "fine_tune")

    # Uncomment parameters to use them in a grid search
    HYPERPARAMETER_GRID = {
        "epochs": [10],
        # "embed_dim": [8, 16, 32],
        # "hidden_dim": [16, 32, 62, 64, 128, 256], 
        # "n_head_gen": [2, 4, 8],
        # "dropout": [0.3, 0.5, 0.7],
        # "attention_dropout": [0.3, 0.5, 0.7],
        # "learning_rate": [0.0001, 0.001, 0.01],
        # "batch_size": [8, 16, 32, 64],
        # "epochs": [5, 10, 15, 20],
        # "fine_tune_epochs": [1, 3, 5],
        # "fine_tune_learning_rate": [0.001, 0.0001, 0.00005],
        # "early_stopping_patience": [5, 7, 10, 20],
        # "early_stopping_patience": [5, 7, 10, 20],
        # "early_stopping_metric": ["loss", "accuracy"],
        # "predictor": ["hard_voting", "avg_pool", 'weighted_avg_pool',  "weighted_max_pool", "avg_pool", "max_pool"], 
    }

    # Grid Search Parameters
    hyperparams = HyperParamsIterator(MOSCAN_CONFIG, HYPERPARAMETER_GRID)

    for c, current_config in enumerate(hyperparams()):
        print(f"\n\nCurrent Configuration: {c+1} out of {len(hyperparams.combinations)}\n\n")

        batched_dataloader_params = {
            "batch_size": get_values(current_config, "batch_size"),
            "shuffle": True,
            "drop_last": False,
        }
        batched_losocv_loader = LOSOCVSensorDataLoader(
            BATCHED_FE, DATASET_CONFIG, **batched_dataloader_params
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
            NON_BATCHED_FE, DATASET_CONFIG, **non_batched_dataloader_params
        )
        non_batched_dataloaders, non_batched_input_dims = non_batched_losocv_loader.get_data_loaders(
            NON_BATCHED_DATASETS_PATH, dataset_type=DATASET_TYPE
        )

        assert (
            batched_input_dims == non_batched_input_dims
        ), "Input dimensions of batched and non-batched dataloaders do not match"

        assert batched_input_dims != 0, "Input dimensions are 0"

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
            if DATASET_TYPE == 'losocv':
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

            # Load Model Parameters
            model_config = load_json(current_config)
            model_config = {**model_config, }

            # Configure LossWrapper for the model
            loss_wrapper = LossWrapper(model_config["loss_fns"])

            # Modify Current Config with 
            model_config = load_json(current_config)
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            model_config["save_path"] = f"src/wesad/WESAD/ckpts/co_attention/wrist_manual_fe/{NON_BATCHED_WINDOW_LENGTH}s_{NON_BATCHED_SLIDING_LENGTH}s_{NON_BATCHED_SPLIT_LENGTH}s/generalized/{current_time}/{fold}"
            model_config["device"] = str(device)
            model_config["input_dims"] = batched_input_dims
            model_config["active_sensors"] = active_sensors
            if name is not None:
                model_config["name"] = name
            save_json(model_config, current_config)

            # Initialize model
            model = moscan_model(**model_config)

            # Initialize trainer
            trainer = PyTorchTrainer(
                model,
                current_config,
            )
            # if idx == 0:
            #     trainer.print_model_summary()

            # Train the model on the batched data without the sliding co-attention buffer
            trainer.model.seq_length = 1
            pre_trained_model_ckpt = trainer.train(train_loader_batched, val_loader_batched, loss_wrapper, 
                use_wandb=True,
                name_wandb=f"{model.NAME}_{fold}",
            )
            print(f"Pre-Trained Model checkpoint saved to: {pre_trained_model_ckpt}\n")

            # Validate model on non-batched data
            print("Validating Pre-Trained Model on Non-Batched Data")
            trainer.model.seq_length = get_values(current_config, "seq_length")
            if DATASET_TYPE == 'losocv':
                result = trainer.validate(val_loader_non_batched, loss_wrapper, ckpt_path=pre_trained_model_ckpt, subject_id=subject_id, pre_trained_run=True, check_overlap=True)
            else: 
                result = trainer.validate(val_loader_non_batched, loss_wrapper, ckpt_path=pre_trained_model_ckpt, subject_id=idx, pre_trained_run=True, check_overlap=True)

            # Fine Tune on non-batched (Optional)
            if FINE_TUNE:
                print("Fine Tuning Model on Non-Batched Data")
                fine_tune_loss_wrapper = LossWrapper(model_config["fine_tune_loss_fns"])

                trainer.model.seq_length = get_values(current_config, "seq_length")
                fine_tuned_model_ckpt = trainer.train(train_loader_non_batched, val_loader_non_batched, fine_tune_loss_wrapper, ckpt_path=pre_trained_model_ckpt, use_wandb=True, name_wandb=f"{model.NAME}_{fold}", fine_tune=True)
                print(f"Fine Tuned Model checkpoint saved to: {fine_tuned_model_ckpt}\n")
                # Validate model on non-batched data
                print("Validating Fine Tuned Model on Non-Batched Data")
                trainer.model.seq_length = get_values(current_config, "seq_length")
                if DATASET_TYPE == 'losocv':
                    result = trainer.validate(val_loader_non_batched, fine_tune_loss_wrapper, fine_tuned_model_ckpt, subject_id=subject_id, fine_tune_run=True)
                else: 
                    result = trainer.validate(val_loader_non_batched, fine_tune_loss_wrapper, fine_tuned_model_ckpt, subject_id=idx, fine_tune_run=True)
            
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


if __name__ == '__main__':
    from src.ml_pipeline.models.attention_models import MOSCAN
    MOSCAN_CONFIG = "config_files/model_training/deep/moscan_config.json"
    DATASET_CONFIG = "config_files/dataset/wesad_wrist_bvp_w_eda_configuration.json"
    moscan(MOSCAN, MOSCAN_CONFIG, DATASET_CONFIG)