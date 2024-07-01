import os
import time
import json
import numpy as np
import torch
from torch import nn
import wandb
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from src.ml_pipeline.utils import print_model_summary


class PyTorchTrainer:
    def __init__(self, model, train_loader, val_loader, loss_func, config_path, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.configs = self.load_config(config_path)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.configs["learning_rate"]
        )
        self.loss_func = loss_func
        self.num_classes = self.configs["num_classes"]
        self.save_path = self.configs["save_path"]

    def __del__(self):
        if wandb.run is not None:
            wandb.finish()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            configs = json.load(f)
        return configs

    def print_model_summary(self):
        model_copy = self.model
        print_model_summary(
            self.model,
            self.model.input_dims,
            batch_size=self.train_loader.batch_size,
            device=self.device.type,
        )
        self.model = model_copy

    def train(self, use_wandb=False, name_wandb=None):
        if use_wandb:
            # Initialize wandb
            if name_wandb is None:
                wandb.init(project="MMSD", config=self.configs)
            else:
                wandb.init(project="MMSD", config=self.configs, name=name_wandb)
            wandb.watch(self.model, log="all", log_freq=10)

        for epoch in range(self.configs["epoch"]):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f'Epoch {epoch+1}/{self.configs["epoch"]}',
            )
            val_metrics = None
            for s, (batch_x, batch_y) in progress_bar:
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                if inputs["bvp"].shape[0] != self.train_loader.batch_size:
                    continue
                labels = batch_y.to(self.device)
                final_output = self.model(inputs)
                loss = self.loss_func(
                    final_output,
                    torch.nn.functional.one_hot(
                        labels - 1, num_classes=self.num_classes
                    ).float(),
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                _, preds = torch.max(final_output, 1)
                epoch_correct += (preds == (labels - 1)).sum().item()
                epoch_total += labels.size(0)

                progress_bar.set_postfix(loss=loss.item())

                # Log intermediate training metrics to wandb
                if use_wandb and s % (len(self.train_loader) // 20) == 0:
                    val_metrics = self.validate()
                    train_acc = epoch_correct / epoch_total
                    train_loss = epoch_loss / (s + 1)

                    # Log intermediate training and validation metrics to wandb
                    step = epoch * len(self.train_loader) + s
                    wandb.log(
                        {
                            "Train Loss": train_loss,
                            "Validation Loss": val_metrics[self.model.NAME]["loss"],
                            "Train Accuracy": train_acc,
                            "Validation Accuracy": val_metrics[self.model.NAME][
                                "accuracy"
                            ],
                            "Step": step,
                        }
                    )

            avg_loss = epoch_loss / len(self.train_loader)
            avg_acc = epoch_correct / epoch_total
            print(
                f'Epoch: {epoch}, | training loss: {avg_loss:.4f}, | training accuracy: {avg_acc:.4f} | validation loss: {val_metrics[self.model.NAME]["loss"]:.4f} | validation accuracy: {val_metrics[self.model.NAME]["accuracy"]:.4f}'
            )

            # Log end of epoch training loss and accuracy to wandb
            if use_wandb:
                wandb.log(
                    {"Train Loss": avg_loss, "Train Accuracy": avg_acc, "Epoch": epoch}
                )

                # Validate and log validation loss and accuracy
                val_metrics = self.validate()
                wandb.log(
                    {
                        "Validation Loss": val_metrics[self.model.NAME]["loss"],
                        "Validation Accuracy": val_metrics[self.model.NAME]["accuracy"],
                        "Epoch": epoch,
                    }
                )

            if (epoch + 1) % 10 == 0:
                save_path = f"{self.save_path}/checkpoint_{epoch + 1}.pth"
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(self.model.state_dict(), save_path)

        final_save_path = f"{self.save_path}/checkpoint_{epoch + 1}.pth"
        directory = os.path.dirname(final_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), final_save_path)
        return final_save_path

    def validate(self, ckpt_path=None, subject_id=None, val_loader=None):
        # Use provided validation data loader if available or use the default one
        if val_loader is not None:
            val_loader = val_loader
        elif self.val_loader is not None:
            val_loader = self.val_loader
        else:
            raise ValueError("Validation data loader is not provided")

        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()
        y_true = []
        y_pred = []

        inference_times = []
        epoch_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                labels = batch_y.to(self.device)

                # Measure inference time for each batch
                if self.device.type == "cuda":
                    torch.cuda.synchronize()  # Synchronize CUDA operations before starting the timer
                start_time = time.time()
                final_output = self.model(inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()  # Synchronize CUDA operations after model inference
                end_time = time.time()

                inference_times.append(
                    (end_time - start_time) * 1000
                )  # Convert to milliseconds

                loss = self.loss_func(
                    final_output,
                    torch.nn.functional.one_hot(
                        labels - 1, num_classes=self.num_classes
                    ).float(),
                )
                epoch_loss += loss.item()

                _, preds = torch.max(final_output, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Calculate average inference time in milliseconds
        avg_inference_time = np.mean(inference_times)
        avg_loss = epoch_loss / len(val_loader)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true) - 1  # correct for labelling starting from index `1`
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(
            y_true, y_pred, labels=[i for i in range(self.num_classes)]
        )
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        results = {}
        results[self.model.NAME] = {
            "subject_id": subject_id,
            "loss": avg_loss,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "inference_time_ms": avg_inference_time,
            "device": str(self.device),
        }

        return results
