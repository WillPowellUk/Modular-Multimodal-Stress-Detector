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
    def __init__(self, model, config_path, device):
        self.model = model.to(device)
        self.device = device
        self.configs = self.load_config(config_path)
        self.num_classes = self.configs["num_classes"]
        self.save_path = self.configs["save_path"]

    def __del__(self):
        if wandb.run is not None:
            wandb.finish()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            configs = json.load(f)
        return configs

    def print_model_summary(self, train_loader):
        model_copy = self.model
        print_model_summary(
            self.model,
            self.model.input_dims,
            batch_size=train_loader.batch_size,
            device=self.device.type,
        )
        self.model = model_copy

    def train(self, train_loader, val_loader, loss_func, ckpt_path=None, use_wandb=False, name_wandb=None, fine_tune=False):
        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

        # Set learning rate and epochs based on fine-tuning or training from scratch
        if fine_tune:
            epochs = self.configs["fine_tune_epochs"]
            learning_rate = self.configs["fine_tune_learning_rate"]
        else:
            epochs = self.configs["epochs"]
            learning_rate = self.configs["learning_rate"]

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        if use_wandb:
            if fine_tune:
                # Continue from the previous wandb run
                wandb.init(project="MMSD", resume="must", name=name_wandb)
            else:
                # Initialize a new wandb run
                if name_wandb is None:
                    wandb.init(project="MMSD", config=self.configs)
                else:
                    wandb.init(project="MMSD", config=self.configs, name=name_wandb)
            wandb.watch(self.model, log="all", log_freq=5)

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f'Epoch {epoch+1}/{epochs}',
            )
            val_metrics = None
            for s, (batch_x, batch_y) in progress_bar:
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                if inputs["bvp"].shape[0] != train_loader.batch_size:
                    print("Batch size mismatch")
                    continue
                labels = batch_y.to(self.device)
                final_output = self.model(inputs)
                loss = loss_func(
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
                accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
                progress_bar.set_postfix(accuracy=accuracy,loss=loss.item())

                # Log intermediate training metrics to wandb
                if use_wandb and s % (len(train_loader) // 10) == 0:
                    val_metrics = self.validate(val_loader, loss_func)
                    train_acc = epoch_correct / epoch_total
                    train_loss = epoch_loss / (s + 1)

                    # Log intermediate training and validation metrics to wandb
                    step = epoch * len(train_loader) + s
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

            avg_loss = epoch_loss / len(train_loader)
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
                val_metrics = self.validate(val_loader, loss_func)
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


    def validate(self, val_loader, loss_func, ckpt_path=None, subject_id=None, pre_trained_run=False, fine_tune_run=False):
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

                loss = loss_func(
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

        if wandb.run is not None and pre_trained_run:
            wandb.log(
                {
                    "Pre-Trained Validation Loss": avg_loss,
                    "Pre-Trained Validation Accuracy": accuracy,
                    "Pre-Trained Validation Precision": precision,
                    "Pre-Trained Validation Recall": recall,
                    "Pre-Trained Validation F1 Score": f1,
                    "Pre-Trained Validation Inference Time (ms)": avg_inference_time,
                }
            )

        if wandb.run is not None and fine_tune_run:
            wandb.log(
                {
                    "Fine-Tuned Validation Loss": avg_loss,
                    "Fine-Tuned Validation Accuracy": accuracy,
                    "Fine-Tuned Validation Precision": precision,
                    "Fine-Tuned Validation Recall": recall,
                    "Fine-Tuned Validation F1 Score": f1,
                    "Fine-Tuned Validation Inference Time (ms)": avg_inference_time,
                }
            )

        return results

    def measure_inference_time(self, val_loader, warmup_batches=20, repetitions=1000, ckpt_path=None):
        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            # Warm up the GPU
            for i, (batch_x, batch_y) in enumerate(val_loader):
                if i >= warmup_batches:
                    break
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                _ = self.model(inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            # Measure inference time
            rep = 0
            for i, (batch_x, batch_y) in enumerate(val_loader):
                if rep >= repetitions:
                    break
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}

                starter.record()
                _ = self.model(inputs)
                ender.record()

                # Wait for GPU to synchronize
                torch.cuda.synchronize()
                
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                rep += 1

        mean_syn = np.mean(timings)
        std_syn = np.std(timings)

        print(f"Mean Inference Time: {mean_syn:.6f} ms")
        print(f"Standard Deviation: {std_syn:.6f} ms")
        
        return mean_syn, std_syn

