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
    def __init__(self, model, config_path):
        self.configs = self.load_config(config_path)
        self.num_classes = self.configs["num_classes"]
        self.save_path = self.configs["save_path"]
        self.device = self.configs["device"]
        self.model = model.to(self.device)
        self.batch_size = self.configs["batch_size"]
        self.print_summary = True

    def __del__(self):
        if wandb.run is not None:
            wandb.finish()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            configs = json.load(f)
        return configs

    def print_model_summary(self, token_length, batch_size):
        model_copy = self.model
        print_model_summary(
            self.model,
            self.model.input_dims,
            token_length,
            batch_size,
            device=self.device,
        )
        self.model = model_copy

    def train(
        self,
        train_loader,
        val_loader,
        loss_func,
        ckpt_path=None,
        mixed_grad=None,
        use_wandb=False,
        name_wandb=None,
        use_local_wandb=False,
        fine_tune=False,
        val_freq_per_epoch=10,
    ):
        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=torch.device(self.device))
            )

        # Set learning rate and epochs based on fine-tuning or training from scratch
        if fine_tune:
            epochs = self.configs["fine_tune_epochs"]
            learning_rate = self.configs["fine_tune_learning_rate"]
        else:
            epochs = self.configs["epochs"]
            learning_rate = self.configs["learning_rate"]

        # Initialize optimizer - if mixed_grad is not None, optimizer will only update the parameters that require gradients
        if mixed_grad:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate,
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if use_wandb:
            # self.initialize_wandb(use_local_server=use_local_wandb)

            # if fine_tune:
            #     wandb.init(project="MMSD", resume="allow", name=name_wandb)
            # else:
            if name_wandb is None:
                wandb.init(project="MMSD", config=self.configs)
            else:
                wandb.init(project="MMSD", config=self.configs, name=name_wandb)
            wandb.watch(self.model, log="all", log_freq=5)

        # Early stopping initialization
        best_metric = (
            float("inf") if self.configs["early_stopping_metric"] == "loss" else 0
        )
        best_model_state = None
        steps_without_improvement = 0
        early_exit = False

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{epochs}",
            )

            for step, data in progress_bar:
                if epoch == 0 and step == 10:
                    break
                    # self.print_model_summary(6, 32)
                    pass

                # Reset attention cache if new segment
                if fine_tune:
                    batch_x, batch_y, new_segment_flag = data
                    if new_segment_flag:
                        self.model.reset_attention_cache()
                else:
                    batch_x, batch_y = data
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}

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
                progress_bar.set_postfix(accuracy=accuracy, loss=loss.item())

                # Log intermediate training metrics and perform early stopping check
                if use_wandb and step % (len(train_loader) // 10) == 0:
                    if step % val_freq_per_epoch == 0:
                        val_metrics = self.validate(val_loader, loss_func)
                        wandb.log(
                            {
                                "Validation Loss": val_metrics[self.model.NAME]["loss"],
                                "Validation Accuracy": val_metrics[self.model.NAME][
                                    "accuracy"
                                ],
                            }
                        )

                    train_acc = epoch_correct / epoch_total
                    train_loss = epoch_loss / (step + 1)

                    # Log intermediate training and validation metrics to wandb
                    global_step = epoch * len(train_loader) + step
                    wandb.log(
                        {
                            "Train Loss": train_loss,
                            "Train Accuracy": train_acc,
                            "Step": global_step,
                        }
                    )

                    # Early stopping check
                    if self.configs["early_stopping"]:
                        current_metric = (
                            val_metrics[self.model.NAME]["loss"]
                            if self.configs["early_stopping_metric"] == "loss"
                            else -val_metrics[self.model.NAME]["accuracy"]
                        )
                        if current_metric <= best_metric:
                            best_metric = current_metric
                            steps_without_improvement = 0
                            best_model_state = self.model.state_dict().copy()
                        else:
                            steps_without_improvement += 1
                            if (
                                steps_without_improvement
                                >= self.configs["early_stopping_patience"]
                            ):
                                print(
                                    f"Early stopping triggered at epoch {epoch + 1}, step {step + 1}"
                                )
                                self.model.load_state_dict(best_model_state)
                                early_exit = True
                                break

            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_correct / epoch_total
            print(
                f'Epoch: {epoch + 1}, | training loss: {avg_loss:.4f}, | training accuracy: {avg_acc:.4f} | validation loss: {val_metrics[self.model.NAME]["loss"]:.4f} | validation accuracy: {val_metrics[self.model.NAME]["accuracy"]:.4f}'
            )

            # Log end of epoch metrics to wandb
            if use_wandb:
                wandb.log(
                    {
                        "Train Loss": avg_loss,
                        "Train Accuracy": avg_acc,
                        "Validation Loss": val_metrics[self.model.NAME]["loss"],
                        "Validation Accuracy": val_metrics[self.model.NAME]["accuracy"],
                        "Epoch": epoch + 1,
                    }
                )

            if early_exit:
                break

        # Save the final model
        final_save_path = f"{self.save_path}/checkpoint_final.pth"
        directory = os.path.dirname(final_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), final_save_path)

        return final_save_path

    def validate(
        self,
        val_loader,
        loss_func,
        ckpt_path=None,
        subject_id=None,
        pre_trained_run=False,
        fine_tune_run=False,
        check_overlap=False,
    ):
        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=torch.device(self.device))
            )

        self.model.eval()
        y_true = []
        y_pred = []

        inference_times = []
        epoch_loss = 0.0
        self.label_buffer = []
        num_overlaps_detected = 0
        num_non_overlaps_detected = 0
        num_incorrect_predictions_with_overlap = 0
        num_incorrect_predictions_without_overlap = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                labels = batch_y.to(self.device)

                # Measure inference time for each batch
                if self.device == "cuda":
                    torch.cuda.synchronize()  # Synchronize CUDA operations before starting the timer
                start_time = time.time()
                final_output = self.model(inputs)
                if self.print_summary:
                    self.print_summary = False
                    self.batch_size = -1
                    self.print_model_summary(1, -1)
                if self.device == "cuda":
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
                y_true.extend(
                    labels.cpu().numpy() - 1
                )  # correct for labelling starting from index `1`
                y_pred.extend(preds.cpu().numpy())

                if check_overlap:
                    # check if prediction is incorrect
                    if y_true[-1] != y_pred[-1]:
                        incorrect_pred = True
                    else:
                        incorrect_pred = False

                    # Check for label overlaps in the last n segment lengths
                    self.label_buffer.extend(labels.cpu().numpy())
                    if len(self.label_buffer) > self.model.source_seq_length:
                        self.label_buffer = self.label_buffer[-self.model.source_seq_length :]

                    overlap_detected = self._check_label_overlap()
                    if overlap_detected:
                        num_overlaps_detected += 1
                        if incorrect_pred:
                            num_incorrect_predictions_with_overlap += 1
                    else:
                        num_non_overlaps_detected += 1
                        if incorrect_pred:
                            num_incorrect_predictions_without_overlap += 1

        if check_overlap:
            # After validation loop
            print("Number of overlaps detected: {}".format(num_overlaps_detected))
            print(
                "Number of non-overlaps detected: {}".format(num_non_overlaps_detected)
            )
            print(
                "Number of incorrect predictions with overlap: {}".format(
                    num_incorrect_predictions_with_overlap
                )
            )
            print(
                "Number of incorrect predictions without overlap: {}".format(
                    num_incorrect_predictions_without_overlap
                )
            )

        # Calculate average inference time in milliseconds
        avg_inference_time = np.mean(inference_times)
        avg_loss = epoch_loss / len(val_loader)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
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

    def measure_inference_time(
        self, val_loader, warmup_batches=20, repetitions=1000, ckpt_path=None
    ):
        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=torch.device(self.device))
            )

        self.model.eval()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            # Warm up the GPU
            for i, (batch_x, batch_y) in enumerate(val_loader):
                if i >= warmup_batches:
                    break
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                _ = self.model(inputs)
                if self.device == "cuda":
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

    def _check_label_overlap(self):
        # Convert self.label_buffer to a set to find unique labels
        unique_labels = set(self.label_buffer)

        # If the number of unique labels is less than self.model.source_seq_length, overlap detected
        if len(unique_labels) > 1:
            return True
        else:
            return False

    def initialize_wandb(self, use_local_server):
        if use_local_server:
            os.environ["WANDB_BASE_URL"] = "http://localhost:8081"
        else:
            os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
