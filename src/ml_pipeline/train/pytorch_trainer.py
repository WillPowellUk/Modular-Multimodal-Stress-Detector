import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, log_loss
from src.ml_pipeline.models.attention_models.san_losses import LossWrapper, FocalLoss
from src.ml_pipeline.utils import print_model_summary

class PyTorchTrainer:
    def __init__(self, model, train_loader, val_loader, config_path, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.configs = self.load_config(config_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['learning_rate'])
        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.num_classes = self.configs['num_classes']
        self.save_path = self.configs['save_path']

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs

    def print_model_summary(self):
        print_model_summary(self.model, self.model.input_dims, batch_size=self.train_loader.batch_size, device=self.device.type)

    def train(self):
        self.writer = SummaryWriter(log_dir=f'{self.save_path}/tensorboard')  # TensorBoard writer
        print(f"Storing tensorboard log to: {self.writer.log_dir}")
        for epoch in range(self.configs['epoch']):
            epoch_loss = 0.0
            self.model.train()
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.configs["epoch"]}')
            
            for s, (batch_x, batch_y) in progress_bar:
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                labels = batch_y.to(self.device)
                modality_outputs, final_output = self.model(inputs)
                one_hot_labels = torch.nn.functional.one_hot(labels - 1, num_classes=self.num_classes).float()
                loss = self.loss_func(final_output, one_hot_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f'Epoch: {epoch}, | training loss: {avg_loss:.4f}')
            
            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/Train', avg_loss, epoch)

            # Validate and log validation loss and accuracy
            val_metrics = self.validate()
            self.writer.add_scalar('Loss/Validation', val_metrics[self.model.NAME]['loss'], epoch)
            if 'accuracy' in val_metrics[self.model.NAME]:
                self.writer.add_scalar('Accuracy/Validation', val_metrics[self.model.NAME]['accuracy'], epoch)

            if (epoch + 1) % 10 == 0:
                save_path = f'{self.save_path}/checkpoint_{epoch + 1}.pth'
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(self.model.state_dict(), save_path)

        final_save_path = f'{self.save_path}/checkpoint_{epoch + 1}.pth'
        directory = os.path.dirname(final_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), final_save_path)
        self.writer.close()
        return final_save_path


    def validate(self, ckpt_path=None):
        if self.val_loader is None:
            raise ValueError("Validation data loader is not provided")

        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        
        self.model.eval()
        y_true = []
        y_pred = []
        y_logits = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                inputs = {key: val.to(self.device) for key, val in batch_x.items()}
                labels = batch_y.to(self.device)
                modality_outputs, final_output = self.model(inputs)
                y_logits.extend(final_output.cpu().numpy())
                
                _, preds = torch.max(final_output, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true) - 1 # correct for labelling starting from index `1`
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(self.num_classes)])
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        one_hot_labels = torch.nn.functional.one_hot(labels-1, num_classes=self.num_classes).float()
        loss = self.loss_func(torch.tensor(y_logits).to(self.device), one_hot_labels).cpu().item()

        results = {}
        results[self.model.NAME] = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": loss
        }

        return results