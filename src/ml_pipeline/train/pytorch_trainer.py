import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, log_loss
from src.ml_pipeline.models.san.san_losses import LossWrapper, FocalLoss

class PyTorchTrainer:
    def __init__(self, model, train_loader, val_loader, config_path, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.configs = self.load_config(config_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['LR'])
        self.loss_func = FocalLoss()

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs

    def train(self):
        for epoch in range(self.configs['EPOCH']):
            epoch_loss = 0.0
            self.model.train()
            for s, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.unsqueeze(1).to(self.device)
                output_ecg, output_eda, output_both = self.model(batch_x)
                loss = self.loss_func(output_ecg, output_eda, output_both, batch_y, atch_missing_eda) #todo fix this
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f'Epoch: {epoch}, | training loss: {avg_loss:.4f}')
            
            if (epoch + 1) % 10 == 0:
                save_path = f'{self.configs["SAVE_PATH"]}/checkpoint_{epoch + 1}.pth'
                torch.save(self.model.state_dict(), save_path)

        final_save_path = f'{self.configs["SAVE_PATH"]}/checkpoint_{epoch + 1}.pth'
        torch.save(self.model.state_dict(), final_save_path)
        return final_save_path

    def validate(self, ckpt_path=None):
        if self.val_loader is None:
            raise ValueError("Validation data loader is not provided")

        # Load model from checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        
        self.model.eval()
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.unsqueeze(1).to(self.device)
                _, _, output_both = self.model(batch_x)
                y_pred = torch.round(torch.sigmoid(output_both))
                all_y_true.extend(batch_y.cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy())

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        loss = log_loss(y_true, y_pred) if y_pred.ndim == 1 else None

        results = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": loss
        }

        return results