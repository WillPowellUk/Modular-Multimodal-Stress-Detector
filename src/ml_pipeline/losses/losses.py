import torch
import torch.nn as nn
from src.ml_pipeline.utils import get_active_key


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, target):
        # Ensure y_pred is in the same shape as target
        if y_pred.size() != target.size():
            raise ValueError(
                f"y_pred size ({y_pred.size()}) must be the same as target size ({target.size()})"
            )

        # Apply softmax to get probabilities
        y_pred = y_pred.sigmoid()

        # Calculate the focal loss
        pt = y_pred * target + (1 - y_pred) * (1 - target)
        log_pt = torch.log(pt)
        focal_loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LossWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss() if "bce" in config and config["bce"] else None
        self.focal = FocalLoss() if "focal" in config and config["focal"] else None

    def forward(self, y_pred, target):
        loss = 0.0
        if self.bce:
            loss += self.bce(y_pred, target)
        if self.focal:
            loss += self.focal(y_pred, target)
        return loss
