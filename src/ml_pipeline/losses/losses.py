import torch
import torch.nn as nn
from src.ml_pipeline.utils import get_active_key


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Reduction {} not implemented.".format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        p_t = torch.where(target == 1, x, 1 - x)
        fl = -1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(target == 1, fl * self.alpha, fl)
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            return x


class LossWrapper(nn.Module):
    def __init__(self, CONFIG_PATH):
        super().__init__()
        self.loss_fns = get_active_key(CONFIG_PATH, "loss_fns")
        if "bce" in self.loss_fns:
            self.bce = nn.BCEWithLogitsLoss()
        if "focal" in self.loss_fns:
            self.focal = FocalLoss()

    def forward(self, y_pred, target):
        loss = 0.0
        if "bce" in self.loss_fns:
            loss += self.bce(y_pred, target)
        if "focal" in self.loss_fns:
            loss += self.focal(y_pred, target)
        return loss
