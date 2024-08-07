import torch
import torch.nn as nn


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
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.ecg_loss_func = nn.BCEWithLogitsLoss()
        self.eda_loss_func = nn.BCEWithLogitsLoss()
        self.both_loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        output_ecg,
        output_eda,
        output_both,
        target,
        missingFlag_ecg,
        missingFlag_eda,
    ):
        both_flag = torch.logical_and(missingFlag_ecg, missingFlag_eda)
        loss_ecg = self.ecg_loss_func(output_ecg, target)
        loss_eda = self.ecg_loss_func(output_eda, target)
        loss_both = self.ecg_loss_func(output_both, target)

        loss = (
            missingFlag_ecg * loss_ecg
            + missingFlag_eda * loss_eda
            + both_flag * loss_both
        )
        return self._reduce(loss)

    def _reduce(self, x):
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            return x


def modalityFusionLoss(output, target, missingFlag_ecg, missingFlag_eda):
    """
    missingFlag for ecg and EDA are the boolean values that
    indicate whether the corresponding sample of ecg or EDA is missing
    """
    both_flag = torch.logical_and(missingFlag_ecg, missingFlag_eda)
    loss = nn.BCEWithLogitsLoss()
    loss = torch.mean((output - target) ** 2)
    return loss
