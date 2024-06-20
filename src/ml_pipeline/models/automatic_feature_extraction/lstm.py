import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Model(torch.nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        
        self.lstm0 = nn.LSTM(
            input_size = 24,
            hidden_size = 64,
            num_layers = 3,
            batch_first = True,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, 256)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm0(x, None)
        bn_out = F.dropout(self.bn1(lstm_out[:, -1, :]), 0.3)
        fc_out = F.relu(F.dropout(self.fc(bn_out), 0.5))
        out = self.out(fc_out)
        return out
