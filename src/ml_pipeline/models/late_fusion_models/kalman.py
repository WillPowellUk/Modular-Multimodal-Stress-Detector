import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    def __init__(self, num_classes, num_branches, device="cpu"):
        super(KalmanFilter, self).__init__()
        self.num_classes = num_classes
        self.num_branches = num_branches
        self.device = device
        # Learnable parameters
        self.F = nn.Parameter(torch.eye(num_classes, dtype=torch.float32))
        self.H = nn.Parameter(torch.eye(num_classes, dtype=torch.float32))
        self.Q = nn.Parameter(
            torch.diag(torch.ones(num_classes, dtype=torch.float32) * 1e-4)
        )

        # Initial state and covariance
        self.initial_x = nn.Parameter(torch.zeros(num_classes, 1, dtype=torch.float32))
        self.initial_P = nn.Parameter(
            torch.eye(num_classes, dtype=torch.float32) * 0.01
        )
        if num_classes == 3:
            self.epsilon = nn.Parameter(torch.tensor(0.4))
            self.gamma = nn.Parameter(
                torch.tensor([0.278, 1.0, 1.0], dtype=torch.float32).reshape(-1, 1)
            )
        elif num_classes == 2:
            self.epsilon = nn.Parameter(torch.tensor(0.7))
            self.gamma = nn.Parameter(
                torch.tensor([0.667, 1.1], dtype=torch.float32).reshape(-1, 1)
            )
        else:
            raise ValueError("Only 2-class and 3-class problems are supported")

    def predict(self, x, P):
        x = self.F @ x
        P = self.F @ P @ self.F.t() + self.Q
        return x, P

    def update(self, x, P, z):
        y = z - self.H @ x

        # Calculate R dynamically based on z
        if self.num_classes == 3:
            R = ((1 - z) * 2 * torch.eye(self.num_classes, device=self.device)) ** 2
        elif self.num_classes == 2:
            R = (((1 - z) / 2) * torch.eye(self.num_classes, device=self.device)) ** 2

        S = self.H @ P @ self.H.t() + R
        K = P @ self.H.t() @ torch.inverse(S)

        x = x + K @ y
        P = (torch.eye(self.num_classes, device=self.device) - K @ self.H) @ P
        return x, P

    def forward(self, z_batch):
        batch_size, num_branches, output_dim = z_batch.shape
        output = torch.zeros(
            (batch_size, output_dim), dtype=torch.float32, device=self.device
        )

        for i in range(batch_size):
            x = self.initial_x.clone()
            P = self.initial_P.clone()

            for j in range(num_branches):
                z = z_batch[i, j].reshape(-1, 1)

                if z.max() > self.epsilon:
                    z = z * self.gamma
                    x, P = self.predict(x, P)
                    x, P = self.update(x, P, z)

            output[i] = x.flatten()

        return output
