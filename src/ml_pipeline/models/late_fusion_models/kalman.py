import torch
import torch.nn as nn

class KalmanFilterPredictor(nn.Module):
    def __init__(self, state_dim, observation_dim):
        super(KalmanFilterPredictor, self).__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim

        # Initialize Kalman filter parameters
        self.F = nn.Parameter(torch.eye(state_dim))  # State transition matrix
        self.H = nn.Parameter(torch.eye(observation_dim, state_dim))  # Observation matrix
        self.Q = nn.Parameter(torch.eye(state_dim) * 0.01)  # Process noise covariance
        self.R = nn.Parameter(torch.eye(observation_dim) * 0.01)  # Observation noise covariance
        self.P = nn.Parameter(torch.eye(state_dim))  # Initial estimate error covariance
        self.x = nn.Parameter(torch.zeros(state_dim))  # Initial state estimate

    def forward(self, concatenated_features):
        batch_size, embed_dim, n_branches = concatenated_features.size()
        # Reshape concatenated features to (batch_size, observation_dim)
        observations = concatenated_features.view(batch_size, -1, self.observation_dim)

        outputs = []
        for t in range(observations.size(1)):
            z = observations[:, t, :]

            # Predict step
            self.x = torch.matmul(self.F, self.x)
            self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.T) + self.Q

            # Update step
            y = z - torch.matmul(self.H, self.x)  # Measurement residual
            S = torch.matmul(torch.matmul(self.H, self.P), self.H.T) + self.R  # Residual covariance
            K = torch.matmul(torch.matmul(self.P, self.H.T), torch.inverse(S))  # Kalman gain
            self.x = self.x + torch.matmul(K, y)
            self.P = self.P - torch.matmul(torch.matmul(K, self.H), self.P)

            outputs.append(self.x)

        # Stack the outputs for each timestep
        final_output = torch.stack(outputs, dim=1)
        return final_output

# # Assuming `net["predictor"]` is an instance of KalmanFilterPredictor
# # Initialize your KalmanFilterPredictor
# state_dim = 10  # Define according to your requirements
# observation_dim = concatenated_features.size(1)  # Example: embedding dimension * number of branches
# net["predictor"] = KalmanFilterPredictor(state_dim, observation_dim)

# # Forward pass
# concatenated_features = torch.cat(list(modality_features.values()), dim=1)
# concatenated_features = concatenated_features.permute(
#     0, 2, 1
# )  # change to shape (batch_size, embed_dim, n_branches)
# final_output = net["predictor"](concatenated_features)
