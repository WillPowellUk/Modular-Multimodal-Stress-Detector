import torch

class KalmanFilter:
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.x = torch.tensor([0.8] + [0.1] * (num_classes - 1), dtype=torch.float32).reshape(-1, 1).to(device)
        self.P = (0.01 * torch.eye(num_classes, dtype=torch.float32)).to(device)
        self.F = torch.eye(num_classes, dtype=torch.float32).to(device)
        self.H = torch.eye(num_classes, dtype=torch.float32).to(device)
        self.Q = (5e-4 * torch.eye(num_classes, dtype=torch.float32)).to(device)
        
        if num_classes == 3:
            self.epsilon = 0.4
            self.gamma = torch.tensor([0.278, 1, 1], dtype=torch.float32).reshape(-1, 1).to(device)
        elif num_classes == 2:
            self.epsilon = 0.7
            self.gamma = torch.tensor([0.667, 1.1], dtype=torch.float32).reshape(-1, 1).to(device)
        else:
            raise ValueError("Only 2-class and 3-class problems are supported")
    
    def process_measurement_noise(self, z):
        R = ((1 - z) * 2 * torch.eye(self.num_classes, dtype=torch.float32).to(self.device)) ** 2
        return R
    
    def update(self, z):
        R = self.process_measurement_noise(z)
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ torch.inverse(S)
        
        # Update state estimate and error covariance
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (torch.eye(self.num_classes, device=self.device) - K @ self.H) @ self.P
    
    def forward(self, z_batch):
        device = z_batch.device
        batch_size, num_branches, output_dim = z_batch.shape
        output = torch.zeros((batch_size, output_dim), dtype=torch.float32, device=device)
        
        for i in range(batch_size):
            self.x = torch.tensor([0.8] + [0.1] * (self.num_classes - 1), dtype=torch.float32).reshape(-1, 1).to(device)
            self.P = (0.01 * torch.eye(self.num_classes, dtype=torch.float32)).to(device)
            
            for j in range(num_branches):
                z = z_batch[i, j].reshape(-1, 1)
                
                if z.max() > self.epsilon:
                    z = z * self.gamma
                    self.update(z)
            
            output[i] = self.x.flatten()
        
        return output
    