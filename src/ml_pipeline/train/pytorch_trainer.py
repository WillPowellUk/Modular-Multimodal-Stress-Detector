import torch
import torch.nn as nn
import torch.optim as optim

class PytorchTrainer:
    def __init__(self, model, train_loader, val_loader=None, use_gpu=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                if self.use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader)}")
            
            if self.val_loader:
                self.validate()
                
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                if self.use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Loss: {val_loss/len(self.val_loader)}, Accuracy: {100 * correct / total}%")
