import json
import os
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

class TraditionalMLTrainer:
    def __init__(self, config_path, train_loader, val_loader=None):
        self.config_path = config_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.models = []
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file {self.config_path} does not exist")
        
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        
        for model_cfg in config.get("models", []):
            model_type = model_cfg.get("type")
            hyperparameters = model_cfg.get("hyperparameters", {})
            
            match model_type:
                case "random_forest":
                    self.models.append(RandomForestClassifier(
                        n_estimators=hyperparameters.get("ntrees", 100),
                        max_depth=hyperparameters.get("max_depth"),
                        min_samples_split=hyperparameters.get("min_rows", 2)
                    ))
                case "gbm":
                    self.models.append(GradientBoostingClassifier(
                        n_estimators=hyperparameters.get("ntrees", 100),
                        max_depth=hyperparameters.get("max_depth", 3),
                        learning_rate=hyperparameters.get("learn_rate", 0.1)
                    ))
                case "glm":
                    self.models.append(LogisticRegression(
                        penalty='elasticnet',
                        l1_ratio=hyperparameters.get("alpha", 0.5),
                        C=hyperparameters.get("lambda", 1.0),
                        solver='saga'
                    ))
                case "svm":
                    self.models.append(SVC(
                        C=hyperparameters.get("C", 1.0),
                        kernel=hyperparameters.get("kernel", 'rbf'),
                        gamma=hyperparameters.get("gamma", 'scale')
                    ))
                case _:
                    print(f"Unknown model type: {model_type}")

    def _loader_to_numpy(self, loader):
        X = []
        y = []
        for data, target in loader:
            X.append(data.numpy().reshape(data.shape[0], -1))  # Flatten the input
            y.append(target.numpy())
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    def train(self):
        X_train, y_train = self._loader_to_numpy(self.train_loader)
        trained_models = []
        for model in self.models:
            print("Training model:", model.__class__.__name__)
            model.fit(X_train, y_train)
            trained_models.append(model)
        return trained_models

    def validate(self, trained_models):
        if self.val_loader is None:
            raise ValueError("Validation data loader is not provided")
        
        X_val, y_val = self._loader_to_numpy(self.val_loader)
        results = {}
        for model in trained_models:
            print("Validating model:", model.__class__.__name__)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results[model.__class__.__name__] = accuracy
        return results
