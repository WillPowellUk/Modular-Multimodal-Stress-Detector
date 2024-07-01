import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)


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

        with open(self.config_path, "r") as file:
            config = json.load(file)

        for model_cfg in config.get("models", []):
            model_type = model_cfg.get("type")
            hyperparameters = model_cfg.get("hyperparameters", {})

            match model_type:
                case "random_forest":
                    n_estimators = hyperparameters.get("n_estimators", 100)
                    max_depth = hyperparameters.get("max_depth")
                    min_samples_split = hyperparameters.get("min_samples_split", 2)
                    self.models.append(
                        {
                            "model": RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                            ),
                            "description": f"RF_est_{n_estimators}_depth_{max_depth}_minsplit_{min_samples_split}",
                        }
                    )
                case "gbm":
                    n_estimators = hyperparameters.get("ntrees", 100)
                    max_depth = hyperparameters.get("max_depth", 3)
                    learning_rate = hyperparameters.get("learn_rate", 0.1)
                    self.models.append(
                        {
                            "model": GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                            ),
                            "description": f"GBM_est_{n_estimators}_depth_{max_depth}_lr_{learning_rate}",
                        }
                    )
                case "glm":
                    penalty = "elasticnet"
                    l1_ratio = hyperparameters.get("alpha", 0.5)
                    C = hyperparameters.get("lambda", 1.0)
                    solver = "saga"
                    self.models.append(
                        {
                            "model": LogisticRegression(
                                penalty=penalty, l1_ratio=l1_ratio, C=C, solver=solver
                            ),
                            "description": f"GLM_pen_{penalty}_alpha_{l1_ratio}_lambda_{C}_solver_{solver}",
                        }
                    )
                case "svm":
                    C = hyperparameters.get("C", 1.0)
                    kernel = hyperparameters.get("kernel", "rbf")
                    gamma = hyperparameters.get("gamma", "scale")
                    self.models.append(
                        {
                            "model": SVC(C=C, kernel=kernel, gamma=gamma),
                            "description": f"SVM_C_{C}_kernel_{kernel}_gamma_{gamma}",
                        }
                    )
                case _:
                    print(f"Unknown model type: {model_type}")

    def _loader_to_numpy(self, loader):
        X = []
        y = []
        for data, target in loader:
            X.append(data.numpy().reshape(data.shape[0], -1))  # Flatten the input
            y.append(target.numpy())
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    def tune_hyperparameters(self, n_jobs=None, cv=None, verbose=1):
        X_train, y_train = self._loader_to_numpy(self.train_loader)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file {self.config_path} does not exist")

        with open(self.config_path, "r") as file:
            config = json.load(file)

        for i, model_cfg in enumerate(config.get("models", [])):
            model_type = model_cfg.get("type")

            match model_type:
                case "random_forest":
                    param_grid = {
                        "n_estimators": [50, 100, 150],
                        "max_depth": [10, 20, 30, None],
                        "min_samples_split": [2, 5, 10],
                    }
                    model = RandomForestClassifier()
                case "gbm":
                    param_grid = {
                        "n_estimators": [50, 100, 150],
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1, 0.2],
                    }
                    model = GradientBoostingClassifier()
                case "glm":
                    param_grid = {
                        "C": [0.1, 1.0, 10.0],
                        "l1_ratio": [0.1, 0.5, 0.9],
                        "solver": ["saga"],
                        "penalty": ["elasticnet"],
                    }
                    model = LogisticRegression()
                case "svm":
                    param_grid = {
                        "C": [0.1, 1.0, 10.0],
                        "kernel": ["linear", "rbf"],
                        "gamma": ["scale", "auto"],
                    }
                    model = SVC()
                case _:
                    print(
                        f"Hyperparameter tuning not implemented for model type: {model_type}"
                    )
                    continue

            print(f"Tuning hyperparameters for {model_type}")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="accuracy",
                cv=cv,
                n_jobs=n_jobs,
                verbose=2,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            description = f"{model_type.upper()}_best_" + "_".join(
                [f"{key}_{value}" for key, value in best_params.items()]
            )
            self.models[i]["model"] = best_model

            print(f"Best hyperparameters for {model_type}: {best_params}")
            print(f"Best cross-validated accuracy: {best_score:.5f}")

        return self.models

    def train(self):
        X_train, y_train = self._loader_to_numpy(self.train_loader)
        for i, model_info in enumerate(self.models):
            model = model_info["model"]
            print("Training model:", model_info["description"])
            model.fit(X_train, y_train)
            self.models[i]["model"] = model
        return self.models

    def validate(self, trained_models):
        if self.val_loader is None:
            raise ValueError("Validation data loader is not provided")

        X_val, y_val = self._loader_to_numpy(self.val_loader)
        results = {}
        for model_info in trained_models:
            model = model_info["model"]
            print("Validating model:", model_info["description"])
            y_pred = model.predict(X_val)
            y_pred_proba = (
                model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
            )

            accuracy = accuracy_score(y_val, y_pred)
            conf_matrix = confusion_matrix(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average="weighted")
            recall = recall_score(y_val, y_pred, average="weighted")
            f1 = f1_score(y_val, y_pred, average="weighted")
            loss = log_loss(y_val, y_pred_proba) if y_pred_proba is not None else None

            results[model_info["description"]] = {
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "loss": loss,
            }
        return results
