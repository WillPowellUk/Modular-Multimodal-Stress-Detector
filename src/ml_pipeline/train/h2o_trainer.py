import h2o
import json
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2OGeneralizedLinearEstimator
import pandas as pd
import numpy as np

class H2OTrainer:
    def __init__(self, config_path, train_loader, val_loader=None, target_column='target'):
        h2o.init()
        self.train_data = self._loader_to_h2o_frame(train_loader, target_column)
        self.val_data = self._loader_to_h2o_frame(val_loader, target_column) if val_loader is not None else None
        self.target_column = target_column
        self.features = [col for col in self.train_data.columns if col != target_column]
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            return json.load(file)
        
    def _loader_to_h2o_frame(self, loader, target_column):
        if loader is None:
            return None
        
        data_list = []
        for inputs, labels in loader:
            inputs = inputs.numpy()
            labels = labels.numpy().reshape(-1, 1)
            combined = np.hstack((inputs, labels))
            data_list.append(combined)
        
        data_np = np.vstack(data_list)
        columns = [f'feature_{i}' for i in range(data_np.shape[1] - 1)] + [target_column]
        data_df = pd.DataFrame(data_np, columns=columns)
        return h2o.H2OFrame(data_df)
        
    def train(self):
        for model_config in self.config['models']:
            model_type = model_config['type']
            hyperparameters = model_config['hyperparameters']
            model = self._initialize_model(model_type, hyperparameters)
            self.model = model
            self.model.train(x=self.features, y=self.target_column, training_frame=self.train_data)
            print(f"Training complete for model: {model_type}")
            
            if self.val_data is not None:
                self.evaluate()
        
    def _initialize_model(self, model_type, hyperparameters):
        match model_type:
            case 'random_forest':
                return H2ORandomForestEstimator(**hyperparameters)
            case 'gbm':
                return H2OGradientBoostingEstimator(**hyperparameters)
            case 'glm':
                return H2OGeneralizedLinearEstimator(**hyperparameters)
            case _:
                raise ValueError(f"Model type '{model_type}' is not supported.")

        
    def evaluate(self):
        if self.val_data is None:
            raise ValueError("Validation data not provided.")
        
        performance = self.model.model_performance(self.val_data)
        print(performance)

    def predict(self, test_loader):
        test_data = self._loader_to_h2o_frame(test_loader, self.target_column)
        predictions = self.model.predict(test_data)
        return predictions.as_data_frame()

    def shutdown(self):
        h2o.shutdown(prompt=False)
