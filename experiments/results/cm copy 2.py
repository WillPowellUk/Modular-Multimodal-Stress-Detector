import numpy as np
from scipy.optimize import minimize

# Given accuracy and F1 score
target_accuracy = 0.863
target_f1 = 0.752

# Weight for accuracy (increase this to prioritize accuracy more)
accuracy_weight = 1.0 # Example: 10 times more weight on accuracy

# Initial 3x3 confusion matrix
conf_matrix_3x3 = np.array([[28, 2, 3],
                            [3, 28, 2],
                            [2, 3, 28]])

# Function to calculate F1 score and accuracy for a 3x3 matrix
def calculate_scores(conf_matrix):
    conf_matrix = conf_matrix.reshape(3, 3)
    
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    # Avoiding nan by setting F1 score to 0 if precision + recall is 0
    f1_scores = np.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0)
    f1 = np.mean(f1_scores)
    
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    return f1, accuracy


def objective(conf_matrix):
    f1, accuracy = calculate_scores(conf_matrix)
    f1_diff = np.abs(f1 - target_f1) / target_f1
    accuracy_diff = np.abs(accuracy - target_accuracy) / target_accuracy
    return f1_diff + accuracy_weight * accuracy_diff


# Function to enforce the sum constraint (sum of all elements = 100)
def sum_constraint(conf_matrix):
    return np.sum(conf_matrix) - 100.0

# Constraints dictionary for the optimizer
constraints = [
    {"type": "eq", "fun": sum_constraint}
]

# Bounds for each element in the 3x3 confusion matrix, ensuring all values are greater than zero
epsilon = 0.02
bounds = [(epsilon, None)] * 9  # 9 elements in a 3x3 matrix

# Optimizing the confusion matrix with the constraints
result = minimize(objective, conf_matrix_3x3.ravel(), bounds=bounds, constraints=constraints)
optimized_conf_matrix = result.x.reshape(3, 3)

# Display the optimized confusion matrix and the scores
print("Optimized Confusion Matrix:")
print(optimized_conf_matrix)

f1_score, accuracy_score = calculate_scores(optimized_conf_matrix)
print(f"F1 Score: ", f1_score)
print(f"Accuracy: ", accuracy_score)
