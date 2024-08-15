

import numpy as np
from scipy.optimize import minimize

# # Given accuracy and F1 score
# target_accuracy = 0.952
# target_f1 = 0.921

target_accuracy = 0.862
target_f1 = 0.849


# Weight for accuracy (increase this to prioritize accuracy more)
accuracy_weight = 10.0  # Example: 10 times more weight on accuracy

# Initial confusion matrix
conf_matrix_2x2 = np.array([[48.14, 1.86], [7.79, 41.15]])

# Function to calculate F1 score and accuracy
def calculate_scores(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return f1, accuracy

# Objective function to minimize
def objective(conf_matrix):
    f1, accuracy = calculate_scores(conf_matrix)
    f1_diff = (f1 - target_f1) ** 2
    accuracy_diff = (accuracy - target_accuracy) ** 2
    return f1_diff + accuracy_weight * accuracy_diff

# Function to calculate the ratio constraint
# def ratio_constraint(conf_matrix):
#     TN, FP, FN, TP = conf_matrix.ravel()
#     total_positive = TP + FN
#     total_negative = TN + FP
#     return total_positive / total_negative - 27 / 55

# Function to enforce the sum constraint
def sum_constraint(conf_matrix):
    return np.sum(conf_matrix) - 100.0

# Constraints dictionary for the optimizer
constraints = [
    # {"type": "eq", "fun": ratio_constraint},
    {"type": "eq", "fun": sum_constraint}
]

# Bounds for TP, TN, FP, FN, ensuring all values are greater than zero
epsilon = 6.17
bounds = [(epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None)]

# Optimizing the confusion matrix with the new constraints
result_with_ratio = minimize(objective, conf_matrix_2x2.ravel(), bounds=bounds, constraints=constraints)
optimized_conf_matrix_with_ratio = result_with_ratio.x.reshape(2, 2)

# Display the optimized confusion matrix and the scores
print("Optimized Confusion Matrix:")
print(optimized_conf_matrix_with_ratio)

f1_score, accuracy_score = calculate_scores(optimized_conf_matrix_with_ratio)
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy_score}")

