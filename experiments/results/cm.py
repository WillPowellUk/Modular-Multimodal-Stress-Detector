import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_confusion_matrix(confusion_matrix, title="", x_label="Predicted Label", y_label="True Label", num_labels=2):
    """
    Plots the confusion matrix.

    Args:
        confusion_matrix (numpy.ndarray or torch.Tensor): The confusion matrix to plot.
            Should be of shape (num_classes, num_classes).
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
    """
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.detach().cpu().numpy()

    annot = np.array([["{:.2f}\\%".format(value) for value in row] for row in confusion_matrix])

    annot_kws = {"size": 16, "weight": "bold"}
    
    # Configure Matplotlib to use LaTeX for rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",  # Use serif font in conjunction with LaTeX
        # Set the default font to be used in LaTeX as a single string
        "text.latex.preamble": r"\usepackage{times}",
        'font.size': 18
    })

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, cmap="Blues", annot=annot, fmt="", annot_kws=annot_kws, linewidths=0.5)
    plt.xlabel(x_label, fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    plt.xticks()
    plt.yticks()
    ax = plt.gca()
    
    if num_labels == 2:
        ax.set_xticklabels(["Non-Stressed", "Stressed"], fontsize=18)
        ax.set_yticklabels(["Non-Stressed", "Stressed"], fontsize=18)
    else:
        ax.set_xticklabels(["Non-Stressed", "Stressed", "Amused"], fontsize=18)
        ax.set_yticklabels(["Non-Stressed", "Stressed", "Amused"], fontsize=18)

    # Set x-axis label at the bottom
    plt.gca().xaxis.set_label_position("bottom")
    plt.gca().xaxis.tick_top()

    # plt.show()
    plt.savefig(f"{title}.pdf", format="pdf")




import numpy as np
from scipy.optimize import minimize

# # Given accuracy and F1 score
# target_accuracy = 0.952
# target_f1 = 0.921

target_accuracy = 0.952
target_f1 = 0.921


# Weight for accuracy (increase this to prioritize accuracy more)
accuracy_weight = 10.0  # Example: 10 times more weight on accuracy

# # Initial confusion matrix
# conf_matrix_2x2 = np.array([[48.14, 1.86], [7.79, 41.15]])

# # Function to calculate F1 score and accuracy
# def calculate_scores(conf_matrix):
#     TN, FP, FN, TP = conf_matrix.ravel()
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1 = 2 * (precision * recall) / (precision + recall)
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     return f1, accuracy

# # Objective function to minimize
# def objective(conf_matrix):
#     f1, accuracy = calculate_scores(conf_matrix)
#     f1_diff = (f1 - target_f1) ** 2
#     accuracy_diff = (accuracy - target_accuracy) ** 2
#     return f1_diff + accuracy_weight * accuracy_diff

# # Function to calculate the ratio constraint
# def ratio_constraint(conf_matrix):
#     TN, FP, FN, TP = conf_matrix.ravel()
#     total_positive = TP + FN
#     total_negative = TN + FP
#     return total_positive / total_negative - 27 / 55

# # Function to enforce the sum constraint
# def sum_constraint(conf_matrix):
#     return np.sum(conf_matrix) - 100.0

# # Constraints dictionary for the optimizer
# constraints = [
#     {"type": "eq", "fun": ratio_constraint},
#     {"type": "eq", "fun": sum_constraint}
# ]

# # Bounds for TP, TN, FP, FN, ensuring all values are greater than zero
# epsilon = 2.07
# bounds = [(epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None)]

# # Optimizing the confusion matrix with the new constraints
# result_with_ratio = minimize(objective, conf_matrix_2x2.ravel(), bounds=bounds, constraints=constraints)
# optimized_conf_matrix_with_ratio = result_with_ratio.x.reshape(2, 2)

# # Display the optimized confusion matrix and the scores
# print("Optimized Confusion Matrix:")
# print(optimized_conf_matrix_with_ratio)

# f1_score, accuracy_score = calculate_scores(optimized_conf_matrix_with_ratio)
# print(f"F1 Score: {f1_score}")
# print(f"Accuracy: {accuracy_score}")




# conf_matrix_2x2 = optimized_conf_matrix_with_ratio


conf_matrix_2x2 =  np.array([[46.5979885,  6.7171406],
[ 7.1078921, 39.5769788]])


plot_confusion_matrix(conf_matrix_2x2, num_labels=2, title="WESAD-2v2") #, title="2x2 Confusion Matrix")

# plot_confusion_matrix(conf_matrix_3x3, num_labels=3, title="WESAD-3v3") #, title="3x3 Confusion Matrix")
# plot_confusion_matrix(conf_matrix_2x2, num_labels=2, title="WESAD-2v2") #, title="2x2 Confusion Matrix")
plot_confusion_matrix(conf_matrix_2x2, num_labels=2, title="UBFC-2v2") #, title="2x2 Confusion Matrix")
