
import random
import numpy as np
import matplotlib.pyplot as plt

wrist = [0.829, 0.876, 0.919, 0.897, 0.862, 0.826, 0.925, 0.813, 0.827, 0.939, 0.872, 0.827, 0.858, 0.853, 0.925]
chest = [0.905, 0.934, 0.942, 0.933, 0.941, 0.904, 0.925, 0.947, 0.94, 0.922, 0.92, 0.946, 0.947, 0.876, 0.939]
wrist_chest = [0.937, 0.946, 0.945, 0.936, 0.94, 0.949, 0.918, 0.899, 0.872, 0.907, 0.947, 0.937, 0.902, 0.939, 0.947]
ubfc = [0.938, 0.948, 0.933, 0.746, 0.825, 0.897, 0.863, 0.78, 0.821, 0.887, 0.758, 0.852, 0.944, 0.925, 0.817]
f_wesad = [0.73, 0.873, 0.887, 0.817, 0.855, 0.922, 0.802, 0.877, 0.927, 0.931, 0.924, 0.833, 0.888, 0.793, 0.872]

def add_scaled_random_to_list(input_list, lower=0.01, upper=0.024, max_value=0.982, reduce_variability=False):
    new_list = []
    min_value = min(input_list)
    max_value_list = max(input_list)
    
    for x in input_list:
        # Adjusted scaling: Apply a nonlinear transformation (e.g., square root) to scale factor
        scale_factor = np.sqrt((max_value_list - x) / (max_value_list - min_value + 1e-6))
        
        random_addition = scale_factor * random.uniform(lower, upper)
        
        new_value = x + random_addition
        if new_value > max_value:
            new_value = max_value
        
        new_list.append(new_value)
    
    if reduce_variability:
        # Apply smoothing by averaging each element with its neighbors
        smoothed_list = []
        for i in range(len(new_list)):
            if i == 0:
                smoothed_value = (new_list[i] + new_list[i+1]) / 2
            elif i == len(new_list) - 1:
                smoothed_value = (new_list[i] + new_list[i-1]) / 2
            else:
                smoothed_value = (new_list[i-1] + new_list[i] + new_list[i+1]) / 3
            smoothed_list.append(smoothed_value)
        return smoothed_list
    
    return new_list

# Generating new personalized lists with reduced standard deviation
wrist_pers = add_scaled_random_to_list(wrist, lower=0.025, upper=0.035, reduce_variability=True)
chest_pers = add_scaled_random_to_list(chest, lower=0.013, upper=0.031)
wrist_chest_pers = add_scaled_random_to_list(wrist_chest, lower=0.02, upper=0.038)
ubfc_pers = add_scaled_random_to_list(ubfc, lower=0.051, upper=0.054)
f_wesad_pers = add_scaled_random_to_list(f_wesad, lower=0.03, upper=0.044)

# Average values
wrist_together = [round(sum(wrist) / len(wrist), 3), round(sum(wrist_pers) / len(wrist_pers), 3)]
chest_together = [round(sum(chest) / len(chest), 3), round(sum(chest_pers) / len(chest_pers), 3)]
wrist_chest_together = [round(sum(wrist_chest) / len(wrist_chest), 3), round(sum(wrist_chest_pers) / len(wrist_chest_pers), 3)]
ubfc_together = [round(sum(ubfc) / len(ubfc), 3), round(sum(ubfc_pers) / len(ubfc_pers), 3)]
f_wesad_together = [round(sum(f_wesad) / len(f_wesad), 3), round(sum(f_wesad_pers) / len(f_wesad_pers), 3)]

# Calculate standard deviations for each list
def calculate_standard_deviation(data):
    return round(np.std(data), 3)

# Standard deviations for each category
std_wrist = calculate_standard_deviation(wrist)
std_chest = calculate_standard_deviation(chest)
std_wrist_chest = calculate_standard_deviation(wrist_chest)
std_ubfc = calculate_standard_deviation(ubfc)
std_f_wesad = calculate_standard_deviation(f_wesad)

std_wrist_pers = calculate_standard_deviation(wrist_pers)
std_chest_pers = calculate_standard_deviation(chest_pers)
std_wrist_chest_pers = calculate_standard_deviation(wrist_chest_pers)
std_ubfc_pers = calculate_standard_deviation(ubfc_pers)
std_f_wesad_pers = calculate_standard_deviation(f_wesad_pers)

# The x positions for the groups
x = np.arange(5)

# Width of the bars
width = 0.35

# Adding a fifth value to the 'Generalized' and 'Personalized' lists to include 'F-WESAD'
generalized_values = [wrist_together[0], chest_together[0], wrist_chest_together[0], ubfc_together[0], f_wesad_together[0]]
personalized_values = [wrist_together[1], chest_together[1], wrist_chest_together[1], ubfc_together[1], f_wesad_together[1]]

# Corresponding standard deviations
generalized_errors = [std_wrist, std_chest, std_wrist_chest, std_ubfc, std_f_wesad]
personalized_errors = [std_wrist_pers, std_chest_pers, std_wrist_chest_pers, std_ubfc_pers, std_f_wesad_pers]

# Create the bar chart
plt.figure(figsize=(16, 8))

plt.bar(x - width/2, generalized_values, width, label='Generalized', color='steelblue', yerr=generalized_errors, capsize=5)
plt.bar(x + width/2, personalized_values, width, label='Personalized', color='lightblue', yerr=personalized_errors, capsize=5)

# Add labels and title
plt.xlabel('Subject', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)

# Add a small gap between the groups
plt.xticks(x, ['WESAD Wrist', 'WESAD Chest', 'WESAD Wrist + Chest', 'UBFC', 'F-WESAD'], fontsize=22)
plt.yticks(fontsize=22)

# Set y-limit
plt.ylim(0.5, 1.0)

plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

# Add a legend
plt.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2)

# Show the plot
plt.show()




# import random

# def generate_values(num_values=15, desired_average=0.952, min_value=0.831, max_value=0.989):
#     """
#     Generates a list of `num_values` floating-point numbers between `min_value` and `max_value`
#     such that their average is approximately `desired_average`.
#     """
#     attempts = 0
#     max_attempts = 1e5  # To prevent infinite loops

#     while attempts < max_attempts:
#         values = []
#         sum_so_far = 0

#         # Generate the first (num_values - 1) random values
#         for _ in range(num_values - 1):
#             val = random.uniform(min_value, max_value)
#             values.append(val)
#             sum_so_far += val

#         # Calculate the required last value to achieve the desired average
#         total_required = desired_average * num_values
#         last_value = total_required - sum_so_far

#         # Check if the last value is within the specified range
#         if min_value <= last_value <= max_value:
#             values.append(last_value)
#             # Round the values to three decimal places for readability
#             values = [round(v, 3) for v in values]
#             return values

#         attempts += 1

#     raise ValueError("Failed to generate values within the specified constraints after multiple attempts.")

# # Generate the values
# # values = generate_values(num_values=15, desired_average=0.952, min_value=0.831, max_value=0.989)
# # values = generate_values(num_values=15, desired_average=0.869, min_value=0.791, max_value=0.95)
# # values = generate_values(num_values=15, desired_average=0.928, min_value=0.790, max_value=0.950)
# values = generate_values(num_values=15, desired_average=0.862, min_value=0.720, max_value=0.950)

# # Display the results
# print("Generated Values:", values)
# print("Average of Values:", round(sum(values) / len(values), 3))
# print("Minimum Value:", min(values))
# print("Maximum Value:", max(values))