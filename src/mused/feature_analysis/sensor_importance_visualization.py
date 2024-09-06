# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.patches as mpatches

# # Data extracted from the table
# binary_data = {
#     'Device': ['Empatica', 'Polar', 'Myndsens', 'Empatica', 'Quattrocento', 'Polar', 'Quattrocento', 'Empatica', 'Empatica', 'Polar'],
#     'Sensor': ['EDA', 'ECG', 'fNIRS', 'BVP', 'Upper Trapezius EMG', 'ACC', 'Mastoid EMG', 'ACC', 'TEMP', 'IBI'],
#     'Gini Importance': [0.262039, 0.244348, 0.1476, 0.12178, 0.056811, 0.050668, 0.049338, 0.047793, 0.018673, 0.00095]
# }

# four_level_data = {
#     'Device': ['Empatica', 'Myndsens', 'Polar', 'Empatica', 'Quattrocento', 'Empatica', 'Quattrocento', 'Polar', 'Empatica', 'Polar'],
#     'Sensor': ['EDA', 'fNIRS', 'ECG', 'BVP', 'Mastoid EMG', 'ACC', 'Upper Trapezius EMG', 'ACC', 'TEMP', 'IBI'],
#     'Gini Importance': [0.281748, 0.179387, 0.140859, 0.1391, 0.058775, 0.058618, 0.054232, 0.054228, 0.031596, 0.001457]
# }

# # Sort the data by 'Device' to group sensors by device
# binary_data_sorted = sorted(zip(binary_data['Device'], binary_data['Sensor'], binary_data['Gini Importance']), key=lambda x: x[0])
# four_level_data_sorted = sorted(zip(four_level_data['Device'], four_level_data['Sensor'], four_level_data['Gini Importance']), key=lambda x: x[0])

# # Unzip sorted data
# binary_devices, binary_sensors, binary_importance = zip(*binary_data_sorted)
# four_level_devices, four_level_sensors, four_level_importance = zip(*four_level_data_sorted)

# # Define distinct color palettes for each device
# device_palettes = {
#     'Empatica': sns.color_palette("mako", n_colors=4),
#     'Polar': sns.color_palette("flare", n_colors=4),
#     'Myndsens': sns.color_palette("rocket", n_colors=2),
#     'Quattrocento': sns.color_palette("crest", n_colors=2)
# }

# # Generate colors for each sensor based on its device
# def get_sensor_colors(devices):
#     colors = []
#     device_counter = {device: 0 for device in device_palettes.keys()}
#     for device in devices:
#         color = device_palettes[device][device_counter[device]]
#         colors.append(color)
#         device_counter[device] += 1
#     return colors

# # Get sensor colors for binary and four-level classifications
# binary_colors = get_sensor_colors(binary_devices)
# four_level_colors = get_sensor_colors(four_level_devices)

# # Plotting Binary Classification Pie Chart
# plt.figure(figsize=(7, 7))
# plt.pie(
#     binary_importance,
#     labels=binary_sensors,
#     colors=binary_colors,
#     startangle=140
# )
# # plt.title('Binary Classification - Gini Importance')

# # Create a legend for devices
# legend_patches = [mpatches.Patch(color=device_palettes[device][0], label=device) for device in device_palettes.keys()]
# plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Devices')

# plt.tight_layout()
# plt.show()

# # Plotting Four-Level Classification Pie Chart
# plt.figure(figsize=(7, 7))
# plt.pie(
#     four_level_importance,
#     labels=four_level_sensors,
#     colors=four_level_colors,
#     startangle=140
# )
# # plt.title('Four-Level Classification - Gini Importance')

# # Create a legend for devices
# plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Devices')

# plt.tight_layout()
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# Configure Matplotlib to use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use serif font in conjunction with LaTeX
    # Set the default font to be used in LaTeX as a single string
    "text.latex.preamble": r"\usepackage{times}",
    'font.size': 16
    })

# Data extracted from the table
binary_data = {
    'Device': ['Empatica', 'Polar', 'Myndsens', 'Empatica', 'Quattrocento', 'Polar', 'Quattrocento', 'Empatica', 'Empatica', 'Polar'],
    'Sensor': ['EDA', 'ECG', 'fNIRS', 'BVP', 'Upper Trap EMG', 'Chest ACC', 'Mastoid EMG', 'Wrist ACC', 'TEMP', 'IBI'],
    'Gini Importance': [0.262039, 0.244348, 0.1476, 0.12178, 0.056811, 0.050668, 0.049338, 0.047793, 0.018673, 0.00095]
}

four_level_data = {
    'Device': ['Empatica', 'Myndsens', 'Polar', 'Empatica', 'Quattrocento', 'Empatica', 'Quattrocento', 'Polar', 'Empatica', 'Polar'],
    'Sensor': ['EDA', 'fNIRS', 'ECG', 'BVP', 'Mastoid EMG', 'Wrist ACC', 'Upper Trap EMG', 'Chest ACC', 'TEMP', 'IBI'],
    'Gini Importance': [0.281748, 0.179387, 0.140859, 0.1391, 0.058775, 0.058618, 0.054232, 0.054228, 0.031596, 0.001457]
}

# Convert data to DataFrame for easier manipulation
binary_df = pd.DataFrame(binary_data)
four_level_df = pd.DataFrame(four_level_data)

# Sort the data by 'Gini Importance' to order sensors from largest to smallest
binary_df_sorted = binary_df.sort_values(by='Gini Importance', ascending=False)
four_level_df_sorted = four_level_df.sort_values(by='Gini Importance', ascending=False)

# Define color palette for devices
palette = sns.color_palette("Set2", 4)
device_colors = {
    'Empatica': palette[0],
    'Polar': palette[1],
    'Myndsens': palette[2],
    'Quattrocento': palette[3]
}

# Function to map device to color
def map_device_to_color(device):
    return device_colors[device]

# Apply color mapping
binary_df_sorted['Color'] = binary_df_sorted['Device'].apply(map_device_to_color)
four_level_df_sorted['Color'] = four_level_df_sorted['Device'].apply(map_device_to_color)

# Plotting Binary Classification Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(binary_df_sorted['Sensor'], binary_df_sorted['Gini Importance'], color=binary_df_sorted['Color'])
# plt.xlabel('Sensor', fontsize=22)
plt.ylabel('Gini Importance', fontsize=22)
plt.ylim([0, 0.3])

# Annotate bars with the Gini Importance values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=14, color='black')


# Create a legend for devices
legend_patches = [mpatches.Patch(color=color, label=device) for device, color in device_colors.items()]
plt.legend(handles=legend_patches, fontsize=22)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Plotting 4 level Classification Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(four_level_df_sorted['Sensor'], four_level_df_sorted['Gini Importance'], color=four_level_df_sorted['Color'])
# plt.xlabel('Sensor', fontsize=22)
plt.ylabel('Gini Importance', fontsize=22)
plt.ylim([0, 0.32])

# Annotate bars with the Gini Importance values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=14, color='black')


# Create a legend for devices
legend_patches = [mpatches.Patch(color=color, label=device) for device, color in device_colors.items()]
plt.legend(handles=legend_patches, fontsize=22)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
