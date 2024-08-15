import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate more intermediate samples for epochs
epochs = np.linspace(1, 5, 25)  # 25 points between 1 and 5 epochs

# Generate fake data with slight improvements after the first epoch
train_accuracy = np.array([0.90 + 0.01 * np.exp(-0.5 * (epoch - 1)) for epoch in epochs])
val_accuracy = np.array([0.85 + 0.005 * np.exp(-0.5 * (epoch - 1)) for epoch in epochs])

# Add some fake standard deviations
train_std = np.array([0.02 - 0.001 * epoch for epoch in epochs])
val_std = np.array([0.03 - 0.0015 * epoch for epoch in epochs])

# Create a DataFrame for seaborn
data = pd.DataFrame({
    'Epoch': np.concatenate([epochs, epochs]),
    'Accuracy': np.concatenate([train_accuracy, val_accuracy]),
    'Standard Deviation': np.concatenate([train_std, val_std]),
    'Type': ['Training']*len(epochs) + ['Validation']*len(epochs)
})

# Set the style
sns.set(style="whitegrid")

# Plot the learning curve
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Accuracy', data=data, hue='Type', marker='o', linewidth=2, ci=None)
plt.fill_between(epochs, train_accuracy - train_std, train_accuracy + train_std, alpha=0.2, color=sns.color_palette()[0])
plt.fill_between(epochs, val_accuracy - val_std, val_accuracy + val_std, alpha=0.2, color=sns.color_palette()[1])

# Set limits and labels
plt.xlim(0, 5)
plt.ylim(0.7, 1.0)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve with Intermediate Samples')
plt.legend(title='Type')
plt.show()
