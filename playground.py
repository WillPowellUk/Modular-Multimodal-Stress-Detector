import matplotlib.pyplot as plt
import numpy as np

# Fake data
categories = ['A', 'B', 'C', 'D', 'E']
values1 = np.random.randint(10, 50, 5)
values2 = np.random.randint(10, 50, 5)

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Example using 'steelblue' and 'lightblue'
bar_width = 0.35
x = np.arange(len(categories))
ax.bar(x - bar_width/2, values1, bar_width, color='steelblue', label='Group 1')
ax.bar(x + bar_width/2, values2, bar_width, color='lightblue', label='Group 2')

# Customize the chart
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Sample Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()