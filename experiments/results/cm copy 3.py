import matplotlib.pyplot as plt
import numpy as np

# Sample data
list1 = [10, 20, 30, 40, 50]
list2 = [15, 25, 35, 45, 55]

# Number of elements
n = len(list1)

# The x positions for the groups
x = np.arange(n)

# Width of the bars
width = 0.4

# Create the bar chart
plt.bar(x - width/2, list1, width, label='List 1')
plt.bar(x + width/2, list2, width, label='List 2')

# Add labels and title
plt.xlabel('Element')
plt.ylabel('Value')
plt.title('Grouped Bar Chart with Touching Bars')

# Add a small gap between the groups
plt.xticks(x, ['A', 'B', 'C', 'D', 'E'])

# Add a legend
plt.legend()

# Show the plot
plt.show()
