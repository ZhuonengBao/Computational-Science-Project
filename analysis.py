import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ast

# data analysis



# calculate average
means_dict = {}
with open("test_data.txt", 'r') as data_file:
    for line in data_file:
        line = line.strip()  # Remove \n and other whitespace
        data = line.split(":")
        print(data)
        key = ast.literal_eval(data[0].strip())  # Convert string key to tuple
        value = ast.literal_eval(data[1].strip())  # Convert string value to list

        # Calculate mean of the value list
        # Calculate mean of non-zero values only
        non_zero_values = [v for v in value if v != 0]  # Filter out zero values
        mean_value = sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0  # Avoid division by zero


        # Store in means_dict
        means_dict[key] = mean_value


# Plot
# Prepare data for the heatmap
# Get unique inter-connectivity and intra-connectivity values
inter_connectivities = sorted(set(key[0] for key in means_dict.keys()))  # x-axis
intra_connectivities = sorted(set(key[1] for key in means_dict.keys()))  # y-axis

# Create a grid for mean values
mean_grid = np.zeros((len(intra_connectivities), len(inter_connectivities)))

# Fill the grid with mean values from the dictionary
for i, inter in enumerate(inter_connectivities):
    for j, intra in enumerate(intra_connectivities):
        mean_grid[j, i] = means_dict.get((inter, intra), 0)  # Default to 0 if no value found

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(mean_grid, cmap='viridis', interpolation='nearest', aspect='auto')

# Set axis labels and title
plt.xlabel('Inter Connectivity')
plt.ylabel('Intra Connectivity')
plt.title('Heatmap of Mean Times')

# Set the ticks for x and y axes
plt.xticks(np.arange(len(inter_connectivities)), inter_connectivities)
plt.yticks(np.arange(len(intra_connectivities)), intra_connectivities)

# Invert the y-axis so that low values are at the bottom
plt.gca().invert_yaxis()

# Add colorbar to indicate time
plt.colorbar(label='Mean Time')

# Show the plot
plt.savefig("heatmap.png")
plt.show()
