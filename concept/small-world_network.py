import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# Parameters for the neurons and rings
n_neurons = 30  
n_rings = 3 
neurons_per_ring = n_neurons // n_rings
k_neighbors = 2

ring_graph = nx.Graph()
ring_graph.add_nodes_from(range(n_neurons))
radii = np.linspace(1, 3, n_rings)  # From smaller to larger radii

# Assign positions for each neuron
positions = {}
for ring_id in range(n_rings):
    start_idx = ring_id * neurons_per_ring
    end_idx = (ring_id + 1) * neurons_per_ring
    neurons_in_ring = range(start_idx, end_idx)
    
    # Calculate positions on the circle for each neuron in this ring
    angle_step = 2 * np.pi / neurons_per_ring
    for i, neuron in enumerate(neurons_in_ring):
        angle = i * angle_step
        x = radii[ring_id] * np.cos(angle)
        y = radii[ring_id] * np.sin(angle)
        positions[neuron] = (x, y)
        
        right = (i + 1) % neurons_per_ring
        left = (i - 1) % neurons_per_ring
        ring_graph.add_edge(neuron, neurons_in_ring[right])
        ring_graph.add_edge(neuron, neurons_in_ring[left])

# Inter-ring connections (connect neurons between rings)
for ring_id in range(1, n_rings):
    start_idx_outer = ring_id * neurons_per_ring
    end_idx_outer = (ring_id + 1) * neurons_per_ring
    neurons_in_outer_ring = range(start_idx_outer, end_idx_outer)
    start_idx_inner = (ring_id - 1) * neurons_per_ring
    end_idx_inner = ring_id * neurons_per_ring
    neurons_in_inner_ring = range(start_idx_inner, end_idx_inner)
    
    # For each neuron in the outer ring, connect it to one neuron in the inner ring
    for neuron_outer in neurons_in_outer_ring:
        # Calculate distances to neurons in the inner ring
        closest_neurons = []
        for neuron_inner in neurons_in_inner_ring:
            x1, y1 = positions[neuron_outer]
            x2, y2 = positions[neuron_inner]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            closest_neurons.append((distance, neuron_inner))
        
        # Sort by distance and pick the closest one
        closest_neurons.sort()  # Sort by distance
        neuron_inner = closest_neurons[0][1]
        ring_graph.add_edge(neuron_outer, neuron_inner)
    
    # For each neuron in the inner ring, connect it to one neuron in the outer ring
    for neuron_inner in neurons_in_inner_ring:
        # Calculate distances to neurons in the outer ring
        closest_neurons = []
        for neuron_outer in neurons_in_outer_ring:
            x1, y1 = positions[neuron_inner]
            x2, y2 = positions[neuron_outer]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            closest_neurons.append((distance, neuron_outer))
        
        # Sort by distance and pick the closest one
        closest_neurons.sort()  # Sort by distance
        neuron_outer = closest_neurons[0][1]
        ring_graph.add_edge(neuron_inner, neuron_outer)

# Draw the neurons on concentric rings
plt.figure(figsize=(8, 8))
nx.draw(ring_graph, pos=positions, with_labels=True, node_color='gray', node_size=700, font_size=10, edge_color='lightgray')
plt.title("Neurons with K Nearest Neighbors on 3 Concentric Rings", fontsize=15)
plt.show()

