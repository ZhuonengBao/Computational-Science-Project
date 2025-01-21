import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random
matplotlib.use('TkAgg')
from hh_model import HodgkinHuxleyNeuron

"""Currently this file was used to test what synaptic weights triggered action potentials, with input from
multiple neurons."""


n_neurons = 300
k = 3
p = 0.2
branch_factor = 3
n_neighbours_to_stim = 5
synaptic_strength = 0.000000275 / (n_neighbours_to_stim)

network = nx.DiGraph()

for i in range(n_neurons):
    network.add_node(i, neuron=HodgkinHuxleyNeuron())

# for i in range(n_neurons - 1):
#     network.add_edge(i + 1, i)

for i in range(n_neurons - 1):
    network.add_edge(n_neurons - 1, i)

# network.add_edge(n_neurons - 1, 0)

# Add directed edges to create a branching structure
# for i in range(n_neurons):
#     # Neuron i will send connections to random other neurons
#     # Control branching with a maximum number of outgoing connections (branch_factor)
#     num_connections = random.randint(1, branch_factor)  # Random number of outgoing edges
#     targets = random.sample(range(n_neurons), num_connections)  # Randomly pick target neurons
#     for target in targets:
#         if target != i:  # Prevent self-loops
#             network.add_edge(i, target)


# for u, v in network.edges():
#     network[u][v]['weight'] = np.random.uniform(0.000001, 0.000001)

T = 200.0
dt = 0.01
time = np.arange(0, T, dt)

for i in range(10):
    network.nodes[i]['neuron'].I_ext = 7.5

    # network.nodes[0]['neuron'].I_ext = 7.5
    # network.nodes[1]['neuron'].I_ext = 7.5
    # network.nodes[2]['neuron'].I_ext = 7.5
    # network.nodes[3]['neuron'].I_ext = 7.5
    # network.nodes[4]['neuron'].I_ext = 7.5

V_record = {node: [] for node in network.nodes()}


for t in time:
    for node in network.nodes():
        neuron = network.nodes[node]['neuron']
        I_syn = 0.0

        for neighbor in network.neighbors(node):
            neighbor_neuron = network.nodes[neighbor]['neuron']
            weight = network[node][neighbor].get('weight', synaptic_strength)
            tau = 5.0  # Synaptic decay time constant
            I_syn += weight * np.exp(-(neuron.V - neighbor_neuron.V) / tau)

        # Update neuron with total synaptic current
        neuron.step(dt, I_syn)
        V_record[node].append(neuron.V)
        # if t > 30.0:
        #     network.nodes[0]['neuron'].I_ext = 0.0


plt.figure(figsize=(10, 8))
for node, V in V_record.items():
    plt.plot(time, V, label=f'Neuron: {node + 1}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(network)
nx.draw(
    network, 
    pos,
    with_labels=True, 
    node_color='skyblue', 
    node_size=500, 
    font_size=10, 
    font_weight='bold',
    edge_color='gray'
)
plt.title("Neural Network Visualization")
plt.show()