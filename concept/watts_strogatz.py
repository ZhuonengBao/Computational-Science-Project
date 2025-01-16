import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from hh_model import HodgkinHuxleyNeuron

n_neurons = 20
k = 3
p = 0.2

network = nx.watts_strogatz_graph(n_neurons, k, p)

for node in network.nodes():
    network.nodes[node]['neuron'] = HodgkinHuxleyNeuron()

# TODO: add (random) weights
for u, v in network.edges():
    network[u][v]['weight'] = np.random.uniform(0.05, 0.2)

T = 50.0
dt = 0.01
time = np.arange(0, T, dt)

network.nodes[0]['neuron'].I_ext = 10.0

V_record = {node: [] for node in network.nodes()}


for t in time:
    for node in network.nodes():
        neuron = network.nodes[node]['neuron']
        I_syn = 0.0

        # Calculate total synaptic current from neighbors
        for neighbor in network.neighbors(node):
            neighbor_neuron = network.nodes[neighbor]['neuron']
            weight = network[node][neighbor].get('weight', 0.1)
            tau = 5.0  # Synaptic decay time constant
            I_syn += weight * np.exp(-(neuron.V - neighbor_neuron.V) / tau)

            weight = network[node][neighbor]['weight']
            V_pre = network.nodes[neighbor]['neuron'].V
            I_syn += weight * (V_pre - neuron.V)  # Simple synaptic model

        # Update neuron with total synaptic current
        neuron.step(dt, I_syn)
        V_record[node].append(neuron.V)
        if t > 30.0:
            network.nodes[0]['neuron'].I_ext = 0.0


# plt.figure(figsize=(10, 8))
# for node, V in V_record.items():
#     plt.plot(time, V, label=f'Neuron: {node + 1}')
# # plt.legend()
# plt.show()

plt.figure(figsize=(10, 10))
pos = nx.circular_layout(network)
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
