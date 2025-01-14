import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from hh_model import HodgkinHuxleyNeuron

n_neurons = 20
k = 4
p = 0.2

network = nx.watts_strogatz_graph(n_neurons, k, p)

for node in network.nodes():
    network.nodes[node]['neuron'] = HodgkinHuxleyNeuron()

# TODO: add (random) weights

T = 50
dt = 0.01
time = np.arange(0, T, dt)

network.nodes[0]['neuron'].I_ext = 10.0

V_record = {node: [] for node in network.nodes()}

for t in time:
    for node in network.nodes():
        neuron = network.nodes[node]['neuron']
        neuron.step(dt)
        V_record[node].append(neuron.V)

plt.figure(figsize=(10, 8))
for node, V in V_record.items():
    plt.plot(time, V, label=f'Neuron: {node + 1}')
plt.legend()
plt.show()

