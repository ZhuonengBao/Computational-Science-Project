"""
 * visualize and analyse data
 *
 * Author's: Zhuoneng Bao, Mink van Maanen, Lynne Vogel and Babet Wijsman
 * Date: 11 January 2025
 * Description: This file was used to explore adding neurons to a network and letting them interact.

 """
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from hh_model import HodgkinHuxleyNeuron

# set parameters
n = 5
start_I_ext = 10.0

# Create neurons
neurons = [HodgkinHuxleyNeuron() for _ in range(n)]

# Set external input for all neurons
for neuron in neurons:
    neuron.I_ext = start_I_ext  

# make network --> misschien hier ander soort netwerk van maken
G = nx.DiGraph()  # Gericht netwerk
G.add_nodes_from(range(n))

# add edges with random weight --> nu zijn alle neuronen verbonden, kunnen we eventueel nog aanpassen
for i in range(n):
    for j in range(n):
        if i != j:  
            weight = np.random.uniform(-1, 1)
            G.add_edge(i, j, weight=weight)

# set parameters for running the simulation
T = 50.0
dt = 0.01
time = np.arange(0, T, dt)

v_record = np.zeros((n, len(time)))
colors = [[] for _ in range(n)]

for i_time, t in enumerate(time):
    # calculate input
    for i_neuron in range(n):
        synaptic_input = 0
        for i_neighboor in G.predecessors(i):
            weight = G[i_neuron][i_neighboor]['weight']
            synaptic_input += weight * neurons[i_neighboor].V
        neurons[i_neuron].I_ext = start_I_ext + synaptic_input  
    
    # Update all neurons
    for i, neuron in enumerate(neurons):
        neuron.step(dt)
        v_record[i, i_time] = neuron.V
        if neuron.V > 0: # nog even bespreken wat we als vuren zien
            color ='red'
        else:
            color = 'green'
        
        colors[i].append(color)

# Plot results in graph
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(time, v_record[i, :], label=f"Neuron {i+1}")
plt.xlabel("time)")
plt.ylabel("Membrane potential (mV)")
plt.title("Dynamics of a network of neurons")
plt.legend()
plt.show()

# visualize network
for t in range(T):
    node_colors = colors[t]

    nx.draw(G, node_color = node_colors)
    plt.draw()
    plt.pause(0.05)
    plt.clf()

plt.show()
