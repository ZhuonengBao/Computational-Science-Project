#!/usr/bin/env python
"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from hh_model import HodgkinHuxleyNeuron



class LayeredNetworkGraph(object):

    def __init__(self, graphs, prob):
        self.graphs = graphs
        self.total_layers = len(graphs)
        self.activate = 3
        self.prob = prob

        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()
        self.combined_network = self.create_combined_network()
        self.node_positions = self.get_node_positions()
      

    def get_nodes(self):
        self.nodes = []
        for i, g in enumerate(self.graphs):
            for node in g.nodes:
                g.nodes[node]["neuron"] = HodgkinHuxleyNeuron()
            self.nodes.extend([(node, i) for node in g.nodes()])

        for i in range(self.activate):
            node, layer = self.nodes[i]  # Unpack the node and layer index
            neuron = self.graphs[layer].nodes[node]["neuron"]  # Access the neuron object
            neuron.I_ext = 7.5  # Apply external current
    
    def get_edges_within_layers(self):
        self.edges_within_layers = []
        for i, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((post_syn, i), (pre_syn, i)) for post_syn, pre_syn in g.edges()])
    
    def get_edges_between_layers(self):
        """Forms connections between nodes from different layers, thus connecting the layers"""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]

            for node1 in g.nodes():
            # Iterate over all nodes in layer z2 (next layer)
                for node2 in h.nodes():
                # Randomly connect node1 and node2 with probability prob
                    if random.random() < self.prob:
                        self.edges_between_layers.append(((node1, z1), (node2, z2)))


    def create_combined_network(self):
        """Combine all layers into a single network with inter-layer connections."""
        combined = nx.Graph()

        # Add nodes and edges from each layer
        for z, g in enumerate(self.graphs):
            for node in g.nodes:
                combined.add_node((node, z), **g.nodes[node])  # Add layer info to nodes
            for n1, n2 in g.edges:
                combined.add_edge((n1, z), (n2, z), **g.edges[n1, n2])

        # Add inter-layer edges
        for (n1, z1), (n2, z2) in self.edges_between_layers:
            combined.add_edge((n1, z1), (n2, z2))

        return combined

    def get_node_positions(self, layout=nx.spring_layout, *args, **kwargs):
        """Compute 3D positions of nodes in the combined network."""
        # Compute 2D layout for the combined graph
        pos_2d = layout(self.combined_network, *args, **kwargs)

        # Add z-coordinate based on layer index
        node_positions = {
            node: (*pos_2d[node], node[1]) for node in self.combined_network.nodes
        }
        return node_positions

    def run_hh_network(self):
        # Parameters
        T = 40
        dt = 0.01
        time = np.arange(0, T, dt)
        # n_neighbours_to_stim = 10 # Amount of neighbouring action potentials needed to stimulate a neuron
        # synaptic_strength = 0.000000275 / n_neighbours_to_stim
        synaptic_strength = 10
        # Record for membrane potentials
        V_record = {node: [] for node in self.combined_network.nodes()}

        # Simulation loop
        for t in time:
            # Loop through all neurons in the network
            for node in self.combined_network.nodes():
                neuron = self.combined_network.nodes[node]['neuron']
                I_syn = 0.0

                # Compute synaptic input from neighbors
                for neighbor in self.combined_network.neighbors(node):
                    neighbor_neuron = self.combined_network.nodes[neighbor]['neuron']
                    tau = 5.0 # Synaptic decay time constant

                    if neighbor_neuron.last_spike_time is not None:
                        t_fire = neighbor_neuron.last_spike_time
                        if t >= t_fire:  # Apply synaptic current only after firing
                            I_syn += synaptic_strength * np.exp(-(t - t_fire) / tau)

                # Update neuron with external current + synaptic current
                neuron.step(dt, I_syn)
                V_record[node].append(neuron.V)

        # Plotting results
        # plt.figure(figsize=(12, 8))
        # for node, voltages in V_record.items():
        #     plt.plot(time, voltages, label=f'Neuron {node}')
        # plt.legend()
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Membrane Potential (mV)")
        # plt.title("Neuron Activity in Network")
        # plt.show()

        return V_record, time

if __name__ == '__main__':
    # define graphs
    n = 50
    g = nx.erdos_renyi_graph(n, p=0.3)
    h = nx.erdos_renyi_graph(n, p=0.3)
    i = nx.erdos_renyi_graph(n, p=0.3)

    # initialise figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    network = LayeredNetworkGraph([g, h, i])
    ax.set_axis_off()
    # plt.show()

    network.run_hh_network()


