"""
 * Multi-Layered Network
 *
 * Author's: Zhuoneng Bao, Mink van Maanen, Lynne Vogel and Babet Wijsman
 * Date: 20 January 2025
 * Description: This module implements a Multi-Layered Network Graph simulation,
                where each node represents has Hodgkin-Huxley neuron model. Each
                layer is a generated Erdős-Rényi directed graph with no
                bidirectional edges. To find the impact of intra- and
                inter-connection we measure the average of the time that the
                impulse is reached at the end. First, all neurons in the
                first layer are stimulated. The next neuron can only reach its
                action potential if it receives sufficient stimulus from its
                parent neurons. To evaluate the impact of intra- and
                inter-layer connections, the simulation measures the average
                time taken for the impulse to propagate and reach the
                last layer. This setup effectively models the dynamic behavior
                of interconnected neurons across multiple network layers.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from hh_model import HodgkinHuxleyNeuron


class LayeredNetworkGraph(object):
    def __init__(self, layers, time, step, inter_prob=0.0, verbose=False):
        """
        Generates a multi-layer network based on the Erdos-Renyi model for each
        layer and optionally plots the network in a 3D figure. In the
        3D visualization, different layers are separated along the z-axis.

        Description:
        - Intra-layer Connectivity: Within each layer, the connectivity between nodes
        is defined by the Erdos-Renyi graph specified in the `layers` parameter.
        - Inter-layer Connectivity: Nodes in consecutive layers are connected if they
        share the same node ID, forming vertical connections between layers.

        Parameters:
        - layers (list of tuples):
            A list of triples (n, p, s), where:
            - n (int): Number of nodes in the layer.
            - p (float): Probability of an edge existing between any pair of nodes.
            - s (float): Scaling factor for the node positions in the layer.
        - time (float): The duration of the simulation.
        - step (float): The time step used to approximate the Hodgkin-Huxley model.
        - verbose (bool): If True, the function plots the network in a 3D figure.
        """
        self.layers = layers
        self.total_layers = len(layers)
        self.time = time
        self.step = step
        self.verbose = verbose
        self.inter_prob = inter_prob
        self.layout = nx.spring_layout

        # create internal representation of nodes and edges
        self.get_graphs()
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()
        self.get_node_positions()

    def generate_erdos_renyi_digraph(self, n, p, s=''):
        """
        Generates a directed Erdos-Renyi graph with no bidirectional edges.

        Parameters:
        - n (int): Number of nodes.
        - p (float): The probability of creating a directed edge between two nodes.
        - s (str): A prefix for naming the nodes (optional).

        Returns:
        - G (object): A Networkx directed graph object.

        Examples:
        >>> Network = LayeredNetworkGraph([(5, 0, '')], 10, 1, inter_prob=0.0)
        >>> len(Network.nodes)
        5
        """
        G = nx.DiGraph()
        names = [s + str(i) for i in range(n)]
        for name in names:
            G.add_node(name)

        # Add edges based on Erdos-Renyi probability
        for i, u in enumerate(names):
            for j, v in enumerate(names):
                if i != j and random.random() < p:
                    G.add_edge(u, v)

        # Create neuron objects
        for node in G.nodes():
            G.nodes[node]['neuron'] = HodgkinHuxleyNeuron()

        return G

    def get_graphs(self):
        """
        Creates an Erdos-Renyi network for each layer.

        Examples:
        >>> Network = LayeredNetworkGraph([(10, 0.3, 's'), (10, 0.3, 't')], 10, 1, inter_prob=0.0)
        >>> len(Network.graphs) # Test for number of nodes
        2
        """
        self.graphs = []

        for n, p, s in self.layers[::-1]:
            self.graphs.append(self.generate_erdos_renyi_digraph(n, p, s))

    def get_nodes(self):
        """
        Construct an internal representation of nodes with the format (node ID, layer).

        Examples:
        >>> Network = LayeredNetworkGraph([(10, 0.3, '')], 10, 1, inter_prob=0.0)
        >>> ('3', 0) in Network.nodes
        True
        """
        self.nodes = []

        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])

        self.update = {}
        for g in self.graphs:
            self.update[g] = list(g.nodes())

        self.V_record = {}
        for nodes in self.update.values():
            for node in nodes:
                self.V_record[node] = []

    def get_edges_within_layers(self):
        """
        Remap edges in the individual layers to the internal representations of the node IDs.

        Examples:
        >>> Network = LayeredNetworkGraph([(10, 1.0, '')], 10, 1, inter_prob=0.0)
        >>> (('0', 0), ('1', 0)) in Network.edges_within_layers
        True
        """
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend(
                [((source, z), (target, z)) for source, target in g.edges()])

    def get_edges_between_layers(self):
        """
        Forms connections between nodes from different layers, thus connecting the layers

        Examples:
        >>> Network = LayeredNetworkGraph([(5, 0, 's'), (5, 0, 't')], 10, 1, inter_prob=1.0)
        >>> (('s0', 1), ('t0', 0)) in Network.edges_between_layers
        True
        """
        self.edges_between_layers = []
        for z1, h in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            g = self.graphs[z2]

            h_nodes = list(h.nodes())

            for node1 in g.nodes():
                for node2 in h_nodes:
                    if random.random() < self.inter_prob:
                        h.add_node(node1)
                        h.nodes[node1]['neuron'] = g.nodes[node1]['neuron']
                        h.add_edge(node1, node2)
                        self.edges_between_layers.append(
                            ((node1, z2), (node2, z1)))

    def get_node_positions(self, *args, **kwargs):
        """
        Generate and store 3D positions for nodes in a sequence of graphs.

        Parameters:
        - *args: Matplotlib layout additional arugments.
        - **kwargs: Matplotlib layout additional keyword arugments.

        >>> Network = LayeredNetworkGraph([(5, 0.3, 's')], 10, 1, inter_prob=0.5)
        >>> np.all([i in list(Network.node_positions.keys()) for i in Network.nodes])
        True
        """
        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update(
                {(node, z): (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
        """Draws the nodes in the 3D plane"""
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
        """Draws the edges in the 3D plane"""
        arrow_size = 0.1
        color = kwargs.get('color', 'blue')
        kwargs = {key: value for key, value in kwargs.items() if key !=
                  'color'}

        for source, target in edges:
            start = self.node_positions[source]
            end = self.node_positions[target]

            direction = np.array(end) - np.array(start)
            length = np.linalg.norm(direction)
            direction /= length

            self.ax.quiver(start[0], start[1], start[2],
                           direction[0], direction[1], direction[2],
                           length=length, color=color,
                           arrow_length_ratio=arrow_size, *args, **kwargs)

    def get_extent(self, pad=0.1):
        """Helper function used to extend the array node_positions."""
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, **kwargs):
        """Draws the plane in the 3D figure"""
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u, v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw_node_labels(self, *args, **kwargs):
        """Draws the node labels in the 3D figure"""
        for node, z in self.nodes:
            self.ax.text(
                *self.node_positions[(node, z)], node, *args, **kwargs)

    def draw(self):
        """Draws the 3D plot"""
        self.draw_edges(self.edges_within_layers, color='k',
                        alpha=0.3, linestyle='-', zorder=2)
        self.draw_edges(self.edges_between_layers, color='k',
                        alpha=0.3, linestyle='--', zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            self.draw_nodes(
                [node for node in self.nodes if node[1] == z], s=150, zorder=3)

        self.draw_node_labels(horizontalalignment='center',
                              verticalalignment='center',
                              zorder=100)

    def __plot(self):
        """
        Generates a 2D plot showing neuron voltages over time and a 3D
        visualization of the network structure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        # Plot the 3D graph
        axes[1].set_axis_off()
        self.ax = fig.add_subplot(122, projection='3d')
        self.ax.set_axis_off()
        self.draw()

        # Plot the 2D results
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Membrane Potential (mV)")
        axes[0].set_title("Neuron Activity in Network")

        time = np.arange(0, self.time, self.step)
        for node, voltages in self.V_record.items():
            axes[0].plot(time, voltages, label=f'Neuron {node}')

        plt.show()

    def run(self):
        """
        Simulates the activity of the neural network over a specified time period.

        Attributes:
        - self.time (float): Total simulation time in milliseconds.
        - self.step (float): Time step for simulation in milliseconds.
        - self.graphs (list): A sequence of graphs representing the layers of
          the network (from input to output).
        - self.update (dict): A mapping of layers to nodes to be updated.
        - self.verbose (bool): If True, plots the results of the simulation.

        Returns:
        - avg (float): The average time (in milliseconds) of action potential
                       peaks in the last layer. Returns 0 if no peaks
                       are detected.

        Examples:
        >>> Network = LayeredNetworkGraph([(5, 1.0, 'g'), (5, 1.0, 'h')], 15, 0.01, inter_prob=1.0)
        >>> time = Network.run()
        >>> 3.5 < time
        True
        >>> time < 3.7
        True
        """
        time = np.arange(0, self.time, self.step)

        # Stimulse of 15nA/cm^2 for 1.0 ms
        I_inp = np.zeros(len(time))
        I_inp[0:int(1 / self.step)] = 15.0
        I_inp[0] = 0

        # Post-Synaptic Constants
        tau = 30
        weight = 0.1

        peak_times = []
        first_layer = self.graphs[-1]
        last_layer = self.graphs[0]
        for i in range(len(time)):

            for layer in self.graphs[::-1]:
                nodes = self.update[layer]
                Network = layer.nodes()

                for node in nodes:
                    I_temp = 0.0
                    neuron = Network[node]['neuron']

                    if layer == first_layer:
                        neuron.step(self.step, I_inp[i])
                    else:
                        for pred in list(layer.predecessors(node)):
                            parent = Network[pred]['neuron']

                            diff = np.clip(neuron.V - parent.V, -50, 50)
                            result = np.exp(-diff / tau)
                            I_temp += weight * result

                        neuron.step(self.step, I_temp)

                    self.V_record[node].append(neuron.V)

                    # Check for action-potential
                    if layer == last_layer and i > 3:
                        prev_V, current_V, next_V = self.V_record[node][-3:]
                        if prev_V < current_V and current_V > next_V and current_V > 30.0:
                            peak_times.append(time[i - 1])

        if self.verbose:
            self.__plot()

        avg = sum(peak_times) / len(peak_times) if peak_times else 0

        return avg


if __name__ == '__main__':
    # define graphs
    n = 5
    p = 0.2
    prob_inter = 0.5

    T = 25
    dt = 0.01
    obj = LayeredNetworkGraph(
        [(n, 0, 'g'), (n, p, 'h')], T, dt, inter_prob=prob_inter, verbose=True)
    a = obj.run()
    print(a)
