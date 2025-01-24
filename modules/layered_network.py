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
    def __init__(self, layers, time, step, inter_prob=0.0, layout=nx.spring_layout, verbose=False):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """
        # book-keeping
        self.layers = layers
        self.total_layers = len(layers)
        self.time = time
        self.step = step
        self.verbose = verbose
        self.inter_prob = inter_prob
        self.layout = layout

        # create internal representation of nodes and edges
        self.get_graphs()
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()
        self.get_node_positions()

    def generate_erdos_renyi_digraph(self, n, p, s=''):
        G = nx.DiGraph()
        names = [s + str(i) for i in range(n)]
        for name in names:
            G.add_node(name)

        # Add edges based on Erdos-Renyi probability
        for i, u in enumerate(names):
            for j, v in enumerate(names):
                if i != j and random.random() < p:
                    G.add_edge(u, v)

        # Remove bidirectional edges
        for u, v in list(G.edges()):
            if G.has_edge(v, u):
                G.remove_edge(v, u)

        # Create neuron objects
        for node in G.nodes():
            G.nodes[node]['neuron'] = HodgkinHuxleyNeuron()

        return G

    def get_graphs(self):
        """Creates the network for each layer."""
        self.graphs = []

        for n, p, s in self.layers[::-1]:
            self.graphs.append(self.generate_erdos_renyi_digraph(n, p, s))

    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
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
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend(
                [((source, z), (target, z)) for source, target in g.edges()])

    def get_edges_between_layers(self):
        """Forms connections between nodes from different layers, thus connecting the layers"""
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
        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update(
                {(node, z): (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
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
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u, v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw_node_labels(self, *args, **kwargs):
        for node, z in self.nodes:
            self.ax.text(
                *self.node_positions[(node, z)], node, *args, **kwargs)

    def draw(self):
        self.draw_edges(self.edges_within_layers,  color='k',
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
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

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
        # Parameters
        time = np.arange(0, self.time, self.step)

        # Stimulse of 15nA/cm^2 for 1.0 ms
        I_inp = np.zeros(len(time))
        I_inp[0:int(1/self.step)] = 15.0
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
                            I_temp += weight * \
                                np.exp(-(neuron.V - parent.V) / tau)

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
    n = 10
    p = 0.5
    prob_inter = 0.2

    T = 25
    dt = 0.01
    obj = LayeredNetworkGraph(
        [(n, 0, 'g'), (n, p, 'h')], T, dt, inter_prob=prob_inter, verbose=True)
    a = obj.run()
    print(a)
