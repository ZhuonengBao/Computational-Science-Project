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


class LayeredNetworkGraph(object):

    def __init__(self, graphs):
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        """REMOVE THIS ALL LATER:"""
        self.node_labels = {nn : str(nn) for nn in range(4*n)}
        self.layout = nx.spring_layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # compute layout and plot
        self.get_node_positions()
        self.draw()

    def get_nodes(self):
        self.nodes = []
        for i, g in enumerate(self.graphs):
            self.nodes.extend([(node, i) for node in g.nodes()])
    
    def get_edges_within_layers(self):
        self.edges_within_layers = []
        for i, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((post_syn, i), (pre_syn, i)) for post_syn, pre_syn in g.edges()])
    
    def get_edges_between_layers(self, prob=0.1):
        #TODO: implement this function with the desired amount of connections and with multiple connections per node
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]

            for node1 in g.nodes():
            # Iterate over all nodes in layer z2 (next layer)
                for node2 in h.nodes():
                # Randomly connect node1 and node2 with probability prob
                    if random.random() < prob:
                        self.edges_between_layers.append(((node1, z1), (node2, z2)))

            # shared_nodes = set(g.nodes()) & set(h.nodes())
            # self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])

    def get_node_positions(self, *args, **kwargs):
        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
            x, y, z = zip(*[self.node_positions[node] for node in nodes])
            self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

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
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)

    def draw(self):
        self.draw_edges(self.edges_within_layers,  color='k', alpha=0.05, linestyle='-', zorder=2)
        self.draw_edges(self.edges_between_layers, color='k', alpha=0.05, linestyle='--', zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            self.draw_nodes([node for node in self.nodes if node[1]==z], s=30, zorder=3)

            # if self.node_labels:
            #     self.draw_node_labels(self.node_labels,
            #                         horizontalalignment='center',
            #                         verticalalignment='center',
            #                         zorder=100)

if __name__ == '__main__':
    # define graphs
    n = 30
    g = nx.erdos_renyi_graph(n, p=0.3)
    h = nx.erdos_renyi_graph(n, p=0.3)
    i = nx.erdos_renyi_graph(n, p=0.3)

    node_labels = {nn : str(nn) for nn in range(4*n)}

    # initialise figure and plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    LayeredNetworkGraph([g, h, i])
    ax.set_axis_off()
    plt.show()

