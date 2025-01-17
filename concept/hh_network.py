from hh_model import HodgkinHuxleyNeuron
from layered_network import LayeredNetworkGraph

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

class HodgkinHuxleyNetwork():
    def __init__(self):
        pass

def create_erdos_renyi_network():
    num_neurons = 20
    g = nx.erdos_renyi_graph(num_neurons, p=0.3)
    h = nx.erdos_renyi_graph(num_neurons, p=0.3)
    i = nx.erdos_renyi_graph(num_neurons, p=0.3)

    LayeredNetworkGraph([g, h, i])