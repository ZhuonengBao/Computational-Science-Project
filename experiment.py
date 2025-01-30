# from concept.visualize_data import time_between_spiking, generate_erdos_renyi_digraph
from modules.layered_network import LayeredNetworkGraph
import networkx as nx
import numpy as np


def create_network_setup(intra_connectivity, layers, num_neurons):
    graphs = [(num_neurons, 0, '')]
    for _ in range(layers - 1):
        graph_tuple = (num_neurons, intra_connectivity, '')
        graphs.append(graph_tuple)
    return graphs

def run_experiment_once(inter_connectivity, intra_connectivity, layers, num_neurons, T, dt, verbose=False):
    # network_setup = create_network_setup(intra_connectivity, layers, num_neurons)

    # (n, 0, 'g'), (n, p, 'h')], T, dt, inter_prob=prob_inter, verbose=True
    network = LayeredNetworkGraph([(num_neurons, 0, 'g'), (num_neurons, intra_connectivity, 'h'), (num_neurons, intra_connectivity, 'i')], T, dt, inter_connectivity, verbose)

    avg_time = network.run()

    # time_between_spikes = time_between_spiking(network, num_neurons, (0, 0), ((num_neurons - 1), (len(network_setup) - 1)))
    return avg_time

from itertools import product

def run_experiment(inter_connectivities, intra_connectivities, layers, num_neurons, num_iterations, T, dt, verbose=False):
    timings = {}

    # Generate all combinations of inter_connectivities and intra_connectivities
    for inter, intra in product(inter_connectivities, intra_connectivities):
        timings[(float(inter), float(intra))] = []  # Initialize an empty list for each combination

        # Run the experiment for num_iterations
        for _ in range(num_iterations):
            timing = run_experiment_once(inter, intra, layers, num_neurons, T, dt, verbose=verbose)
            timings[(inter, intra)].append(float(timing))  # Append timing to the list

    print(timings)
    return timings

def data_to_file(filename, data):
    with open(filename, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")


if __name__=="__main__":
    inter_connectivities = np.arange(0.001, 0.31, 0.03)
    intra_connectivities = np.arange(0.001, 0.31, 0.03)
    layers = 3
    num_neurons = 100
    num_iterations = 2

    # n = 5
    # p = 0.2
    # prob_inter = 0.5

    T = 25
    dt = 0.01
    # obj = LayeredNetworkGraph(
    #     [(n, 0, 'g'), (n, p, 'h')], T, dt, inter_prob=prob_inter, verbose=True)
    # a = obj.run()

    # inter_connectivities = [0.01, 0.02]
    # intra_connectivities = [0.03, 0.04]

    data = run_experiment(inter_connectivities, intra_connectivities, layers, num_neurons, num_iterations, T, dt, verbose=False)
    
    filename = "test_data.txt"
    data_to_file(filename, data)



    # with Pool(processes=os.cpu_count()) as pool:
    #     peak_times = pool.map(layered_sim, inter_ps)