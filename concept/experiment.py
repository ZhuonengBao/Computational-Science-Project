from visualize_data import time_between_spiking, generate_erdos_renyi_digraph
from layered_network import LayeredNetworkGraph
import networkx as nx


def create_network_setup(intra_connectivity, layers, num_neurons):
    graphs = []
    for _ in range(layers):
        graph_tuple = (num_neurons, intra_connectivity, '')
        graphs.append(graph_tuple)
    return graphs

def run_experiment_once(inter_connectivity, intra_connectivity, layers, num_neurons, T, dt, verbose=False):
    network_setup = create_network_setup(intra_connectivity, layers, num_neurons)

    # (n, 0, 'g'), (n, p, 'h')], T, dt, inter_prob=prob_inter, verbose=True
    network = LayeredNetworkGraph(network_setup, T, dt, inter_connectivity, verbose)

    avg_time = network.run()

    # time_between_spikes = time_between_spiking(network, num_neurons, (0, 0), ((num_neurons - 1), (len(network_setup) - 1)))
    return avg_time

def run_experiment(inter_connectivities, intra_connectivities, layers, num_neurons, num_iterations, T, dt, verbose=False):
    timings = {}
    for i in range(num_iterations):
        for i in range(len(inter_connectivities)):
            timing = run_experiment_once(inter_connectivities[i], intra_connectivities[i], layers, num_neurons, T, dt, verbose=False)
            timings[(inter_connectivities[i], intra_connectivities[i])] = timing
    print(timings)
    return timings


if __name__=="__main__":
    inter_connectivities = [0.01, 0.2]
    intra_connectivities = [0.04, 0.04]
    assert len(inter_connectivities) == len(intra_connectivities), "The lengths of inter_connectivities and \
        intra_connectivities should be the same"
    layers = 3
    num_neurons = 100
    num_iterations = 1

    # n = 5
    # p = 0.2
    # prob_inter = 0.5

    T = 25
    dt = 0.01
    # obj = LayeredNetworkGraph(
    #     [(n, 0, 'g'), (n, p, 'h')], T, dt, inter_prob=prob_inter, verbose=True)
    # a = obj.run()

    run_experiment(inter_connectivities, intra_connectivities, layers, num_neurons, num_iterations, T, dt, verbose=False)