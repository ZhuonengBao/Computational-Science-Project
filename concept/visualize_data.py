import matplotlib.pyplot as plt
from hh_model import HodgkinHuxleyNeuron
from layered_network import LayeredNetworkGraph
import networkx as nx

def visualize_hh_network(network, n):

    # Run the Hodgkin-Huxley model and get the data
    V_record, time = network.run_hh_network()
    selected_indices = []

    # Choose some neurons to visualize
    for layer in range(network.total_layers):
        # Convert the generator expression to a list
        selected_indices.extend([(0, layer), (n-1, layer)])

    # Plot the selected neurons
    plt.figure(figsize=(12, 8))

    for node in selected_indices:
        plt.plot(time, V_record[node], label=f'Neuron {node}')

    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Selected Neurons Activity")
    plt.show()

def calc_potentials_per_layer(network, n):

    V_record, time = network.run_hh_network()
    layer_time = []

    for layer in range(network.total_layers):
        indices = [(node, layer) for node in range(n)]
        timing = []
        neurons_reached = 0

        for node in indices:
            if node not in V_record:
                print(f"Warning: No data recorded for node {node}")
            for t, V in zip(time, V_record[node]):
                if V > -50:
                    neurons_reached += 1
                    timing.append(t)
                    print(f"Node {node} spiked at time {t} with potential {V}")
                    break
            
        layer_time.append(max(timing))

            
    plt.bar(range(network.total_layers), layer_time)
    plt.xlabel("Layer")
    plt.ylabel("Time to Spike (ms)")
    plt.title("Spike Timing Across Layers")
    plt.show()

def time_between_spiking(network, n, start, end):
    V_record, time = network.run_hh_network()

    for t, V in zip(time, V_record[start]):
        if V > -50:
            start_time = t
            break

    for t, V in zip(time, V_record[end]):
        if V > -50:
            end_time = t
            break

    print(end_time - start_time, 'ms')


if __name__ == "__main__":
    n = 500
    k = round(n / 67)
    p = 0.0001
    g = nx.watts_strogatz_graph(n, k, p)
    h = nx.watts_strogatz_graph(n, k, p)
    i = nx.watts_strogatz_graph(n, k, p)
    combined_networks = [g, h, i]

    # Create the layered network
    network = LayeredNetworkGraph(combined_networks)
    #visualize_hh_network(network, n)
    #calc_potentials_per_layer(network, n)
    time_between_spiking(network, n, (0,0), ((n - 1), (len(combined_networks) - 1)))