import matplotlib.pyplot as plt
from hh_model import HodgkinHuxleyNeuron
from layered_network import LayeredNetworkGraph
import networkx as nx
import pandas as pd
import numpy as np

def visualize_hh_network(network, n):
    """
    This funtion plots the voltage over time for the first and last node of each layer in a network.
    """
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
    """
    This function calculated the time it takes for each layer of the network to be fully activated
    """
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
                if V > -54.387:
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

def time_between_spiking(network, start, end):
    """
    This function calculated the time difference in spiking of a start and end neuron
    """
    V_record, time = network.run_hh_network()

    for t, V in zip(time, V_record[start]):
        if V > -54.387:
            start_time = t
            break

    for t, V in zip(time, V_record[end]):
        if V > -54.387:
            end_time = t
            break

        end_time = start_time

    time_interval = end_time - start_time
    print(end_time - start_time, 'ms')
    return time_interval

def plot_spiking_time(n, start, end, trials):
    properties = np.linspace(0, 1, 5)
    data = []

    for connectivity in properties:
        results = []
        for _ in range(trials):
            g = nx.erdos_renyi_graph(n, p= connectivity)
            h = nx.erdos_renyi_graph(n, p= connectivity)
            i = nx.erdos_renyi_graph(n, p= connectivity)

            # Create the layered network
            network = LayeredNetworkGraph([g, h, i])

            time_interval = time_between_spiking(network, n, start, end)
            data.append(time_interval)

            results.append((connectivity, np.mean(data), np.std(data)))

        df = pd.DataFrame(results, columns=["Connectivity", "Mean", "Std"])

        # Plot the results
        plt.errorbar(df["Connectivity"], df["Mean"], yerr=df["Std"], fmt='o', capsize=5, label="Spike Timing")
        plt.xlabel('connectivity')
        plt.ylabel('time between first and last neuron')
        plt.show()
   

if __name__ == "__main__":
    n = 50
    g = nx.erdos_renyi_graph(n, p=0.4)
    h = nx.erdos_renyi_graph(n, p=0.4)
    i = nx.erdos_renyi_graph(n, p=0.4)

    # Create the layered network
    network = LayeredNetworkGraph([g, h, i])
    
    #visualize_hh_network(network, n)
    #calc_potentials_per_layer(network, n)
    #time_between_spiking(network, n, (0,0), (40,2))
    plot_spiking_time(n, (0,0), (49,2), 3)