import matplotlib.pyplot as plt
from layered_network import LayeredNetworkGraph
import networkx as nx

import numpy as np
import scipy.stats as st
import random

def generate_erdos_renyi_digraph(n, p):
    G = nx.erdos_renyi_graph(n, p, directed=True)

    # Remove bidirected edges
    for u, v in list(G.edges()):
        if G.has_edge(v, u):
            G.remove_edge(v, u)

    # Create neuron objects
    for node in G.nodes():
        G.nodes[node]['neuron'] = HodgkinHuxleyNeuron()

    return G

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

def time_between_spiking(network, start, end):
    """
    This function calculated the time difference in spiking of a start and end neuron
    """

    V_record, time = network.run_hh_network()
    fired = True
    time_interval = np.nan

    # check the time the start neuron gets to an action potential
    for t, V in zip(time, V_record[start]):
        # an action potential occurs when the threshold is reached
        if V > -50:
            start_time = t
            break

    # check the time an action potential occurs in the end neuron
    for t, V in zip(time, V_record[end]):
        # an action potential occurs when the threshold is reached
        if V > -50:
            end_time = t
            break

        # don't count if the neuron did not fire
        fired = False

    # calculate time between start and end neuron firing
    if fired:
        time_interval = end_time - start_time
        print(time_interval, 'ms')

    return time_interval

def plot_spiking_time_within(n, trials, total_replace):
    """
    This function plots time interval for different connectivities (could be changed to other variable later on)
    """
    properties = np.linspace(0.5, 1, 2)

    mean_time = []
    lower_CI_list = []
    higher_CI_list = []
    connectivity_list = []
    for connectivity in properties:
        data = []
        for _ in range(trials):
            g = nx.erdos_renyi_graph(n, p= connectivity)
            h = nx.erdos_renyi_graph(n, p= connectivity)
            i = nx.erdos_renyi_graph(n, p= connectivity)

            # Create the layered network
            network = LayeredNetworkGraph([g, h, i])

            start_nodes = [(random.randrange(0, network.activate - 1), 0) for _ in range(total_replace)]
            end_nodes = [(random.randrange(0, n - 1), network.total_layers - 1) for _ in range(total_replace)]
            replace_time = []

            for start, end in zip(start_nodes, end_nodes):
                time_interval = time_between_spiking(network, start, end)

                if not np.isnan(time_interval):
                    replace_time.append(time_interval)

            if replace_time:    
                data.append(np.mean(replace_time))

        if data:       
            # calculate confidence interval
            confidence_interval = st.t.interval(0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
            lower_CI, higher_CI = confidence_interval
            lower_CI_list.append(lower_CI)
            higher_CI_list.append(higher_CI)

            mean_time.append(np.mean(data))
            connectivity_list.append(connectivity)

    #plot time per connectivity with confidence interval.
    plt.plot(connectivity_list, mean_time)
    plt.ylim(0, 20)
    plt.fill_between(connectivity_list, lower_CI_list, higher_CI_list, alpha=0.1)
    plt.xlabel('connectivity')
    plt.ylabel('time between first and last neuron')
    plt.show()

def plot_spiking_time_between(n, trials, total_replace):
    """
    This function plots time interval for different connectivities (could be changed to other variable later on)
    """
    properties = np.linspace(0.01, 0.05, 2)

    mean_time = []
    lower_CI_list = []
    higher_CI_list = []
    prob_list = []
    for prob in properties:
        data = []
        for _ in range(trials):
            g = nx.erdos_renyi_graph(n, p= 0.4)
            h = nx.erdos_renyi_graph(n, p= 0.4)
            i = nx.erdos_renyi_graph(n, p= 0.4)

            # Create the layered network
            network = LayeredNetworkGraph([g, h, i], prob)

            start_nodes = [(random.randrange(0, network.activate - 1), 0) for _ in range(total_replace)]
            end_nodes = [(random.randrange(0, n - 1), network.total_layers - 1) for _ in range(total_replace)]
            replace_time = []

            for start, end in zip(start_nodes, end_nodes):
                time_interval = time_between_spiking(network, start, end)

                if not np.isnan(time_interval):
                    replace_time.append(time_interval)

            if replace_time:  
                data.append(np.mean(replace_time))

        if data:       
            # calculate confidence interval
            confidence_interval = st.t.interval(0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
            lower_CI, higher_CI = confidence_interval
            lower_CI_list.append(lower_CI)
            higher_CI_list.append(higher_CI)

            mean_time.append(np.mean(data))
            prob_list.append(prob)

    #plot time per connectivity with confidence interval.
    plt.plot(prob_list, mean_time)
    plt.ylim(0, 20)
    plt.fill_between(prob_list, lower_CI_list, higher_CI_list, alpha=0.1)
    plt.xlabel('connectivity')
    plt.ylabel('time between first and last neuron')
    plt.show()

if __name__ == "__main__":
    n = 50
    g = nx.erdos_renyi_graph(n, p=0.3)
    h = nx.erdos_renyi_graph(n, p=0.3)
    i = nx.erdos_renyi_graph(n, p=0.3)
    prob = 0.025

    # Create the layered network
    network = LayeredNetworkGraph([g, h, i], prob)
    
    #visualize_hh_network(network, n)
    #calc_potentials_per_layer(network, n)
    #time_between_spiking(network, (0,0))
    #plot_spiking_time_within(n, 2, 2)
    plot_spiking_time_between(n, 2, 2)