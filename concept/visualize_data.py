"""
 * visualize and analyse data
 *
 * Author's: Zhuoneng Bao, Mink van Maanen, Lynne Vogel and Babet Wijsman
 * Date: 18 January 2025
 * Description: This file visualizes data and calculates the time interval between spiking.

 """
import matplotlib.pyplot as plt
from layered_network import LayeredNetworkGraph
import networkx as nx
from hh_model import HodgkinHuxleyNeuron

import numpy as np
import scipy.stats as st

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

    print('Ran network')

    fired = False
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
            fired = True
            break

    # calculate time between start and end neuron firing
    if fired:
        time_interval = end_time - start_time
        print(time_interval, 'ms')

    return time_interval

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset.

    Parameters:
        data (list): List of data points.
        confidence (float): Confidence level (default: 0.95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    if len(data) < 2:
        raise ValueError("Insufficient data to calculate confidence interval.")
    mean = np.mean(data)
    sem = st.sem(data)
    interval = st.t.interval(confidence, df=len(data) - 1, loc=mean, scale=sem)
    return interval


def calculate_spiking_time(network, start, end):
    """
    Calculate the time interval for spiking between start and end neurons.
    """
    time_interval = time_between_spiking(network, start, end)
    return time_interval if not np.isnan(time_interval) else None


def process_trials(trials, n, property_value, mode, prob):
    """
    Process trials for either 'within-layer' or 'between-layer' mode.
    
    Parameters:
        trials (int): Number of trials to run.
        n (int): Number of nodes in each layer.
        property_value (float): Connectivity or probability, depending on the mode.
        mode (str): Mode of operation ('within' or 'between').
        prob (float): Probability for LayeredNetworkGraph for 'within' mode.

    Returns:
        list: List of spiking time intervals.
    """
    data = []

    for _ in range(trials):
        print(f'Mode: {mode}, trials: {_}')
        # Create three layers of Erdős-Rényi graphs
        g = generate_erdos_renyi_digraph(n, p=property_value if mode == 'within' else 1)
        h = generate_erdos_renyi_digraph(n, p=property_value if mode == 'within' else 1)
        i = nx.erdos_renyi_graph(n, p=property_value if mode == 'within' else 1)

        print('Generated erdos renyi graphs')

        # Create the layered network
        network = LayeredNetworkGraph([g, h, i], prob=1 if mode == 'within' else property_value)

        print('Generated network')

        # Compute spiking time interval
        start = (0, 0)
        end_nodes = [(curr_node,network.total_layers - 1) for curr_node in range(n)]
        replace_time = []
        not_fired = 0
        for end in end_nodes:
            time_interval = time_between_spiking(network, start, end)

            print(f'Calculated time interval for endnode: {end}')

            if not np.isnan(time_interval):
                replace_time.append(time_interval)
            else: 
                not_fired += 1

        if replace_time:    
            data.append(np.mean(replace_time))

        print('Calculated data')
    
    average_not_fired = not_fired / trials
    

    return data, average_not_fired


def run_combined_spiking_time(n, trials, prob):
    """
    Run spiking time analysis for both within-layer and between-layer modes.
    
    Parameters:
        n (int): Number of nodes in each layer.
        trials (int): Number of trials to run for averaging.
        prob (float): Probability parameter for within-layer spiking.
    """
    within_properties = np.linspace(0.5, 1, 3)  # Connectivity range for 'within'
    between_properties = np.linspace(0.9, 1, 3)  # Probability range for 'between'

    within_mean_time, within_lower_CI_list, within_higher_CI_list = [], [], []
    between_mean_time, between_lower_CI_list, between_higher_CI_list = [], [], []
    within_list = []
    between_list = []
    not_fired_within = {}
    not_fired_between = {}

    for within_connectivity, between_prob in zip(within_properties, between_properties):
        # Process 'within-layer' trials
        within_data, average_not_fired = process_trials(trials, n, within_connectivity, mode='within', prob=prob)
        not_fired_within[within_connectivity] = average_not_fired

        if within_data:
            within_mean_time.append(np.mean(within_data))
            lower_CI, upper_CI = calculate_confidence_interval(within_data)
            within_lower_CI_list.append(lower_CI)
            within_higher_CI_list.append(upper_CI)
            within_list.append(within_connectivity)

        # Process 'between-layer' trials
        between_data, average_not_fired = process_trials(trials, n, between_prob, mode='between', prob=prob)
        not_fired_between[between_prob] = average_not_fired

        if between_data:
            between_mean_time.append(np.mean(between_data))
            lower_CI, upper_CI = calculate_confidence_interval(between_data)
            between_lower_CI_list.append(lower_CI)
            between_higher_CI_list.append(upper_CI)
            between_list.append(between_prob)

    # Plot results for 'within'
    plt.plot(within_list, within_mean_time, label='Within-Layer')
    plt.fill_between(within_list, within_lower_CI_list, within_higher_CI_list, alpha=0.1)

    plt.ylim(0, 20)
    plt.xlabel('Connectivity')
    plt.ylabel('Time Between First and Last Neuron')
    plt.legend()
    plt.title('Spiking Time: Within vs Between Layers')
    plt.show()

    # Plot results for 'between'
    plt.plot(between_list, between_mean_time, label='Between-Layer')
    plt.fill_between(between_list, between_lower_CI_list, between_higher_CI_list, alpha=0.1)

    plt.ylim(0, 20)
    plt.xlabel('Probability')
    plt.ylabel('Time Between First and Last Neuron')
    plt.legend()
    plt.title('Spiking Time: Within vs Between Layers')
    plt.show()

    print(within_mean_time)
    print(between_mean_time)
    print(not_fired_between)
    print(not_fired_within)

if __name__ == "__main__":
    n = 50
    g = generate_erdos_renyi_digraph(n, p=0.3)
    h = generate_erdos_renyi_digraph(n, p=0.3)
    i = generate_erdos_renyi_digraph(n, p=0.3)
    prob = 0.025

    # Create the layered network
    network = LayeredNetworkGraph([g, h, i], prob)
    
    visualize_hh_network(network, n)
    #calc_potentials_per_layer(network, n)
    #time_between_spiking(network, (0,0))
    # plot_spiking_time_within(n, 2, 2, prob)
    # plot_spiking_time_between(n, 2, 2)
    run_combined_spiking_time(10, 2, 1)