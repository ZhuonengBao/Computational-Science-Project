# Hodgkin-Huxley Neuronal Network Simulation

This project implements the **Hodgkin-Huxley (HH) model** to simulate the activity of neurons in a layered network. The model describes how action potentials in neurons are initiated and propagated, incorporating synaptic interactions between neurons in a multi-layered network.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Key Parameters](#key-parameters)

---

## Overview

This code uses the Hodgkin_Huxley model to simulate a multi layered network of neurons. It then plots timing between spiking for different between layer connectivities.

The simulation includes:
- Membrane dynamics based on HH equations.
- Synaptic currents between neurons.
- A 3D visualization of the layered network.
- Timing analysis for neuron activation across layers.

---

## File Descriptions

### `concept`
This map outlines the scope of our experimental work. Please note that some files may not function as intended.

### `modules`
This map represents the finalized framework of our project.

#### `modules/hh_model.py`
This file uses the hodgkin and huxley model to determine the voltage in a neuron over multiple timesteps.

#### `modules/layered_network.py`
This file creates a multi layered network. This network contains multiple neurons, generated from hh_model.py. It determines the interactions between these neurons and records voltage over time for each node.

#### `modules/visualize_data.py`
This file uses layered_network.py to get the voltage over time of all neurons in a layered network. 
- It then plots the voltage of a few neurons
- calculates the spiking time bewteen a start and end neuron
- plots the spiking time for different connectivities between layers
- plots the spiking time for different connectivities within layers

### `Jupyter Notebook`
This file contains summarized code for the project, progressing from a single-layer network to a multi-layered network.

---

## Requirements
Python version:
* Python 3.11

The following Python libraries are required:
- `numpy`
- `matplotlib`
- `networkx`
- `scipy`
- `random`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage
- **plot a single neuron**: run `hh_model.py`.
- **simulate multi layered network**: run 'layered_network.py'.
- **plot a few neurons per layer in a multi layered network**: call `visualize_hh_network(network, n)` in main of `visualize_data.py`
- **calculate time between a start and end neuron spiking:** call `time_between_spiking(network, n, start, end)` in main of `visualize_data.py`. 
- **plot time between spiking for different connectivities within  and between layers**: call `combined_spiking_time(n, trials, total_replace)` in main of `visualize_data.py`.
.

  ## Key parameters
  - **n**: number of neurons per network
  - **start and end**: Neurons to measure time difference between spiking. These are structured like (node, layer)
  - **trials**: The number of networks generated to run the simulations.
  - **total replace**: This determines the ammount of times the start and end neuron get replaced.
