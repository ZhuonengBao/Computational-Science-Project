"""
 * Hodgkin-Huxley Model
 *
 * Author's: Zhuoneng Bao, Mink van Maanen, Lynne Vogel and Babet Wijsman
 * Date: 9 January 2025
 * Description: This program implements the Hodgkin-Huxley model of a neuron, 
 *              describing the initiation and propagation of action potentials.
 *              The constants are based on the work of 
 *              Rinzel, J. and Ermentrout, G.B. (1998),
 *              "Analysis of Neural Excitability and Oscillations."
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class HodgkinHuxleyNeuron:
    def __init__(self):
        # Membrane properties
        self.C_m = 1.0  # Membrane capacitance
        self.V = -65.0  # Initial membrane potential

        # Sodium (Na) channel
        self.g_Na = 120.0  # Maximum conductance
        self.E_Na = 50.0   # Reversal potential
        self.m = 0.05      # Activation gate
        self.h = 0.6       # Inactivation gate

        # Potassium (K) channel
        self.g_K = 36.0    # Maximum conductance
        self.E_K = -77.0   # Reversal potential
        self.n = 0.32      # Activation gate

        # Leak channel
        self.g_L = 0.3      # Maximum conductance
        self.E_L = -54.387  # Reversal potential

        # External current
        self.I_ext = 0.0

        # Last time neuron fired
        self.last_spike_time = None

    # Ionic currents
    def I_Na(self, V, m, h): return self.g_Na * (m**3) * h * (V - self.E_Na)
    def I_K(self, V, n): return self.g_K * (n**4) * (V - self.E_K)
    def I_L(self, V): return self.g_L * (V - self.E_L)
    
    # Gating variable dynamics
    def alpha_m(self, V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    def beta_m(self, V): return 4.0 * np.exp(-(V + 65) / 18)
    def alpha_h(self, V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(self, V): return 1 / (1 + np.exp(-(V + 35) / 10))
    def alpha_n(self, V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    def beta_n(self, V): return 0.125 * np.exp(-(V + 65) / 80)

    # Derivatives for gating variables
    def dm_dt(self, V, m): return self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
    def dh_dt(self, V, h): return self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
    def dn_dt(self, V, n): return self.alpha_n(V) * (1 - n) - self.beta_n(V) * n

    # Membrane potential
    def dV_dt(self, V, m, h, n):
        I_Na = self.I_Na(V, m, h)
        I_K = self.I_K(V, n)
        I_L = self.I_L(V)
        I_ion = I_Na + I_K + I_L
        return (self.I_ext - I_ion) / self.C_m

    # Membrane potential with synaptic current
    def dV_dt(self, V, m, h, n, I_syn):
        I_Na = self.I_Na(V, m, h)
        I_K = self.I_K(V, n)
        I_L = self.I_L(V)
        I_ion = I_Na + I_K + I_L
        return (self.I_ext - I_ion + I_syn) / self.C_m

    # Perform one step using Runge-Kutta
    def step(self, dt, I_syn):
        V = self.V
        m, h, n = self.m, self.h, self.n

        # I_syn = self.synaptic_current(t_const, t_firings, w, tau)
        k1_V = self.dV_dt(V, m, h, n, I_syn)
        # k1_V = self.dV_dt(V, m, h, n)
        k1_m = self.dm_dt(V, m)
        k1_h = self.dh_dt(V, h)
        k1_n = self.dn_dt(V, n)

        k2_V = self.dV_dt(V + dt * k1_V / 2, m + dt * k1_m / 2, h + dt * k1_h / 2, n + dt * k1_n / 2, I_syn)
        k2_m = self.dm_dt(V + dt * k1_V / 2, m + dt * k1_m / 2)
        k2_h = self.dh_dt(V + dt * k1_V / 2, h + dt * k1_h / 2)
        k2_n = self.dn_dt(V + dt * k1_V / 2, n + dt * k1_n / 2)

        k3_V = self.dV_dt(V + dt * k2_V / 2, m + dt * k2_m / 2, h + dt * k2_h / 2, n + dt * k2_n / 2, I_syn)
        k3_m = self.dm_dt(V + dt * k2_V / 2, m + dt * k2_m / 2)
        k3_h = self.dh_dt(V + dt * k2_V / 2, h + dt * k2_h / 2)
        k3_n = self.dn_dt(V + dt * k2_V / 2, n + dt * k2_n / 2)

        k4_V = self.dV_dt(V + dt * k3_V, m + dt * k3_m, h + dt * k3_h, n + dt * k3_n, I_syn)
        k4_m = self.dm_dt(V + dt * k3_V, m + dt * k3_m)
        k4_h = self.dh_dt(V + dt * k3_V, h + dt * k3_h)
        k4_n = self.dn_dt(V + dt * k3_V, n + dt * k3_n)

        # Update values
        self.V += dt * (k1_V + 2 * k2_V + 2 * k3_V + k4_V) / 6
        self.m += dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
        self.h += dt * (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6
        self.n += dt * (k1_n + 2 * k2_n + 2 * k3_n + k4_n) / 6

        if self.V > -50 and self.last_spike_time is None:
            # Record the time of the spike
            self.last_spike_time = dt  
        elif self.V <= -50:
            # Reset if not firing
            self.last_spike_time = None  

def main():
    # Parameters
    T = 50.0
    dt = 0.01
    time = np.arange(0, T, dt)
    
    # Create neuron
    neuron = HodgkinHuxleyNeuron()
    neuron.I_ext = 10.0  # External current
    I_syn = 0
    
    # Record data
    V_record = []
    for t in time:
        neuron.step(dt, I_syn)
        V_record.append(neuron.V)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(time, V_record)
    plt.show()

if __name__ == "__main__":
    main()

