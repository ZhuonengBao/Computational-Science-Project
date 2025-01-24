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
    def I_Na(self, V, m, h):
        """Calculate the sodium ionic current.

        Parameters:
        - V (float): Membrane potential (mV)
        - m (float): Activation gate value (between 0 and 1)
        - h (float): Inactivation gate value (between 0 and 1)

        Returns:
        Sodium ionic current (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.I_Na(-65, 0.05, 0.6), 5)  # Test for resting potential
        -1.035
        >>> round(neuron.I_Na(-50, 0.1, 0.8), 5)  # Test for depolarized potential
        -9.6
        >>> round(neuron.I_Na(50, 1.0, 1.0), 5)  # Test for maximum activation and depolarization
        0.0
        """
        return self.g_Na * (m**3) * h * (V - self.E_Na)

    def I_K(self, V, n):
        """Calculate the potassium ionic current.

        Parameters:
        - V (float): Membrane potential (mV)
        - n (float): Activation gate value (between 0 and 1)

        Returns:
        Potassium ionic current (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.I_K(-65, 0.32), 5)  # Test for resting potential
        4.52985
        >>> round(neuron.I_K(-50, 0.6), 5)  # Test for depolarized potential
        125.9712
        >>> round(neuron.I_K(-77, 0.5), 5)  # Test when V equals E_K (no current expected)
        0.0
        """
        return self.g_K * (n**4) * (V - self.E_K)

    def I_L(self, V):
        """Calculate the leaking current.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Leak current (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.I_L(-65), 5)  # Test for resting potential
        -3.1839
        >>> round(neuron.I_L(-70), 5)  # Test for hyperpolarized potential
        -4.6839
        >>> round(neuron.I_L(-54.387), 5)  # Test when V equals E_L (no current expected)
        0.0
        """
        return self.g_L * (V - self.E_L)

    # Gating variable dynamics
    def alpha_m(self, V):
        """Calculate the rate constant alpha_m for the sodium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        float: Alpha_m rate constant

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.alpha_m(-65), 5)  # Test for resting potential
        0.22356
        >>> round(neuron.alpha_m(-50), 5)  # Test for depolarized potential
        0.58198
        """
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(self, V):
        """Calculate the rate constant beta_m for the sodium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Beta_m rate constant (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.beta_m(-65), 5)  # Test for resting potential
        4.0
        >>> round(neuron.beta_m(-40), 5)  # Test for depolarized potential
        0.99741
        """
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V):
        """Calculate the rate constant alpha_h for the sodium inactivation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Alpha_h rate constant (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.alpha_h(-65), 5)  # Test for resting potential
        0.07
        >>> round(neuron.alpha_h(-50), 5)  # Test for depolarized potential
        0.03307
        """
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V):
        """Calculate the rate constant beta_h for the sodium inactivation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Beta_h rate constant (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.beta_h(-65), 5)  # Test for resting potential
        0.04743
        >>> round(neuron.beta_h(-40), 5)  # Test for depolarized potential
        0.37754
        """
        return 1 / (1 + np.exp(-(V + 35) / 10))

    def alpha_n(self, V):
        """Calculate the rate constant alpha_n for the potassium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Alpha_n rate constant (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.alpha_n(-65), 5)  # Test for resting potential
        0.0582
        >>> round(neuron.alpha_n(-50), 5)  # Test for depolarized potential
        0.12707
        """
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(self, V):
        """Calculate the rate constant beta_n for the potassium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Beta_n rate constant (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.beta_n(-65), 5)  # Test for resting potential
        0.125
        >>> round(neuron.beta_n(-50), 5)  # Test for depolarized potential
        0.10363
        """
        return 0.125 * np.exp(-(V + 65) / 80)

    # Derivatives for gating variables
    def dm_dt(self, V, m):
        """Calculate the derivative of the sodium activation gate (m).

        Parameters:
        - V (float): Membrane potential (mV)
        - m (float): Current value of the sodium activation gate.

        Returns:
        Rate of change of m (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.dm_dt(-65, 0.1), 5) # Test for resting potential
        -0.19879
        >>> round(neuron.dm_dt(-50, 0.05), 5) # Test for depolarized potential
        0.46596
        """
        return self.alpha_m(V) * (1 - m) - self.beta_m(V) * m

    def dh_dt(self, V, h):
        """Calculate the derivative of the sodium inactivation gate (h).

        Parameters:
        - V (float): Membrane potential (mV)
        - h (float): Current value of the sodium inactivation gate.

        Returns:
        Rate of change of h (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.dh_dt(-65, 0.5), 5) # Test for resting potential
        0.01129
        >>> round(neuron.dh_dt(-50, 0.6), 5) # Test for depolarized potential
        -0.09623
        """
        return self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

    def dn_dt(self, V, n):
        """Calculate the derivative of the potassium activation gate (n).

        Parameters:
        - V (float): Membrane potential (mV)
        - n (float): Current value of the potassium activation gate.

        Returns:
        Rate of change of n (float)

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.dn_dt(-65, 0.3), 5) # Test for resting potential
        0.00324
        >>> round(neuron.dn_dt(-50, 0.32), 5) # Test for depolarized potential
        0.05325
        """
        return self.alpha_n(V) * (1 - n) - self.beta_n(V) * n

    # Membrane potential
    def dV_dt(self, V, m, h, n):
        I_Na = self.I_Na(V, m, h)
        I_K = self.I_K(V, n)
        I_L = self.I_L(V)
        I_ion = I_Na + I_K + I_L
        return (self.I_ext - I_ion) / self.C_m

    # Membrane potential with synaptic current
    def dV_dt(self, V, m, h, n, I_syn):
        """Calculate the rate of change of the membrane potential (V).

        Parameters:
        - V (float): Membrane potential (mV)
        - m (float): Sodium activation gate variable (between 0 and 1).
        - h (float): Sodium inactivation gate variable (between 0 and 1).
        - n (float): Potassium activation gate variable (between 0 and 1).
        - I_syn (float): Synaptic current in µA/cm^2.

        Returns:
        float: Rate of change of membrane potential (dV/dt).

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> round(neuron.dV_dt(-65, 0.05, 0.6, 0.32, 0.0), 5) # Test for resting potential
        -0.31095
        >>> round(neuron.dV_dt(-50, 0.1, 0.3, 0.5, 5.0), 5) # Test for depolarized potential
        -53.4661
        """
        I_Na = self.I_Na(V, m, h)
        I_K = self.I_K(V, n)
        I_L = self.I_L(V)
        I_ion = I_Na + I_K + I_L
        return (self.I_ext - I_ion + I_syn) / self.C_m

    def compute_rk4(self, dt, I_syn):
        """
        Compute the RK4 intermediate steps for V, m, h, n.

        Returns:
        dict: A dictionary with keys 'k1', 'k2', 'k3', 'k4', each containing
            a tuple of (k_V, k_m, k_h, k_n).

        Examples:
        >>> neuron = HodgkinHuxleyNeuron()
        >>> rk4_steps = neuron.compute_rk4(0.01, 0.0)
        >>> isinstance(rk4_steps, dict)
        True
        >>> len(rk4_steps)
        4
        >>> all(len(step) == 4 for step in rk4_steps.values())
        True
        """
        V, m, h, n = self.V, self.m, self.h, self.n

        # Compute k1
        k1_V = self.dV_dt(V, m, h, n, I_syn)
        k1_m = self.dm_dt(V, m)
        k1_h = self.dh_dt(V, h)
        k1_n = self.dn_dt(V, n)

        # Compute k2
        k2_V = self.dV_dt(V + dt * k1_V / 2, m + dt * k1_m / 2, h + dt * k1_h / 2, n + dt * k1_n / 2, I_syn)
        k2_m = self.dm_dt(V + dt * k1_V / 2, m + dt * k1_m / 2)
        k2_h = self.dh_dt(V + dt * k1_V / 2, h + dt * k1_h / 2)
        k2_n = self.dn_dt(V + dt * k1_V / 2, n + dt * k1_n / 2)

        # Compute k3
        k3_V = self.dV_dt(V + dt * k2_V / 2, m + dt * k2_m / 2, h + dt * k2_h / 2, n + dt * k2_n / 2, I_syn)
        k3_m = self.dm_dt(V + dt * k2_V / 2, m + dt * k2_m / 2)
        k3_h = self.dh_dt(V + dt * k2_V / 2, h + dt * k2_h / 2)
        k3_n = self.dn_dt(V + dt * k2_V / 2, n + dt * k2_n / 2)

        # Compute k4
        k4_V = self.dV_dt(V + dt * k3_V, m + dt * k3_m, h + dt * k3_h, n + dt * k3_n, I_syn)
        k4_m = self.dm_dt(V + dt * k3_V, m + dt * k3_m)
        k4_h = self.dh_dt(V + dt * k3_V, h + dt * k3_h)
        k4_n = self.dn_dt(V + dt * k3_V, n + dt * k3_n)

        return {
            "k1": (k1_V, k1_m, k1_h, k1_n),
            "k2": (k2_V, k2_m, k2_h, k2_n),
            "k3": (k3_V, k3_m, k3_h, k3_n),
            "k4": (k4_V, k4_m, k4_h, k4_n),
        }

    def update_state(self, dt, rk4_steps):
        """
        Update the state variables V, m, h, n using the RK4 steps.

        Parameters:
        dt (float): Time step.
        rk4_steps (dict): Dictionary of RK4 intermediate steps (k1, k2, k3, k4).
        """
        k1, k2, k3, k4 = rk4_steps["k1"], rk4_steps["k2"], rk4_steps["k3"], rk4_steps["k4"]

        self.V += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        self.m += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        self.h += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        self.n += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6

    def detect_spike(self, threshold=-50):
        """
        Detect and record spike timing.

        Parameters:
        threshold (float): Voltage threshold for spike detection (default: -50 mV).
        """
        if self.V > threshold and self.last_spike_time is None:
            self.last_spike_time = 0  # Spike detected
        elif self.V <= threshold:
            self.last_spike_time = None  # Reset spike detection

    def step(self, dt, I_syn):
        """
        Perform a single simulation step.

        Parameters:
        dt (float): Time step in milliseconds.
        I_syn (float): Synaptic current input in µA.
        """
        # Compute RK4 intermediate steps
        rk4_steps = self.compute_rk4(dt, I_syn)

        # Update the state variables
        self.update_state(dt, rk4_steps)

        # Detect spike events
        self.detect_spike()


def main():
    # Parameters
    T = 10.0
    dt = 0.01
    step = np.arange(0, T, dt)
    I_inp = np.full(len(step), 15.0)
    I_inp[0] = 0
    I_inp[-1] = 0

    # Post-Synaptic Constants
    tau = 30
    weight = 0.1

    # Create neuron
    neuron = HodgkinHuxleyNeuron()

    # Record data
    V_record = []
    I_out = np.zeros(len(step))

    edit = 0
    diff = 0
    for i, t in enumerate(step):
        neuron.step(dt, I_inp[i])
        V_record.append(neuron.V)
        I_out[i] = weight * np.exp(-(-65.0 - neuron.V) / tau)

    # Plot
    plt.figure(figsize=(8, 6))

    # Currents
    plt.subplot(2, 1, 1)
    plt.plot(step, I_inp, label='Input')
    plt.plot(step, I_out, label='Output')
    plt.title('Stimulus')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA/cm$^2$)')
    plt.legend()
    plt.grid()

    # Membrane potential V
    plt.subplot(2, 1, 2)
    plt.plot(step, V_record)
    plt.plot(step, np.full(len(step), -65.0),
             linestyle='--', color='gray', label='Resting')
    plt.title('Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
