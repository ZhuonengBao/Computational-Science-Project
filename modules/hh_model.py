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
from numba.experimental import jitclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


@jitclass
class HodgkinHuxleyNeuron:
    dt: float
    C_m: float
    V: float
    g_Na: float
    E_Na: float
    m: float
    h: float
    g_K: float
    E_K: float
    n: float
    g_L: float
    E_L: float
    """
    Creates a neuron object based on the Hodgkin-Huxley model.

    The constants used are derived from the work of Rinzel, J. and
    Ermentrout, G.B. (1998),
    "Analysis of Neural Excitability and Oscillations."

    Parameters:
    - step (float): The time step used to approximate the ordinary
      differential equations (ODEs) of the Hodgkin-Huxley model using the
      Runge-Kutta numerical method.

    Returns:
    - (object): An instance representing a neuron modeled using
                the Hodgkin-Huxley model.
    """

    def __init__(self, step: float):
        # Timestep
        self.dt = step

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
        >>> neuron = HodgkinHuxleyNeuron(0.01)
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
        """
        return self.g_K * (n**4) * (V - self.E_K)

    def I_L(self, V):
        """Calculate the leaking current.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Leak current (float)
        """
        return self.g_L * (V - self.E_L)

    # Gating variable dynamics
    def alpha_m(self, V):
        """Calculate the rate constant alpha_m for the sodium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        float: Alpha_m rate constant
        """
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(self, V):
        """Calculate the rate constant beta_m for the sodium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Beta_m rate constant (float)
        """
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V):
        """Calculate the rate constant alpha_h for the sodium inactivation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Alpha_h rate constant (float)
        """
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V):
        """Calculate the rate constant beta_h for the sodium inactivation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Beta_h rate constant (float)
        """
        return 1 / (1 + np.exp(-(V + 35) / 10))

    def alpha_n(self, V):
        """Calculate the rate constant alpha_n for the potassium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Alpha_n rate constant (float)
        """
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(self, V):
        """Calculate the rate constant beta_n for the potassium activation gate.

        Parameters:
        - V (float): Membrane potential (mV)

        Returns:
        Beta_n rate constant (float)
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
        """
        return self.alpha_m(V) * (1 - m) - self.beta_m(V) * m

    def dh_dt(self, V, h):
        """Calculate the derivative of the sodium inactivation gate (h).

        Parameters:
        - V (float): Membrane potential (mV)
        - h (float): Current value of the sodium inactivation gate.

        Returns:
        Rate of change of h (float)
        """
        return self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

    def dn_dt(self, V, n):
        """Calculate the derivative of the potassium activation gate (n).

        Parameters:
        - V (float): Membrane potential (mV)
        - n (float): Current value of the potassium activation gate.

        Returns:
        Rate of change of n (float)
        """
        return self.alpha_n(V) * (1 - n) - self.beta_n(V) * n

    # Membrane potential
    def dV_dt(self, I, V, m, h, n):
        """Calculate the rate of change of the membrane potential (V).

        Parameters:
        - I (float): Synaptic current in µA/cm^2.
        - V (float): Membrane potential (mV)
        - m (float): Sodium activation gate variable (between 0 and 1).
        - h (float): Sodium inactivation gate variable (between 0 and 1).
        - n (float): Potassium activation gate variable (between 0 and 1).

        Returns:
        float: Rate of change of membrane potential (dV/dt).
        """
        I_Na = self.I_Na(V, m, h)
        I_K = self.I_K(V, n)
        I_L = self.I_L(V)
        I_ion = I_Na + I_K + I_L
        return (I - I_ion) / self.C_m

    def step(self, I):
        """
        Compute a numerical step using Runge-Kutta (RK4).

        Parameters:
        - I (float): Synaptic current in µA/cm^2.

        Side-effect:
            Numerically computes the voltage for the next time step and
            updates the self.V attribute.
        """
        V = self.V
        m, h, n = self.m, self.h, self.n

        k1_V = self.dV_dt(I, V, m, h, n)
        k1_m = self.dm_dt(V, m)
        k1_h = self.dh_dt(V, h)
        k1_n = self.dn_dt(V, n)

        k2_V = self.dV_dt(
            I,
            V + self.dt * k1_V / 2,
            m + self.dt * k1_m / 2,
            h + self.dt * k1_h / 2,
            n + self.dt * k1_n / 2)
        k2_m = self.dm_dt(V + self.dt * k1_V / 2, m + self.dt * k1_m / 2)
        k2_h = self.dh_dt(V + self.dt * k1_V / 2, h + self.dt * k1_h / 2)
        k2_n = self.dn_dt(V + self.dt * k1_V / 2, n + self.dt * k1_n / 2)

        k3_V = self.dV_dt(
            I,
            V + self.dt * k2_V / 2,
            m + self.dt * k2_m / 2,
            h + self.dt * k2_h / 2,
            n + self.dt * k2_n / 2)
        k3_m = self.dm_dt(V + self.dt * k2_V / 2, m + self.dt * k2_m / 2)
        k3_h = self.dh_dt(V + self.dt * k2_V / 2, h + self.dt * k2_h / 2)
        k3_n = self.dn_dt(V + self.dt * k2_V / 2, n + self.dt * k2_n / 2)

        k4_V = self.dV_dt(
            I,
            V + self.dt * k3_V,
            m + self.dt * k3_m,
            h + self.dt * k3_h,
            n + self.dt * k3_n)
        k4_m = self.dm_dt(V + self.dt * k3_V, m + self.dt * k3_m)
        k4_h = self.dh_dt(V + self.dt * k3_V, h + self.dt * k3_h)
        k4_n = self.dn_dt(V + self.dt * k3_V, n + self.dt * k3_n)

        # Update values
        self.V += self.dt * (k1_V + 2 * k2_V + 2 * k3_V + k4_V) / 6
        self.m += self.dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
        self.h += self.dt * (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6
        self.n += self.dt * (k1_n + 2 * k2_n + 2 * k3_n + k4_n) / 6


def test():
    """
    >>> neuron = HodgkinHuxleyNeuron(0.01)

    # Sodium (Na) Test
    >>> round(neuron.I_Na(-65, 0.05, 0.6), 5)  # Test for resting potential
    -1.035
    >>> round(neuron.I_Na(-50, 0.1, 0.8), 5)  # Test for depolarized potential
    -9.6
    >>> round(neuron.I_Na(50, 1.0, 1.0), 5)  # Test for maximum activation and depolarization
    0.0

    # Potassium (K) Test
    >>> round(neuron.I_K(-65, 0.32), 5)  # Test for resting potential
    4.52985
    >>> round(neuron.I_K(-50, 0.6), 5)  # Test for depolarized potential
    125.9712
    >>> round(neuron.I_K(-77, 0.5), 5)  # Test when V equals E_K (no current expected)
    0.0

    # Leakage Test
    >>> round(neuron.I_L(-65), 5)  # Test for resting potential
    -3.1839
    >>> round(neuron.I_L(-70), 5)  # Test for hyperpolarized potential
    -4.6839
    >>> round(neuron.I_L(-54.387), 5)  # Test when V equals E_L (no current expected)
    0.0


    # Variable Dynamics Test
    >>> round(neuron.alpha_m(-65), 5)  # Test for resting potential
    0.22356
    >>> round(neuron.alpha_m(-50), 5)  # Test for depolarized potential
    0.58198
    >>> round(neuron.beta_m(-65), 5)  # Test for resting potential
    4.0
    >>> round(neuron.beta_m(-40), 5)  # Test for depolarized potential
    0.99741
    >>> round(neuron.alpha_h(-65), 5)  # Test for resting potential
    0.07
    >>> round(neuron.alpha_h(-50), 5)  # Test for depolarized potential
    0.03307
    >>> round(neuron.beta_h(-65), 5)  # Test for resting potential
    0.04743
    >>> round(neuron.beta_h(-40), 5)  # Test for depolarized potential
    0.37754
    >>> round(neuron.alpha_n(-65), 5)  # Test for resting potential
    0.0582
    >>> round(neuron.alpha_n(-50), 5)  # Test for depolarized potential
    0.12707
    >>> round(neuron.beta_n(-65), 5)  # Test for resting potential
    0.125
    >>> round(neuron.beta_n(-50), 5)  # Test for depolarized potential
    0.10363
    >>> round(neuron.dm_dt(-65, 0.1), 5) # Test for resting potential
    -0.19879
    >>> round(neuron.dm_dt(-50, 0.05), 5) # Test for depolarized potential
    0.46596
    >>> round(neuron.dh_dt(-65, 0.5), 5) # Test for resting potential
    0.01129
    >>> round(neuron.dh_dt(-50, 0.6), 5) # Test for depolarized potential
    -0.09623
    >>> round(neuron.dn_dt(-65, 0.3), 5) # Test for resting potential
    0.00324
    >>> round(neuron.dn_dt(-50, 0.32), 5) # Test for depolarized potential
    0.05325

    # Voltage Test
    >>> round(neuron.dV_dt(0.0, -65, 0.05, 0.6, 0.32), 5) # Test for resting potential
    -0.31095
    >>> round(neuron.dV_dt(5.0, -50, 0.1, 0.3, 0.5), 5) # Test for depolarized potential
    -53.4661

    # Runge-Kutta Numerical Test
    >>> neuron.step(0.0)
    >>> round(neuron.V)
    -65
    """
    return


def main():
    T = 20
    dt = 0.01
    time = np.arange(0, T, dt)

    # Neuron
    neuron = HodgkinHuxleyNeuron(dt)

    # Post-Synaptic constants
    weight = 0.1
    tau = 30

    # Stimulus
    I_inp = np.zeros(len(time))
    I_inp[int(0.5 / dt):int(1.0 / dt)] = 15.0

    # Record data
    V_record = []
    I_out = np.zeros(len(time))

    for i, t in enumerate(time):
        neuron.step(I_inp[i])
        V_record.append(neuron.V)
        I_out[i] = weight * np.exp(-(-65.0 - neuron.V) / tau)

    # Plot
    plt.figure(figsize=(8, 6))

    # Membrane potential V
    plt.subplot(2, 1, 2)
    plt.plot(time, V_record)
    plt.plot(time, np.full(len(time), -65),
             linestyle='--', color='gray', label='Resting')
    plt.title('Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.grid()
    plt.legend()

    # Currents
    plt.subplot(2, 1, 1)
    plt.plot(time, I_inp, label='Input')
    plt.plot(time, I_out, label='Output')
    plt.title('Stimulus')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA/cm$^2$)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # $terminal: python3 -m doctest -v modules/layered_network.py
    test()
