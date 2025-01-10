import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Time
start = 0
end = 3
step = 0.001
t = np.arange(start, end + step, step)

# Constants
C = 0.1
gK = 36
gNa = 120
gL = 0.3

VK = -77
VNa = 50
VL = -54.4

I = np.zeros(len(t)) 
V = np.zeros(len(t)) 
n = np.zeros(len(t)) 
m = np.zeros(len(t)) 
h = np.zeros(len(t)) 


# Initial Conductances
# I[int(0.5/step):int(1.0/step)] = 190
print(I)
n[0] = 0.317
m[0] = 0.05
h[0] = 0.6
V[0] = -65


def dVdt(V, Icur, nvar, mvar, hvar):
    I_Na = np.power(mvar, 3) * gNa * hvar * (V-VNa)
    I_K = np.power(nvar, 4) * gK * (V-VK)
    I_L = gL * (V-VL)
    I_ion = Icur - I_K - I_Na - I_L
    return I_ion / C


def dNdt(N, alpha, beta):
    return alpha * (1 - N) - beta * N


# Simulate
for i in range(len(t)-1):
    # coefficients
    alpha_n = 0.01 * (V[i] + 10) / (np.exp((V[i] + 10) / 10) - 1)
    alpha_m = 0.01 * (V[i] + 25) / (np.exp((V[i] + 25) / 10) - 1)
    alpha_h = 0.07 * np.exp(V[i] / 20)
    beta_n = 0.125 * np.exp(V[i] / 80)
    beta_m = 4 * np.exp(V[i] / 18)
    beta_h = 1 / (np.exp((V[i] + 30) / 10) + 1)

    # calculate derivatives using Euler first order approximation
    k1 = dVdt(V[i], I[i], n[i], m[i], h[i])
    k2 = dVdt(V[i] + step * k1 / 2, I[i], n[i], m[i], h[i])
    k3 = dVdt(V[i] + step * k2 / 2, I[i], n[i], m[i], h[i])
    k4 = dVdt(V[i] + step * k3, I[i], n[i], m[i], h[i])
    V[i+1] = V[i] + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    k1 = dNdt(n[i], alpha_n, beta_n)
    k2 = dNdt(n[i] + step * k1 / 2, alpha_n, beta_n)
    k3 = dNdt(n[i] + step * k2/ 2, alpha_n, beta_n)
    k4 = dNdt(n[i] + step * k3, alpha_n, beta_n)
    n[i+1] = n[i] + (step / 6) * (k1 + 2 * k2 + 2* k3 + k4)

    k1 = dNdt(m[i], alpha_m, beta_m)
    k2 = dNdt(m[i] + step * k1 / 2, alpha_m, beta_m)
    k3 = dNdt(m[i] + step * k2 / 2, alpha_m, beta_m)
    k4 = dNdt(m[i] + step * k3, alpha_m, beta_m) 
    m[i+1] = m[i] + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    k1 = dNdt(h[i], alpha_h, beta_h)
    k2 = dNdt(h[i] + step * k1 / 2, alpha_h, beta_h)
    k3 = dNdt(h[i] + step * k2/ 2, alpha_h, beta_h)
    k4 = dNdt(h[i] + step * k3, alpha_h, beta_h)  
    h[i+1] = h[i] + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Display Results
plt.figure()
plt.plot(t, V, color='b')
plt.show()

