    dt = 0.001
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
    plt.plot(time / time[615], V_record)
    plt.plot(time / time[615], np.full(len(time), -65),
             linestyle='--', color='gray', label='Resting')
    plt.title('Membrane Potential')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.grid()
    plt.legend()

    # Currents
    plt.subplot(2, 1, 1)
    plt.plot(time / time[615], I_inp, label='Input')
    plt.plot(time / time[615], I_out, label='Output')
    plt.title('Stimulus')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA/cm$^2$)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()