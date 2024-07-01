# Imports, as always...
from random import choice
import numpy as np

# Quantum circuitry.
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.quantum_info.random import random_statevector

# Random seed setting.
np.random.seed(42)


def add_random_layer(circuit, seed=None):
    """
    Randomly add (at most) one layer to the given circuit.

    :param circuit: A Qiskit QuantumCircuit object to which new elements are to be added.
    :param seed: Optional seed for randomness control.
    :return: A Qiskit QuantumCircuit object with the new element(s) added.
    """

    # Resetting the seed if asked.
    if seed: np.random.seed(seed)

    # Whether to use two-qubit gates.
    if np.random.uniform() < .5:
        # Choose two random qubits and do the thing.
        control, target = np.random.choice(range(circuit.num_qubits), 2, replace=False)
        circuit.cx(control, target)

    # Otherwise, use single-qubit gates.
    else:
        # Consider each qubit independently.
        for qubit in range(circuit.num_qubits):
            # Whether to add a gate.
            if np.random.uniform() < .5:
                # Pick a gate.
                gate_choice = choice(range(4))

                # Apply it.
                if gate_choice == 0:
                    circuit.x(qubit)
                elif gate_choice == 1:
                    circuit.z(qubit)
                elif gate_choice == 2:
                    circuit.h(qubit)
                elif gate_choice == 3:
                    circuit.s(qubit)

            # Otherwise, move on.
            else:
                continue

    # Let's have it back too -- why not.
    return circuit


def build_circuit(n, d, seed=None):
    """
    Build an n-qubit circuit with randomly generated layers to a given depth d.

    :param n: Number of qubits.
    :param d: Depth of the circuit.
    :param seed: Optional seed for randomness control.
    :return: Qiskit QuantumCircuit object.
    """

    if seed: np.random.seed(seed)

    # Initialise the circuit.
    circuit = QuantumCircuit(n)

    # Add layers until we reach the target depth.
    while circuit.depth() < d:
        circuit = add_random_layer(circuit)

    return circuit


def estimate_output_distribution(circuit, input_repetitions, shots=None):
    """
    Estimate the output distribution for the given circuit over Haar randomly distributed input.

    :param circuit: The Qiskit QuantumCircuit object to estimate the output distribution.
    :param input_repetitions: Number of input states to try.
    :param shots: Number of times to run the circuit on each input to estimate the probability distribution over output states. If None, then probabilities are calculated outright.
    :return: NumPy Array object indicating the probability to sample each output state over the sampled input states (mean average).
    """

    probs = []

    for _ in range(input_repetitions):
        # Create a copy of the circuit that we can run and measure.
        dummy = QuantumCircuit(circuit.num_qubits)

        # Initialise.
        random_input = random_statevector(2 ** circuit.num_qubits)
        dummy.initialize(random_input)

        # Tag the circuit onto the end, and measure.
        dummy = dummy.compose(circuit.measure_all(inplace=False))

        # Use the Sampler primitive to compute the probability distribution over the output states.
        sampler = Sampler()
        job = sampler.run(circuits=dummy, shots=shots)
        probs.append(list(job.result().quasi_dists[0].binary_probabilities().values()))

    return np.mean(np.array(probs), axis=0)