# Imports, as always...
from random import choice
from functools import reduce
from tqdm.notebook import tqdm
import numpy as np

from qiskit import QuantumCircuit, execute
from qiskit_aer import AerSimulator
from qiskit.quantum_info.states import Statevector


# Labels for the stabiliser states, as specified by Qiskit.
stabiliser_state_labels = ['0', '1', '+', '-', 'r', 'l']


# Function to randomly generate a single-qubit stabiliser state from the above set.
def generate_random_stab_1():
    return Statevector.from_label(choice(stabiliser_state_labels))


# Function to build a circuit implementing the experiment given in Huang et al. (2024)'s dataset procedure.
def build_experiment_circuit(U):
    # Instantiate a new circuit.
    circuit = QuantumCircuit(U.num_qubits)

    # We initialise each of the qubits to specify the input state (as a product of stable states).
    input_states = []
    for i in range(circuit.num_qubits):
        # Choose and keep track of the initialised state.
        init_state = generate_random_stab_1()
        input_states.append(init_state)

        # Prepare the qubit.
        circuit.initialize(init_state, i)

    # Barrier for clarity.
    circuit.barrier()

    # Create a copy of the given unitary so that we may add to the circuit.
    U_copy = U.copy()
    circuit = circuit.compose(U_copy)

    # Barrier for clarity.
    circuit.barrier()

    # For each qubit, we randomly select a Pauli basis to measure it in (to collapse them to stabiliser states).
    output_bases = []
    for i in range(circuit.num_qubits):
        # Choose a Pauli basis, and keep track of it.
        j = choice([0, 1, 2])
        output_bases.append(j)

        # Measuring in the X basis.
        if j == 1:
            circuit.h(i)
            continue

        # Measuring in the Y basis.
        elif j == 2:
            circuit.h(i), circuit.sdg(i)
            continue

        # Measuring in the Z basis -- no projection needed.
        else:
            continue

    # Measure all qubits. A barrier is placed automatically here.
    circuit.measure_all()

    # Return the circuit, the input states, and the output bases.
    return circuit, input_states, output_bases


# Function to translate the measurement outcome to the corresponding output states.
def measurements_to_state_vectors(output_bases, measurement_outcomes):
    output_states = []
    for basis, measurement in zip(output_bases, measurement_outcomes):
        # This makes assumptions about how we've been indexing to work.
        output_states.append(Statevector.from_label(stabiliser_state_labels[2 * basis + int(measurement)]))

    return output_states


# Helper function to combine state vectors into a single product state vector.
def big_tensor(state_vector_list):
    return reduce(lambda x, y : x ^ y, state_vector_list)


# The full dataset generation procedure.
def generate_dataset(U, N, product_states=True):
    # Instantiate a backend simulator and dataset list.
    backend = AerSimulator()
    dataset = []

    # Running the experiment N times to produce a dataset of N pairs.
    for _ in tqdm(range(N), desc='Generating dataset'):
        # Set up the experiment.
        circuit, input_states, output_bases = build_experiment_circuit(U)

        # Execute the circuit (once) and note the measurement outcome string.
        job = execute(circuit, backend=backend, shots=1)
        measurement_outcomes = list(job.result().get_counts(circuit).keys())[0]

        # Translate the measurement outcome string to the corresponding state vector.
        state_outcomes = measurements_to_state_vectors(output_bases, measurement_outcomes)

        if product_states:
            # Covert the state lists into product states.
            input_states = big_tensor(input_states)
            state_outcomes = big_tensor(state_outcomes)

        # Append to the dataset.
        dataset.append((input_states, state_outcomes))

    return dataset


# Reuse a given classical dataset to create 3n datasets to be used for learning approx. Heisenberg-evolved Pauli obs.
def expand_into_datasets(dataset):
    # Assumes the dataset does NOT consist of product states but instead arrays of qubits. It'll be easier this way.

    # Storing each dataset as a value in a dictionary, indexed by its Pauli basis P and qubit i (i.e. tuple indexing).
    datasets = {}

    # Let's have the incoming dataset as a numpy array.
    dataset = np.array(dataset)

    # Defining the Pauli bases.
    X, Y, Z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
    pauli_bases = [X, Y, Z]

    # Loop over all P and all i.
    for i in tqdm(range(dataset.shape[2]), desc='Expanding dataset'):
        for P in range(3):
            # Instantiate the new dataset array.
            datasets[(P, i)] = []

            # Loop over all output samples in the dataset.
            for phi_i in dataset[:, 1, i, :]:
                # Compute the new output as per Lemma 12 and add to the appropriate dataset.
                datasets[(P, i)].append(3 * (phi_i.T @ pauli_bases[P] @ phi_i))

    return datasets
