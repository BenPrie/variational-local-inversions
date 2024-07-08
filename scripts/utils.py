# Imports...
import numpy as np

# Classical ML.
from torch import manual_seed

# Quantum ML.
from qiskit_algorithms.utils import algorithm_globals
from qiskit.quantum_info.random import random_statevector


def set_seed(seed):
    if seed is None: return

    algorithm_globals.random_seed = seed
    np.random.seed(seed)
    manual_seed(seed)


def sample_haar_random_state_angles(seed=None):

    # Reset the seed for RNG if requested.
    if seed: set_seed(seed)

    # Sample a single-qubit state vector randomly (Haar distributed).
    state_vector = random_statevector(dims=2).data

    # Convert to Cartesian co-ordinates.
    x = 2 * np.real(state_vector[0] * np.conj(state_vector[1]))
    y = 2 * np.imag(state_vector[0] * np.conj(state_vector[1]))
    z = np.real(state_vector[0] * np.conj(state_vector[0]) - state_vector[1] * np.conj(state_vector[1]))

    # Convert to polar.
    theta = np.arctan2(y, x)
    phi = np.arccos(z)

    return np.array([theta, phi])


def sample_n_states(n_states, n_qubits, seed=None):

    # Reset the seed for RNG if requested.
    if seed: set_seed(seed)

    return np.array([
        np.array([sample_haar_random_state_angles() for _ in range(n_qubits)]).ravel() for _ in range(n_states)
    ])
