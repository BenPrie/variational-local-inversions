# Imports, as always...
import time
from tqdm.notebook import tqdm
import numpy as np
from os import makedirs, path
import csv

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, ParameterVector
from qiskit.circuit.library import TwoLocal

# Classical ML.
import torch
from torch.nn import Module
from torch.optim import Adam
from torch import manual_seed

# Quantum ML.
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA, ADAM, GradientDescent

# Scripts.
from scripts.utils import sample_n_states

# Plotting.
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns

# Styling.
sns.set_context('paper')
sns.set_theme(style='darkgrid', palette='Dark2')
palette = sns.color_palette('Dark2', n_colors=2)

# Device management.
if torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')


class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x):
        # Mean probability to measure anything other than 0...0 (i.e. prob. for failure outcome).
        # Single-size batches don't have two dimensions, so they need special treatment.
        if x.dim() == 1: return 1 - x[0]
        else:            return 1- torch.mean(x[:, 0])


def set_seed(seed):
    if seed is None: return

    algorithm_globals.random_seed = seed
    np.random.seed(seed)
    manual_seed(seed)


def plot_loss_curves(train_losses, val_losses, n_epochs):
    # Clear previous plots.
    clear_output(wait=True)

    # Plot losses.
    plt.plot(range(len(train_losses)), train_losses, color=palette[0], label='train')
    plt.plot(range(len(val_losses)), val_losses, color=palette[1], label='val')

    # Beautification.
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(-.05, .45)
    plt.xlim(0, n_epochs)
    plt.legend()

    plt.show()


def build_qnn_circuit(U, ansatz_reps, target_qubits, entanglement_method='full', multi_swap=False, seed=42):

    # Reset the seed for RNG.
    set_seed(seed)

    # Instantiate a new circuit with classical register for measurement.
    q_reg = QuantumRegister(U.num_qubits + len(target_qubits) + len(target_qubits) ** multi_swap)
    c_reg = ClassicalRegister(len(target_qubits) ** multi_swap)
    circuit = QuantumCircuit(q_reg, c_reg)

    # Prepare an input state (parameterised) consistent between the two sets of qubits.
    input_parameters = ParameterVector('in', 2 * U.num_qubits)
    for i in range(U.num_qubits):
        circuit.r(theta=input_parameters[2 * i], phi=input_parameters[2 * i + 1], qubit=i)
        if i in target_qubits: circuit.r(theta=input_parameters[2 * i], phi=input_parameters[2 * i + 1],
                                         qubit=U.num_qubits + i - 1)

    # Barrier for clarity.
    circuit.barrier()

    # Add U.
    U_copy = U.copy()
    circuit.compose(U_copy, inplace=True)

    # Barrier for clarity.
    circuit.barrier()

    # Then add the ansatz.
    ansatz = TwoLocal(
        num_qubits=U.num_qubits,
        rotation_blocks=['rx', 'ry'],
        entanglement_blocks='cx',
        entanglement=entanglement_method,
        reps=ansatz_reps,
        name='Ansatz'
    )
    circuit.compose(ansatz, inplace=True)

    # Barrier for clarity.
    circuit.barrier()

    # Define the auxiliary qubit index(/ices) and prepare them.
    aux_idx = list(range(circuit.num_qubits - len(target_qubits) ** multi_swap, circuit.num_qubits))
    circuit.h(aux_idx)

    # Global cost function.
    if multi_swap:
        circuit.cswap(control_qubit=aux_idx, target_qubit1=target_qubits,
                      target_qubit2=np.array(target_qubits) + (U.num_qubits - 1))

    # Local cost function.
    else:
        for i, q in enumerate(target_qubits):
            circuit.cswap(control_qubit=aux_idx, target_qubit1=q, target_qubit2=U.num_qubits + i)

    # Measure.
    circuit.h(aux_idx)
    circuit.measure(aux_idx, c_reg)

    # Return the circuit, the input parameters, and the weight parameters.
    return circuit, input_parameters, ansatz.parameters


def train_by_scipy(qnn, xs_train, xs_val, n_epochs, initial_weights=None, method='cobyla', lr=1e-3, zero_threshold=1e-2, early_stop_window=100, early_stop_threshold=1e-3, tol=1e-3, stats_save_dir=None, live_plot=False, seed=42):

    # Reset seed for RNG.
    set_seed(seed)

    # Save directory.
    if stats_save_dir: makedirs(stats_save_dir, exist_ok=True)

    # Figure size.
    if live_plot: plt.rcParams['figure.figsize'] = (4, 3)

    # Initialise arrays for stats.
    train_losses = []
    val_losses = []
    runtimes = []

    # Loss function according to the SWAP test.
    def loss_fn(weights):
        # Start timing.
        start = time.time()

        # Compute loss over the train and validation sets.
        # Loss is 1 - P[Y=0] = 1/2 - ((inner prod.)/2)^2.
        train_loss = 1 - np.mean(qnn.forward(input_data=xs_train, weights=weights)[:, 0])
        val_loss = 1 - np.mean(qnn.forward(input_data=xs_val, weights=weights)[:, 0])

        # Stop timing.
        end = time.time()
        runtime = end - start

        # Remember the stats.
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        runtimes.append(runtime)

        # Plotting.
        if live_plot: plot_loss_curves(train_losses, val_losses, n_epochs)

        # Saving.
        if stats_save_dir:
            with open(path.join(stats_save_dir, 'stats.csv'), 'a') as save_file:
                writer = csv.writer(save_file, delimiter=',')
                writer.writerow({'Train loss': train_loss, 'Val loss': val_loss, 'Runtime': runtime})

        # Conditions for early stopping: close enough to zero, not enough improvement in a recent interval (over the best), or close enough to convergence.
        close_to_zero = val_loss < zero_threshold
        no_improvement = (len(val_losses) > early_stop_window) and ((abs(min(val_losses) - min(val_losses[-early_stop_window:]))) > early_stop_threshold)
        probably_converged = (len(val_losses) > early_stop_window) and max(val_losses[-early_stop_window:]) - min(val_losses[-early_stop_window:]) < early_stop_threshold
        if close_to_zero or no_improvement or probably_converged: raise StopIteration

        # Return train loss only.
        return train_loss

    # Optimiser and initial weights (random for now, but all zeros might be a good idea for degrees of freedom).
    # For the time being, only gradient-free methods (e.g. COBYLA) appear to be working properly.
    if method.lower() == 'adam': optimiser = ADAM(maxiter=n_epochs, lr=lr)
    elif method.lower() == 'gd': optimiser = GradientDescent(maxiter=n_epochs, learning_rate=lr)
    else:                        optimiser = COBYLA(maxiter=n_epochs, tol=tol)

    if initial_weights is None: initial_weights = algorithm_globals.random.random(qnn.num_weights)

    # Do the thing.
    try:
        optimiser.minimize(loss_fn, x0=initial_weights)
    except:
        pass

    # Return the stats.
    return {'train_loss': train_losses, 'val_loss': val_losses, 'runtime': runtimes}


def train_by_COBYLA_no_recylcing(qnn, batch_size, n_epochs, initial_weights=None, stats_save_dir=None, live_plot=False, seed=42):

    # Reset seed for RNG.
    set_seed(seed)

    # Save directory.
    if stats_save_dir: makedirs(stats_save_dir, exist_ok=True)

    # Figure size.
    if live_plot: plt.rcParams['figure.figsize'] = (4, 3)

    # Initialise arrays for stats.
    losses = []
    runtimes = []

    # Loss function according to the SWAP test.
    def loss_fn(weights):
        # Start timing.
        start = time.time()

        # Sample a batch of input states.
        xs_train = sample_n_states(n_states=batch_size, n_qubits=qnn.num_inputs//2)

        # Compute loss over the train and validation sets.
        loss = 1 - np.mean(qnn.forward(input_data=xs_train, weights=weights)[:, 0])

        # Stop timing.
        end = time.time()
        runtime = end - start

        # Remember the stats.
        losses.append(loss)
        runtimes.append(runtime)

        # Plotting.
        if live_plot: plot_loss_curves(losses, losses, n_epochs)

        # Saving.
        if stats_save_dir:
            with open(path.join(stats_save_dir, 'stats.csv'), 'a') as save_file:
                writer = csv.writer(save_file, delimiter=',')
                writer.writerow({'Train loss': loss, 'Runtime': runtime})

        # Return train loss only.
        return loss

    # Optimiser and initial weights (random for now, but all zeros might be a good idea for degrees of freedom).
    optimiser = COBYLA(maxiter=n_epochs)
    if initial_weights is None: initial_weights = algorithm_globals.random.random(qnn.num_weights)

    # Do the thing.
    optimiser.minimize(loss_fn, x0=initial_weights)

    # Return the stats.
    return {'loss': losses, 'runtime': runtimes}


# Training step (i.e. one epoch).
def train_step(model, loader, optimiser, criterion):
    # Use train mode.
    model.train()

    # Track the running loss.
    running_loss = .0

    for xs in tqdm(loader):
        xs = xs.to(device)

        # Zero gradients.
        optimiser.zero_grad()

        # Run and estimate outcome distribution.
        outcome_dist = model(xs)

        # Compute loss.
        loss = criterion(outcome_dist)
        running_loss += loss.item()

        # Compute gradients and adjust weights.
        loss.backward()
        optimiser.step()

    # Return average loss in each batch over the epoch.
    return running_loss / len(loader)


# Evaluation step.
def evaluate(model, loader, criterion):
    # Use evaluation mode.
    model.eval()

    # Track running loss.
    running_loss = .0

    # No gradient tracking in evaluation.
    with torch.no_grad():
        for xs in loader:
            xs = xs.to(device)

            # Run and estimate outcome distribution.
            outcome_dist = model(xs)

            # Compute loss.
            loss = criterion(outcome_dist)
            running_loss += loss.item()

    # Return average loss in each batch over the loader.
    return running_loss / len(loader)


def train_by_torch(model, train_loader, val_loader, n_epochs, lr=1e-3, live_plot=False, save_dir=None):

    # Save directory.
    if save_dir: makedirs(save_dir, exist_ok=True)

    # Initialise arrays for stats.
    train_losses = []
    val_losses = []

    # Optimiser and loss function.
    optimiser = Adam(params=model.parameters(), lr=lr)
    criterion = Loss()

    # Track the best validation loss to recover the best model weights post-training.
    best_val_loss = np.inf
    best_state = model.state_dict()

    for _ in tqdm(np.arange(1, n_epochs + 1), desc='Training'):
        # Start timing.
        start = time.time()

        # Train step and evaluation.
        train_loss = train_step(model, train_loader, optimiser, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        # Stop timing.
        end = time.time()
        runtime = end - start

        # Store the stats.
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update to the bests.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

            # Save model weights.
            if save_dir: torch.save(best_state, path.join(save_dir, 'model_state_dict.pt'))

        # Save train-time stats.
        if save_dir:
            with open(path.join(save_dir, 'stats.csv'), 'a') as save_file:
                writer = csv.writer(save_file, delimiter=',')
                writer.writerow({'Train loss': train_loss, 'Val loss': val_loss, 'Runtime': runtime})

        # Plotting.
        if live_plot: plot_loss_curves(train_losses, val_losses, n_epochs)

    return {'train_loss': train_losses, 'val_loss': val_losses}
