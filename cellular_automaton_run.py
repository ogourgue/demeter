"""Update cellular automaton state.

Todo: Docstrings.

"""
import sys

import numpy as np
from mpi4py import MPI

# Base class of MPI communicators.
comm = MPI.COMM_WORLD

# Number of MPI processes.
nproc = comm.Get_size()

# Rank of mpi process.
rank = comm.Get_rank()

################################################################################
def run(state, p_est, p_die, r_exp, nt):
    """Update cellular automaton state.

    Args:
        state (NumPy array): Initial cellular automaton state.
        p_est (NumPy array): Probability of establishment.
        p_die (NumPy array): Probability of die-back.
        r_exp (NumPy array): Lateral expansion rate (number of grid cells per run).
        nt (int): Number of iterations.

    """
    # Rescale probabilities for each iteration.
    p_est = 1 - (1 - p_est) ** (1 / nt)
    p_die = 1 - (1 - p_die) ** (1 / nt)
    p_exp = r_exp / nt

    # Note: in Demeter 1.0, we were doing two separate loops for (i)
    # establishment and expansion, and (ii) for die-back. This required to limit
    # establishment and expansion by potential die-back. We integrate all
    # processes in one loop here, but it needs to be tested.

    # Loop over iterations.
    for i in range(nt):
        # Number of neighbors.
        nn = get_number_neighbors(state)
        # Update probability of expansion. Multiplied by .25 so that number-of-
        # neighbor contribution is 1 on average.
        p_exp_nn = p_exp * nn * .25
        # Establishment and expansion.
        state = update_state(0, 1, state, 1 - (1 - p_est) * (1 - p_exp_nn))
        # Die-back.
        state = update_state(1, 0, state, p_die)

    return state

################################################################################
def get_number_neighbors(state):
    """Calculate number of vegetated cells among 8 neighboring cells.

    Args:
        state (NumPy array): Cellular automaton state.

    """

    # Initialize number of neighbors.
    nn = np.zeros(state.shape, dtype = np.int8)

    # Calculate number of neighbors.
    nn[:, :-1] += (state[:, 1:] == 1) # north
    nn[:-1, :-1] += (state[1:, 1:]  == 1) # north-east
    nn[:-1, :] += (state[1:, :] == 1) # east
    nn[:-1, 1:] += (state[1:, :-1] == 1) # south-east
    nn[:, 1:] += (state[:, :-1] == 1) # south
    nn[1:, 1:] += (state[:-1, :-1] == 1) # south-west
    nn[1:, :] += (state[:-1, :] == 1) # west
    nn[1:, :-1] += (state[:-1, 1:]  == 1) # north-west

    return nn

################################################################################
def update_state(i, j, state, p):
    """Update cellular automaton state for transition i to j based on
    probability p.

    Args:
        i (int): State before transition.
        j (int): State after transition.
        state (NumPy array): Cellular automaton state before transition.
        p (NumPy array): Probability to transition from state i to j.

    """
    # Indices where cellular automaton state is i.
    ind = (state == i)

    # Generate random numbers for cells where cellular automaton state is i.
    test = np.array(np.random.rand(np.sum(ind)))

    # Test probabilities where cellular automaton state is i.
    tmp = state[ind].copy()
    tmp[p[ind] > test] = j

    # Update cellular automaton state.
    state[ind] = tmp

    return state

################################################################################
if __name__ == '__main__':

    # Parameter input.
    nt = int(sys.argv[1])

    # Mesh partitioning by domain decomposition.
    if rank == 0:

        # Intermediate file names.
        state_0_global_fn = './tmp_cellular_automaton/state_0_global.txt'
        state_1_global_fn = './tmp_cellular_automaton/state_1_global.txt'
        p_est_global_fn = './tmp_cellular_automaton/p_est_global.txt'
        p_die_global_fn = './tmp_cellular_automaton/p_die_global.txt'
        r_exp_global_fn = './tmp_cellular_automaton/r_exp_global.txt'

        # Load intermediate files.
        state_0 = np.loadtxt(state_0_global_fn)
        p_est = np.loadtxt(p_est_global_fn)
        p_die = np.loadtxt(p_die_global_fn)
        r_exp = np.loadtxt(r_exp_global_fn)

        # Domain decomposition.
        # KISS: Divide in vertical strips with np.array_split.
        # ...

    # Global mesh reconstruction for primary processor.
    if rank == 0:

        # ...

        # Save intermediate file.
        np.savetxt(state_1_global_fn, state_0)

