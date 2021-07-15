"""Update cellular automaton state.

Todo: Docstrings.

"""
import numpy as np

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

    # Update state (0 -> 1).
    for i in range(nt):
        # Number of neighbors.
        nn = get_number_neighbors(state)
        # Update probability of expansion. Multiplied by .25 so that number-of-
        # neighbor contribution is 1 on average.
        p_exp_nn = p_exp * nn * .25
        # Update state.
        state = update_state(0, 1, state, 1 - (1 - p_est) * (1 - p_exp_nn))

    # Update state (1 -> 0).
    for i in range(nt):
        state = update_state(1, 0, state, p_die)

    return state

################################################################################
def get_number_neighbors(state):
    """Calculate number of vegetated cells among 8 neighboring cells.

    Args:
        state (NumPy array): Cellular automaton state.

    """

    # Initialize number of neighbors.
    nn = np.zeros(state.shape, dtype = int)

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






