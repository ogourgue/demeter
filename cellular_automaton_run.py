"""Update cellular automaton state.

Todo: Docstrings.

"""
################################################################################
def run(s, p_est, p_die, r_exp, nt):
    """Update cellular automaton state.

    Args:
        s0 (NumPy array): Initial cellular automaton state.
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
        # Neighboring state.
        sn = get_neighboring_state(s0)
        # Update probability of expansion.
        p_exp *= sn
        # Combine probabilities.
        p = 1 - (1 - p_est) * (1 - p_exp)
        # Update state.
        s = update_state(0, 1, s, p)

    # Update state (1 -> 0).
    for i in range(nt):
        s = update_state(1, 0, s, p_die)

    return s

################################################################################
def get_neighboring_state(s0):

################################################################################
def update_state(s0, s1, s, p):
