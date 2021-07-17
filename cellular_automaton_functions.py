"""Set of probability functions.

Todo: Docstrings.

"""
import numpy as np

################################################################################
def step_tanh(x, x50, x95):
    """Calculate a smooth step function with a hyperbolic tangent.

    Args:
        x (NumPy array): Variable for which a step function is computed.
        x50 (float): Pivot value for which the step function is 0.5.
        x95 (float): Value for which the step function is .95 (used to determine
            the steepness of the step function).

    """
    # Parameters.
    a = x50
    b = (x95 - x50) / np.arctanh(.9)

    return .5 + .5 * np.tanh((x - a) / b)
