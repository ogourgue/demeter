import numpy as np

################################################################################
class CellularAutomaton(object):
    """Process Cellular Automaton input/output.

    Attributes: Todo.

    """

    def __init__(self, x0, y0, nx, ny, dx):
        """Create a Cellular Automaton instance from grid parameters.

        Args:
            x0 (float): x-coordinate of the center of the lower-left grid cell.
            y0 (float): y-coordinate of the center of the lower-left grid cell.
            nx (int): Number of grid cells along x-axis.
            ny (int): Number of grid cells along y-axis.
            dx (float): Grid cell length.

        """
        # Grid parameters.
        self.x0 = x0
        self.y0 = y0
        self.nx = nx
        self.ny = ny
        self.dx = dx

        # Grid coordinates.
        self.x = np.linspace(x0, x0 + (nx - 1) * dx, nx)
        self.y = np.linspace(y0, y0 + (ny - 1) * dx, ny)

        # Cellular automaton state (empty).
        self.state = np.zeros((0, nx, ny))

        # Probability of establishment.
        self.p_est = np.zeros((nx, ny))

        # Lateral expansion rate.
        self.r_exp = np.zeros((nx, ny))

        # Probability of die-back.
        self.p_die = np.zeros((nx, ny))

    ############################################################################
    @classmethod
    def from_file(cls, filename):
        """Create a Cellular Automaton instance from an output file.

        Args:
            filename (str): Name of the output file to import.

        """
        # Todo.

    ############################################################################
    def append_state(self, state):
        """Append cellular automaton state.

        Args:
            state (NumPy array): Cellular automaton state to append.

        """
        # Append state.
        state = state.reshape((1, self.nx, self.ny))
        self.state = np.append(self.state, state, axis = 0)

    ############################################################################
    def update_probabilities(self, p_est, p_die, r_exp):
        """Update cellular automaton probabilities.

        Args:
            p_est (NumPy array or float): Probability of establishment.
            p_die (NumPy array or float): Probability of die-back.
            r_exp (NumPy array or float): Lateral expansion rate.

        """
        # Convert float to NumPy arrays.
        if type(p_est) in [float, int]:
            p_est = np.zeros((self.nx, self.ny)) + p_est
        if type(p_die) in [float, int]:
            p_die = np.zeros((self.nx, self.ny)) + p_die
        if type(r_exp) in [float, int]:
            r_exp = np.zeros((self.nx, self.ny)) + r_exp

        # Update class attributes.
        self.p_est = p_est
        self.p_die = p_die
        self.r_exp = r_exp

    ######################
    def run(self, nt, nproc = 1):
        """Run cellular automaton and update state.

        Args:
            nt (int): Number of iterations.
            nproc (int): Number of MPI processes.

        """
        # Class attributes.
        s0 = self.state[-1, :, :]
        p_est = self.p_est
        p_die = self.p_die
        r_exp = self.r_exp

        # Rescale lateral expansion rate (meter/yr -> grid cells per unit time).
        r_exp /= dx

        if nproc <= 1:

            ################
            # Serial mode. #
            ################

            # Call run_ca function.
            from demeter import cellular_automaton_run as ca_run
            s1 = ca_run.run(s0, p_est, p_die, r_exp, nt)

        else:

            #################
            # Parallel mode #
            #################

            print('Todo.')

        # Append new cellular automaton state.
        self.append_state(s1)



################################################################################
def number_iterations(r_exp, dx, n = 2):
    """Calculate the optimal number of iterations for lateral expansion.

    The optimal number of iterations is calculated so that the maximum expansion
    rate (i.e., theoretical expansion rate if all attempts are successful) is
    just lower than the average expansion rate (r_exp) plus a given number of
    times (n) its variance.

    Args:
        r_exp (float): Lateral expansion rate (unit length per unit time).
        dx (float): Grid cell size (unit length).
        n (int, optional): Parameter. Default to 2.
    """
    # Initialize number of iterations.
    nt = np.maximum(int(r_exp / dx), 1) + 1

    # Initialize probability of expansion.
    p_exp = r_exp / (nt * dx)

    # Initialize variance.
    var = nt * p_exp * (1 - p_exp) * dx

    # Initialize maximum expansion rate.
    r_max = nt * dx

    # Loop.
    while r_max < r_exp + n * var:
        nt += 1
        p_exp = r_exp / (nt * dx)
        var = nt * p_exp * (1 - p_exp) * dx
        r_max = nt * dx

    return nt






