import os
import shutil

import numpy as np

################################################################################
class CellularAutomaton(object):
    """Process Cellular Automaton input/output.

    Attributes: Todo.

    """

    def __init__(self, x0, y0, nx, ny, dx, times = None, state = None,
                 with_age = False, age = None):
        """Create a cellular automaton instance from grid parameters.

        Args:
            x0 (float): x-coordinate of the center of the lower-left grid cell.
            y0 (float): y-coordinate of the center of the lower-left grid cell.
            nx (int): Number of grid cells along x-axis.
            ny (int): Number of grid cells along y-axis.
            dx (float): Grid cell length.
            times (NumPy array, optional): Time steps (yr). Default to None (no
                time step).
            state (NumPy array, optional): Cellular automaton state for each
                time step. Default to None (no time step).
            with_age (bool, optional): True if cellular automaton age must be
                computed. Default to False.
            age (NumPy array, optional): Cellular automaton age for each time
                step. Default to None (no time step).

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

        # Time.
        if times is None:
            self.times = []
        else:
            self.times = times

        # Cellular automaton state.
        if state is None:
            self.state = np.zeros((0, nx, ny), dtype = int)
        else:
            self.state = state

        # Cellular automaton age.
        if with_age:
            if age is None:
                self.age = np.zeros((0, nx, ny), dtype = int)
            else:
                self.age = age
        else:
            self.age = None

        # Probability of establishment.
        self.p_est = np.zeros((nx, ny))

        # Probability of die-back.
        self.p_die = np.zeros((nx, ny))

        # Lateral expansion rate.
        self.r_exp = np.zeros((nx, ny))

    ############################################################################
    @classmethod
    def from_file(cls, filename, filename_age = None, step = None):
        """Create a cellular automaton instance from an output file.

        Args:
            filename (str): Name of the output file to import.
            step (int, optional): Time step to import (-1 for last time step).
                Default to None (all time steps are imported).
            filename_age (str, optional): Name of the age output file to import.
                Default to None (age is not computed).

        """
        # Import file.
        data, times, state = import_state(filename, step, True, True)
        x0 = data[0]
        y0 = data[1]
        nx = data[2]
        ny = data[3]
        dx = data[4]

        # Grid coordinates.
        x = np.linspace(x0, x0 + (nx - 1) * dx, nx)
        y = np.linspace(y0, y0 + (ny - 1) * dx, ny)

        # Import age file, if needed.
        if filename_age is not None:
            with_age = True
            age = import_age(filename_age, step)

        else:
            with_age = False
            age = None

        return cls(x0, y0, nx, ny, dx, times, state, with_age, age)

    ############################################################################
    def export(self, filename, step = None):
        """Export cellular automaton state output file.

        Args:
            filename (str): Name of the file to export.
            step (int, optional): Time step to export (-1 for last time step).
                Default to None (all time steps are imported).

        """
        # Class attributes.
        x0 = self.x0
        y0 = self.y0
        nx = self.nx
        ny = self.ny
        dx = self.dx

        # Time steps.
        if step is None:
            steps = list(range(len(self.times)))
        else:
            steps = [step]

        # Time.
        times = []
        for step in steps:
            times.append(self.times[step])

        # Cellular automaton state.
        state = self.state[steps, :, :]

        # Export file.
        export_state(filename, x0, y0, nx, ny, dx, times, state)

    ############################################################################
    def export_age(self, filename, step = None):
        """Export cellular automaton age output file.

        Args:
            filename (str): Name of the file to export.
            step (int, optional): Time step to export (-1 for last time step).
                Default to None (all time steps are imported).

        """
        # Class attributes.
        x0 = self.x0
        y0 = self.y0
        nx = self.nx
        ny = self.ny
        dx = self.dx

        # Time steps.
        if step is None:
            steps = list(range(len(self.times)))
        else:
            steps = [step]

        # Time.
        times = []
        for step in steps:
            times.append(self.times[step])

        # Cellular automaton age.
        age = self.age[steps, :, :]

        # Export file.
        export_age(filename, x0, y0, nx, ny, dx, times, age)

    ############################################################################
    def append_times(self, time):
        """Append a time step to the cellular automaton instance.

        Args:
            time (float): Time step to append.

        """
        self.times.append(time)

    ############################################################################
    def append_state(self, state):
        """Append cellular automaton state.

        Args:
            state (NumPy array): Cellular automaton state to append.

        """
        # Reshape array to append.
        state = state.reshape((1, self.nx, self.ny))

        # Append state.
        self.state = np.append(self.state, state, axis = 0)

    ############################################################################
    def append_age(self, age):
        """Append cellular automaton age.

        Args:
            age (NumPy array): Cellular automaton age to append.

        """
        # Reshape array to append.
        age = age.reshape((1, self.nx, self.ny))

        # Append age.
        self.age = np.append(self.age, age, axis = 0)

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

    ############################################################################
    def run(self, nt, nproc = 1, launcher = 'mpiexec'):
        """Run cellular automaton and update state (and age, if computed).

        Args:
            nt (int): Number of iterations.
            nproc (int): Number of MPI processes.

        """
        # Class attributes.
        dx = self.dx
        state_0 = self.state[-1, :, :]
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
            state_1 = ca_run.run(state_0, p_est, p_die, r_exp, nt)

        else:

            ##################
            # Parallel mode. #
            ##################

            # Create directory to store intermediate input files.
            if os.path.isdir('./tmp_cellular_automaton'):
                shutil.rmtree('./tmp_cellular_automaton')
            os.mkdir('./tmp_cellular_automaton')

            # Intermediate file names.
            state_0_global_fn = './tmp_cellular_automaton/state_0_global.txt'
            state_1_global_fn = './tmp_cellular_automaton/state_1_global.txt'
            p_est_global_fn = './tmp_cellular_automaton/p_est_global.txt'
            p_die_global_fn = './tmp_cellular_automaton/p_die_global.txt'
            r_exp_global_fn = './tmp_cellular_automaton/r_exp_global.txt'

            # Save intermediate files.
            np.savetxt(state_0_global_fn, state_0, fmt = '%d')
            np.savetxt(p_est_global_fn, p_est)
            np.savetxt(p_die_global_fn, p_die)
            np.savetxt(r_exp_global_fn, r_exp)

            # Generate random seed (required for reproducibility).
            seed = np.random.randint(2 ** 32)

            # Run parallel Cellular Automaton run module.
            os.system(launcher + ' -n %d python ' % nproc +
                      '$DEMPATH/cellular_automaton_run.py %d %d' % (nt, seed))

            # Load intermediate file.
            state_1 = np.loadtxt(state_1_global_fn, dtype = int)

            # Delete intermediate directory.
            shutil.rmtree('./tmp_cellular_automaton')

        # Append new cellular automaton state.
        self.append_state(state_1)

        # Update cellular automaton age.
        if self.age is not None:
            age = self.age[-1, :, :]
            age[state_1 > 0] += 1

        # Append new cellular automator age.
        if self.age is not None:
            self.append_age(age)

    ############################################################################
    def remove_time_step(self, step = 0):
        """Remove one time step from state and times.

        Args:
            step (int, optional): Time step index to remove. Default to 0.

        """
        # Class attributes.
        times = self.times
        state = self.state
        age = self.age

        # Remove time step.
        times = times[:step] + times[step + 1:]
        state = np.concatenate((state[:step, :], state[step + 1:, :]), axis = 0)
        if age is not None:
            age = np.concatenate((age[:step, :], age[step + 1:, :]), axis = 0)

        # Update class attributes.
        self.times = times
        self.state = state
        self.age = age

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

################################################################################
def export_state(filename, x0, y0, nx, ny, dx, times, state):
    """Export cellular automaton state output file.

    Args:
        filename (str): Name of the file to export.
        x0 (float): x-coordinate of the center of the lower-left grid cell.
        y0 (float): y-coordinate of the center of the lower-left grid cell.
        nx (int): Number of grid cells along x-axis.
        ny (int): Number of grid cells along y-axis.
        dx (float): Grid cell length.
        times (NumPy array): Time steps (yr).
        state (NumPy array): Cellular automaton state.

    """
    # Open file.
    file = open(filename, 'w')

    # Export header.
    np.array(x0, dtype = float).tofile(file)
    np.array(y0, dtype = float).tofile(file)
    np.array(nx, dtype = int).tofile(file)
    np.array(ny, dtype = int).tofile(file)
    np.array(dx, dtype = float).tofile(file)
    np.array(len(times), dtype = int).tofile(file)

    # Export time.
    np.array(times, dtype = float).tofile(file)

    # Export data per time step.
    for i in range(len(times)):
        np.array(state[i, :, :], dtype = np.int8).tofile(file)

    # Close file.
    file.close()

################################################################################
def import_state(filename, step = None, with_header = False, with_time = False):
    """Import cellular automaton state output file.

    Args:
        filename (str): Name of the output file to import.
        step (int, optional): Time step to import (-1 for last time step).
            Default to None (all time steps are imported).
        with_header (bool, optional): Return header information if True. Default
            to False.
        with_header (bool, optional): Return time if True. Default to False.

    """
    # Open file.
    file = open(filename, 'rb')

    # Read header.
    x0 = np.fromfile(file, dtype = float, count = 1)[0]
    y0 = np.fromfile(file, dtype = float, count = 1)[0]
    nx = np.fromfile(file, dtype = int, count = 1)[0]
    ny = np.fromfile(file, dtype = int, count = 1)[0]
    dx = np.fromfile(file, dtype = float, count = 1)[0]
    nt = np.fromfile(file, dtype = int, count = 1)[0]

    # Read times.
    times = list(np.fromfile(file, dtype = float, count = nt))

    # Import all time steps.
    if step is None:
        state = np.fromfile(file, dtype = np.int8, count = nt * nx * ny)
        state = state.reshape((nt, nx, ny)).astype(int)

    # Import last time step.
    elif step == -1:
        times = [times[-1]]
        # Skip preceding time steps.
        file.seek(nx * ny * (nt - 1), 1)
        # Read data.
        state = np.fromfile(file, dtype = np.int8, count = nx * ny)
        state = state.reshape((1, nx, ny)).astype(int)

    # Import specific time step.
    else:
        times = [times[step]]
        # Skip preceding time steps.
        file.seek(nx * ny * step, 1)
        # Read data.
        state = np.fromfile(file, dtype = np.int8, count = nx * ny)
        state = state.reshape((1, nx, ny)).astype(int)

    # Close file.
    file.close()

    if with_header and with_time:
        return (x0, y0, nx, ny, dx, nt), times, state
    if with_header and not with_time:
        return (x0, y0, nx, ny, dx, nt), state
    if not with_header and with_time:
        return times, state
    if not with_header and not with_time:
        return state

################################################################################
def export_age(filename, x0, y0, nx, ny, dx, times, age):
    """Export cellular automaton age output file.

    Args:
        filename (str): Name of the file to export.
        x0 (float): x-coordinate of the center of the lower-left grid cell.
        y0 (float): y-coordinate of the center of the lower-left grid cell.
        nx (int): Number of grid cells along x-axis.
        ny (int): Number of grid cells along y-axis.
        dx (float): Grid cell length.
        times (NumPy array): Time steps (yr).
        age (NumPy array): Cellular automaton age (yr).

    """
    # Open file.
    file = open(filename, 'w')

    # Export header.
    np.array(x0, dtype = float).tofile(file)
    np.array(y0, dtype = float).tofile(file)
    np.array(nx, dtype = int).tofile(file)
    np.array(ny, dtype = int).tofile(file)
    np.array(dx, dtype = float).tofile(file)
    np.array(len(times), dtype = int).tofile(file)

    # Export time.
    np.array(times, dtype = float).tofile(file)

    # Export data per time step.
    for i in range(len(times)):
        np.array(age[i, :, :], dtype = np.uint16).tofile(file)

    # Close file.
    file.close()

################################################################################
def import_age(filename, step = None, with_header = False, with_time = False):
    """Import cellular automaton age output file.

    Args:
        filename (str): Name of the output file to import.
        step (int, optional): Time step to import (-1 for last time step).
            Default to None (all time steps are imported).
        with_header (bool, optional): Return header information if True. Default
            to False.
        with_header (bool, optional): Return time if True. Default to False.

    """
    # Open file.
    file = open(filename, 'rb')

    # Read header.
    x0 = np.fromfile(file, dtype = float, count = 1)[0]
    y0 = np.fromfile(file, dtype = float, count = 1)[0]
    nx = np.fromfile(file, dtype = int, count = 1)[0]
    ny = np.fromfile(file, dtype = int, count = 1)[0]
    dx = np.fromfile(file, dtype = float, count = 1)[0]
    nt = np.fromfile(file, dtype = int, count = 1)[0]

    # Read times.
    times = list(np.fromfile(file, dtype = float, count = nt))

    # Import all time steps.
    if step is None:
        age = np.fromfile(file, dtype = np.uint16, count = nt * nx * ny)
        age = age.reshape((nt, nx, ny)).astype(int)

    # Import last time step.
    elif step == -1:
        times = [times[-1]]
        # Skip preceding time steps.
        file.seek(nx * ny * (nt - 1) * 2, 1)
        # Read data.
        age = np.fromfile(file, dtype = np.uint16, count = nx * ny)
        age = age.reshape((1, nx, ny)).astype(int)

    # Import specific time step.
    else:
        times = [times[step]]
        # Skip preceding time steps.
        file.seek(nx * ny * step * 2, 1)
        # Read data.
        age = np.fromfile(file, dtype = np.uint16, count = nx * ny)
        age = age.reshape((1, nx, ny)).astype(int)

    # Close file.
    file.close()

    if with_header and with_time:
        return (x0, y0, nx, ny, dx, nt), times, age
    if with_header and not with_time:
        return (x0, y0, nx, ny, dx, nt), age
    if not with_header and with_time:
        return times, age
    if not with_header and not with_time:
        return age
