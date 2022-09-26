"""Cellular Automaton to Telemac coupling functions.

Todo: Docstrings.

"""
import os
import numpy as np

################################################################################
def voronoi_coverage(X, Y, STATE, x, y, tri, nproc = 1, launcher = 'mpiexec'):
    """Calculate coverage over Voronoi neighborhoods.

    Args:
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        STATE (numPy array): Cellular Automaton state.
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        tri (NumPy array): Telemac grid connectivity table.
        nproc (int, optional): Number of MPI processes. Default to 1 (no MPI).
        launcher (str, optional): MPI launcher. Default to 'mpiexec'.

    """
    if nproc <= 1:

        ################
        # Serial mode. #
        ################

        # Call voronoi coverage function.
        from demeter import ca2tel_voronoi_coverage
        cov = ca2tel_voronoi_coverage.voronoi_coverage(X, Y, STATE, x, y, tri)

    else:

        #################
        # Parallel mode #
        #################

        # Create directory to store intermediate input files.
        if not os.path.isdir('./tmp_ca2tel'):
            os.mkdir('./tmp_ca2tel')

        # Intermediate file names.
        X_global_fn = './tmp_ca2tel/ca_x_global.txt'
        Y_global_fn = './tmp_ca2tel/ca_y_global.txt'
        STATE_global_fn = './tmp_ca2tel/state_global.txt'
        x_global_fn = './tmp_ca2tel/tel_x_global.txt'
        y_global_fn = './tmp_ca2tel/tel_y_global.txt'
        tri_global_fn = './tmp_ca2tel/tri_global.txt'
        cov_global_fn = './tmp_ca2tel/cov_global.txt'

        # Save intermediate files.
        if not os.path.isfile(X_global_fn):
            np.savetxt(X_global_fn, X)
        if not os.path.isfile(Y_global_fn):
            np.savetxt(Y_global_fn, Y)
        np.savetxt(STATE_global_fn, STATE, fmt = '%d')
        if not os.path.isfile(x_global_fn):
            np.savetxt(x_global_fn, x)
        if not os.path.isfile(y_global_fn):
            np.savetxt(y_global_fn, y)
        if not os.path.isfile(tri_global_fn):
            np.savetxt(tri_global_fn, tri, fmt = '%d')

        # Run Cellular Automaton to Telemac voronoi coverage.
        os.system(launcher + ' -n %d python $DEMPATH/ca2tel_voronoi_coverage.py'
                  % nproc)

        # Load intermediate file.
        cov = np.loadtxt(cov_global_fn)

    return cov

################################################################################
def voronoi_age(X, Y, STATE, AGE, x, y, tri, nproc = 1, launcher = 'mpiexec'):
    """Calculate mean age of vegetated cells over Voronoi neighborhoods.

    Args:
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        STATE (numPy array): Cellular Automaton state.
        AGE (numPy array): Cellular Automaton age.
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        tri (NumPy array): Telemac grid connectivity table.
        nproc (int, optional): Number of MPI processes. Default to 1 (no MPI).
        launcher (str, optional): MPI launcher. Default to 'mpiexec'.

    """
    if nproc <= 1:

        ################
        # Serial mode. #
        ################

        # Call voronoi age function.
        from demeter import ca2tel_voronoi_age
        age = ca2tel_voronoi_age.voronoi_age(X, Y, STATE, AGE, x, y, tri)

    return age

