"""Cellular Automaton to Telemac coupling functions.

Todo: Docstrings.

"""
import os
import shutil

import numpy as np

################################################################################
def voronoi_coverage(X, Y, STATE, x, y, tri, nproc = 1):
    """Calculate coverage over Voronoi neighborhoods.

    Args:
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        STATE (numPy array): Cellular Automaton state.
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        tri (NumPy array): Telemac grid connectivity table.
        nproc (int, optional): Number of MPI processes. Default to 1 (no MPI).

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
        if os.path.isdir('./tmp_ca2tel'):
            shutil.rmtree('./tmp_ca2tel')
        os.mkdir('./tmp_ca2tel')

        # Intermediate file names.
        X_global_fn = './tmp_ca2tel/X_global.txt'
        Y_global_fn = './tmp_ca2tel/Y_global.txt'
        STATE_global_fn = './tmp_ca2tel/STATE_global.txt'
        x_global_fn = './tmp_ca2tel/x_global.txt'
        y_global_fn = './tmp_ca2tel/y_global.txt'
        tri_global_fn = './tmp_ca2tel/tri_global.txt'
        cov_global_fn = './tmp_ca2tel/cov_global.txt'

        # Save intermediate files.
        np.savetxt(X_global_fn, X)
        np.savetxt(Y_global_fn, Y)
        np.savetxt(STATE_global_fn, STATE, fmt = '%d')
        np.savetxt(x_global_fn, x)
        np.savetxt(y_global_fn, y)
        np.savetxt(tri_global_fn, tri, fmt = '%d')

        # Run Cellular Automaton to Telemac voronoi coverage.
        os.system('mpiexec -n %d python $DEMPATH/ca2tel_voronoi_coverage.py'
                  % nproc)

        # Load intermediate file.
        cov = np.loadtxt(cov_global_fn)

        # Delete intermediate directory.
        shutil.rmtree('./tmp_ca2tel')

    return cov