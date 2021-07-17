"""Telemac to Cellular Automaton coupling functions.

Todo: Docstrings.

"""
import os
import shutil

import numpy as np

################################################################################
def interpolation(x, y, f, tri, X, Y, nproc = 1):
    """Interpolate a Telemac variable on a Cellular Automaton grid.

    Args:
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        f (NumPy array): Telemac variable to interpolate.
        tri (NumPy array): Telemac grid connectivity table.
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        nproc (int, optional): Number of MPI processes. Default to 1 (no MPI).

    """
    if nproc <= 1:

        ################
        # Serial mode. #
        ################

        # Call run_ca function.
        from demeter import tel2ca_interpolation
        F = tel2ca_interpolation.interpolation(x, y, f, tri, X, Y)

    else:

        #################
        # Parallel mode #
        #################

        # Create directory to store intermediate input files.
        if os.path.isdir('./tmp_tel2ca'):
            shutil.rmtree('./tmp_tel2ca')
        os.mkdir('./tmp_tel2ca')

        # Intermediate file names.
        x_global_fn = './tmp_tel2ca/x_global.txt'
        y_global_fn = './tmp_tel2ca/y_global.txt'
        f_global_fn = './tmp_tel2ca/f_global.txt'
        tri_global_fn = './tmp_tel2ca/tri_global.txt'
        X_global_fn = './tmp_tel2ca/X_global.txt'
        Y_global_fn = './tmp_tel2ca/Y_global.txt'
        F_global_fn = './tmp_tel2ca/F_global.txt'

        # Save intermediate files.
        np.savetxt(x_global_fn, x)
        np.savetxt(y_global_fn, y)
        np.savetxt(f_global_fn, f)
        np.savetxt(tri_global_fn, tri, fmt = '%d')
        np.savetxt(X_global_fn, X)
        np.savetxt(Y_global_fn, Y)

        # Run Telemac to Cellular Automaton interpolation module.
        os.system('mpiexec -n %d python $DEMPATH/tel2ca_interpolation.py'
                  % nproc)

        # Load intermediate file.
        F = np.loadtxt(F_global_fn)

        # Delete intermediate directory.
        shutil.rmtree('./tmp_tel2ca')

    return F