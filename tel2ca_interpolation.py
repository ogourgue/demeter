"""Interpolation from Telemac to Cellular Automaton.

Todo: Docstrings.

"""
import numpy as np
from mpi4py import MPI

# Base class of MPI communicators.
comm = MPI.COMM_WORLD

# Number of MPI processes.
nproc = comm.Get_size()

# Rank of mpi process.
rank = comm.Get_rank()

################################################################################
def interpolation(x, y, f, tri, X, Y):
    """Interpolate a Telemac variable on a Cellular Automaton grid.

    Args:
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        f (NumPy array): Telemac variable to interpolate.
        tri (NumPy array): Telemac grid connectivity table.
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).

    """
    # Initialize output.
    F = np.zeros((len(X), len(Y))) + np.nan

    # Loop over Telemac triangles.
    for i in range(tri.shape[0]):

        # Triangle vertex coordinates.
        x0 = x[tri[i, 0]]
        x1 = x[tri[i, 1]]
        x2 = x[tri[i, 2]]
        y0 = y[tri[i, 0]]
        y1 = y[tri[i, 1]]
        y2 = y[tri[i, 2]]

        # Variable on triangle vertex coordinates.
        f0 = f[tri[i, 0]]
        f1 = f[tri[i, 1]]
        f2 = f[tri[i, 2]]

        # Bounding box coordinates around the triangle (to limit the search for
        # Cellular Automaton grid cells within the triangle).
        xmin = np.min([x0, x1, x2])
        xmax = np.max([x0, x1, x2])
        ymin = np.min([y0, y1, y2])
        ymax = np.max([y0, y1, y2])

        # Indices of Cellular Automaton grid cells with central point inside the
        # bounding box.
        INDX = np.where((X > xmin) * (X < xmax))[0]
        INDY = np.where((Y > ymin) * (Y < ymax))[0]

        # Meshgrid on the bounding box.
        XLOC, YLOC = np.meshgrid(X[INDX], Y[INDY], indexing = 'ij')

        # Barycentric coordinates.
        S0 = ((y1 - y2) * (XLOC - x2) + (x2 - x1) * (YLOC - y2)) / \
             ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
        S1 = ((y2 - y0) * (XLOC - x2) + (x0 - x2) * (YLOC - y2)) / \
             ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
        S2 = 1 - S0 - S1

        # Barycentric interpolation.
        FLOC = f0 * S0 + f1 * S1 + f2 * S2

        # Update output array for Cellular Automaton grid cells with central
        # point inside the triangle (i.e., for which barycentric coordinates are
        # all positive).
        for j in range(len(INDX)):
            for k in range(len(INDY)):
                if S0[j, k] >= 0 and S1[j, k] >= 0 and S2[j, k] >= 0:
                    F[INDX[j], INDY[k]] = FLOC[j, k]

    return F