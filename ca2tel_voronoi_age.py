"""Calculate coverage from Cellular Automaton to Telemac.

Todo: Docstrings.

"""
import os

import numpy as np
from mpi4py import MPI

from demeter import cellular_automaton_mpi as ca_mpi

# Base class of MPI communicators.
comm = MPI.COMM_WORLD

# Number of MPI processes.
nproc = comm.Get_size()

# Rank of mpi process.
rank = comm.Get_rank()

################################################################################
def voronoi_age(X, Y, STATE, AGE, x, y, tri):
    """Calculate coverage over Voronoi neighborhoods.

    Args:
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        STATE (numPy array): Cellular Automaton state.
        AGE (numPy array): Cellular Automaton age.
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        tri (NumPy array): Telemac grid connectivity table.

    """
    # Voronoi array.
    vor = voronoi(X, Y, x, y, tri)

    # Age.
    age = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if np.sum(vor == i) == 0:
            age[i] = np.nan
        else:
            age[i] = np.mean(AGE[np.logical_and(vor == i, STATE > 0)])

    return age

################################################################################
def voronoi(X, Y, x, y, tri):
    """Calculate Voronoi neighborhood from Cellular Automaton and Telemac grids.

    Args:
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        tri (NumPy array): Telemac grid connectivity table.

    """
    # Initialize Voronoi array.
    vor = np.zeros((X.shape[0], Y.shape[0]), dtype = int) - 1

    # Loop over Telemac grid triangles;
    for i in range(tri.shape[0]):

        # triangle vertex coordinates.
        x0 = x[tri[i, 0]]
        x1 = x[tri[i, 1]]
        x2 = x[tri[i, 2]]
        y0 = y[tri[i, 0]]
        y1 = y[tri[i, 1]]
        y2 = y[tri[i, 2]]

        # Triangle bounding box and corresponding indices on Telemac grid.
        xmin = min([x0, x1, x2])
        xmax = max([x0, x1, x2])
        ymin = min([y0, y1, y2])
        ymax = max([y0, y1, y2])
        try:imin = int(np.argwhere(X <= xmin)[-1])
        except:imin = 0
        try:imax = int(np.argwhere(X >= xmax)[0])
        except:imax = len(X) - 1
        try:jmin = int(np.argwhere(Y <= ymin)[-1])
        except:jmin = 0
        try:jmax = int(np.argwhere(Y >= ymax)[0])
        except:jmax = len(Y) - 1

        # Local grid of the bounding box.
        X_loc, Y_loc = np.meshgrid(X[imin:imax + 1], Y[jmin:jmax + 1],
                                   indexing = 'ij')

        # Compute barycentric coordinates.
        s0 = ((y1 - y2) * (X_loc - x2) + (x2 - x1) * (Y_loc - y2)) / \
             ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
        s1 = ((y2 - y0) * (X_loc - x2) + (x0 - x2) * (Y_loc - y2)) / \
             ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
        s2 = 1 - s0 - s1

        # The entries of the array s below are True if all barycentric
        # coordinates are positive, which means that the corresponding Cellular
        # Automaton grid cells are inside the Telemac triangle.
        s = (s0 >= 0.) * (s1 >= 0.) * (s2 >= 0.)

        # Distance to triangle vertices.
        d = [(x0 - X_loc) * (x0 - X_loc) + (y0 - Y_loc) * (y0 - Y_loc),
             (x1 - X_loc) * (x1 - X_loc) + (y1 - Y_loc) * (y1 - Y_loc),
             (x2 - X_loc) * (x2 - X_loc) + (y2 - Y_loc) * (y2 - Y_loc)]
        d = np.array(d)

        # If inside the triangle, the entries of the array tmp below are the
        # indices of the closest vertex. If outside, the entries are the values
        # of the Voronoi array.
        tmp = tri[i, np.argmin(d, 0)]
        vor_loc = vor[imin:imax + 1, jmin:jmax + 1]
        tmp[s == False] = vor_loc[s == False]

        # Update Voronoi array for Cellular Automaton grid cells inside the
        # triangle.
        vor[imin:imax + 1, jmin:jmax + 1] = tmp

    return vor