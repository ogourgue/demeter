"""Cellular Automaton to Telemac coupling functions.

Todo: Docstrings.

"""
import os
import numpy as np

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
        AGE_global_fn = './tmp_ca2tel/ca_age_global.txt'
        x_global_fn = './tmp_ca2tel/tel_x_global.txt'
        y_global_fn = './tmp_ca2tel/tel_y_global.txt'
        tri_global_fn = './tmp_ca2tel/tri_global.txt'
        age_global_fn = './tmp_ca2tel/tel_age_global.txt'

        # Save intermediate files.
        if not os.path.isfile(X_global_fn):
            np.savetxt(X_global_fn, X)
        if not os.path.isfile(Y_global_fn):
            np.savetxt(Y_global_fn, Y)
        np.savetxt(STATE_global_fn, STATE, fmt = '%d')
        np.savetxt(AGE_global_fn, AGE, fmt = '%d')
        if not os.path.isfile(x_global_fn):
            np.savetxt(x_global_fn, x)
        if not os.path.isfile(y_global_fn):
            np.savetxt(y_global_fn, y)
        if not os.path.isfile(tri_global_fn):
            np.savetxt(tri_global_fn, tri, fmt = '%d')

        # Run Cellular Automaton to Telemac voronoi coverage.
        os.system(launcher + ' -n %d python $DEMPATH/ca2tel_voronoi_age.py'
                  % nproc)

        # Load intermediate file.
        age = np.loadtxt(age_global_fn)

    return age

