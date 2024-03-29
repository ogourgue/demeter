"""Interpolation from Telemac to Cellular Automaton.

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

################################################################################
if __name__ == '__main__':

    # Global intermediate file names.
    x_global_fn = './tmp_tel2ca/x_global.txt'
    y_global_fn = './tmp_tel2ca/y_global.txt'
    f_global_fn = './tmp_tel2ca/f_global.txt'
    tri_global_fn = './tmp_tel2ca/tri_global.txt'
    X_global_fn = './tmp_tel2ca/X_global.txt'
    Y_global_fn = './tmp_tel2ca/Y_global.txt'
    F_global_fn = './tmp_tel2ca/F_global.txt'
    part_fn = './tmp_tel2ca/part.txt'
    empty_fn = './tmp_tel2ca/empty.txt'

    # Test if mesh partitioning files are already available.
    ready = False
    if os.path.isfile(part_fn):
        ready = True

    # Compute mesh partitioning by domain decomposition.
    if rank == 0 and not ready:

        # Load intermediate files.
        x = np.loadtxt(x_global_fn)
        y = np.loadtxt(y_global_fn)
        tri = np.loadtxt(tri_global_fn, dtype = int)
        X = np.loadtxt(X_global_fn)
        Y = np.loadtxt(Y_global_fn)

        # Cellular Automaton grid size.
        DX = X[1] - X[0]

        # Maximum triangle surface area (smax) and corresponding characteristic
        # length (lmax).
        x0 = x[tri[:, 0]]
        x1 = x[tri[:, 1]]
        x2 = x[tri[:, 2]]
        y0 = y[tri[:, 0]]
        y1 = y[tri[:, 1]]
        y2 = y[tri[:, 2]]
        smax = np.max(.5 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)))
        lmax = np.sqrt(2 * smax)

        # Domain decomposition (Cellular Automaton grid). Convert X, Y to 2D
        # arrays before splitting, then convert split arrays to 1D arrays.
        X, Y = np.meshgrid(X, Y, indexing = 'ij')
        nx, ny = X.shape
        mpi_nx, mpi_ny = ca_mpi.get_domain_decomposition(nx, ny, nproc)
        X_list = ca_mpi.mpi_split_array(X, mpi_nx, mpi_ny, contiguous = False)
        Y_list = ca_mpi.mpi_split_array(Y, mpi_nx, mpi_ny, contiguous = False)
        for i in range(nproc):
            X_list[i] = X_list[i][:, 0]
            Y_list[i] = Y_list[i][0, :]

        # Domain decomposition (Telemac grid). Take all triangles with at least
        # one vertex inside the Cellular Automaton sub-domain.
        x_list = []
        y_list = []
        tri_list = []
        glo_list = []
        for i in range(nproc):
            # Cellular Automaton sub-domain bounding box. Extend by lmax to
            # include triangles at sub-domain corners that crosses the
            # sub-domain, but with no vertex inside.
            X0 = X_list[i][0] - .5 * DX - lmax
            X1 = X_list[i][-1] + .5 * DX + lmax
            Y0 = Y_list[i][0] - .5 * DX - lmax
            Y1 = Y_list[i][-1] + .5 * DX + lmax
            # Global indices of nodes inside the Cellular Automaton sub-domain.
            nod_glo = np.argwhere((x >= X0) * (x <= X1) * (y >= Y0) * (y <= Y1))
            nod_glo = nod_glo.reshape(-1)
            # Global indices of triangles with at least one vertex inside the
            # Cellular Automaton sub-domain.
            tmp = np.zeros(tri.shape[0], dtype = bool)
            for j in range(3):
                tmp[np.isin(tri[:, j], nod_glo)] = True
            tri_glo = tri[tmp, :]
            # Local nodes with global indices (including some outside the
            # Cellular Automaton sub-domain but that are vertices of triangles
            # partly inside).
            nod_glo = np.unique(tri_glo)
            # Local node coordinates and field values.
            x_list.append(x[nod_glo])
            y_list.append(y[nod_glo])
            # Local connectivity table with local indices.
            tri_loc = np.zeros(tri_glo.shape, dtype = int)
            for j in range(tri_loc.shape[0]):
                for k in range(tri_loc.shape[1]):
                    tri_loc[j, k] = int(np.argwhere(tri_glo[j, k] == nod_glo))
            tri_list.append(tri_loc)
            # Local global node indices.
            glo_list.append(nod_glo)

        # Save local intermediate files.
        for i in range(nproc):

            # Local intermediate file names.
            x_local_fn = './tmp_tel2ca/x_local_%d.txt' % i
            y_local_fn = './tmp_tel2ca/y_local_%d.txt' % i
            tri_local_fn = './tmp_tel2ca/tri_local_%d.txt' % i
            X_local_fn = './tmp_tel2ca/X_local_%d.txt' % i
            Y_local_fn = './tmp_tel2ca/Y_local_%d.txt' % i
            glo_local_fn = './tmp_tel2ca/glo_local_%d.txt' % i

            # Save local intermediate files.
            np.savetxt(x_local_fn, x_list[i])
            np.savetxt(y_local_fn, y_list[i])
            np.savetxt(tri_local_fn, tri_list[i], fmt = '%d')
            np.savetxt(X_local_fn, X_list[i])
            np.savetxt(Y_local_fn, Y_list[i])
            np.savetxt(glo_local_fn, glo_list[i], fmt = '%d')

        # Save partition file.
        np.savetxt(part_fn, np.array([mpi_nx, mpi_ny]), fmt = '%d')

        # Save empty sub-domain file.
        empty = np.zeros(nproc, dtype = bool)
        for i in range(nproc):
            empty[i] = (len(x_list[i]) == 0)
        np.savetxt(empty_fn, empty, fmt = '%d')

    # Primary processor decomposes input variable for each partition and sends
    # partitions to secondary processors.
    if rank == 0:

        # Load global input variable.
        f = np.loadtxt(f_global_fn)

        # Load empty sub-domain file.
        empty = np.loadtxt(empty_fn, dtype = bool)

        # Load local global node index files.
        glo_list = []
        for i in range(nproc):
            glo_local_fn = './tmp_tel2ca/glo_local_%d.txt' % i
            if not empty[i]:
                glo_list.append(np.loadtxt(glo_local_fn, dtype = int))
            else:
                glo_list.append(np.array([], dtype = int))

        # Input variable for each partition.
        f_list = []
        for i in range(nproc):
            f_list.append(f[glo_list[i]])

        # Primary processor partition.
        f_loc = f_list[0]

        # Send partitions to secondary processors.
        for i in range(1, nproc):
            npoin_loc = f_list[i].shape[0]
            comm.send(npoin_loc, dest = i, tag = 500)
            comm.Send([f_list[i], MPI.FLOAT], dest = i, tag = 501)

    # Secondary processors receive input variable partitions from primary
    # processor.
    if rank > 0:

        # Receive partition from primary processor.
        npoin_loc = comm.recv(source = 0, tag = 500)
        f_loc = np.empty(npoin_loc, dtype = float)
        comm.Recv([f_loc, MPI.FLOAT], source = 0, tag = 501)

    # Local intermediate file names.
    x_local_fn = './tmp_tel2ca/x_local_%d.txt' % rank
    y_local_fn = './tmp_tel2ca/y_local_%d.txt' % rank
    tri_local_fn = './tmp_tel2ca/tri_local_%d.txt' % rank
    X_local_fn = './tmp_tel2ca/X_local_%d.txt' % rank
    Y_local_fn = './tmp_tel2ca/Y_local_%d.txt' % rank

    # Load local mesh partition.
    if f_loc.shape[0] > 0:
        x_loc = np.loadtxt(x_local_fn)
        y_loc = np.loadtxt(y_local_fn)
        tri_loc = np.loadtxt(tri_local_fn, dtype = int)
    else:
        x_loc = np.array([])
        y_loc = np.array([])
        tri_loc = np.array([], dtype = int)
    X_loc = np.loadtxt(X_local_fn)
    Y_loc = np.loadtxt(Y_local_fn)

    # Interpolation.
    F_loc = interpolation(x_loc, y_loc, f_loc, tri_loc, X_loc, Y_loc)

    # Secondary processors send output variable partitions to primary processor.
    if rank > 0:

        # Send partition to primary processor.
        nx_loc = F_loc.shape[0]
        ny_loc = F_loc.shape[1]
        comm.send(nx_loc, dest = 0, tag = 502)
        comm.send(ny_loc, dest = 0, tag = 503)
        comm.Send([F_loc, MPI.FLOAT], dest = 0, tag = 504)

    # Primary processor receives partitions from secondary processors and
    # reconstructs output variable.
    if rank == 0:

        # Primary processor partition.
        F_list = [F_loc]

        # Receive partitions from secondary processors.
        for i in range(1, nproc):
            nx_loc = comm.recv(source = i, tag = 502)
            ny_loc = comm.recv(source = i, tag = 503)
            F_list.append(np.empty((nx_loc, ny_loc), dtype = float))
            comm.Recv([F_list[i], MPI.FLOAT], source = i, tag = 504)

        # Load partition file.
        mpi_nx, mpi_ny = np.loadtxt(part_fn, dtype = int)

        # Reconstruction.
        F = ca_mpi.mpi_aggregate_array(F_list, mpi_nx, mpi_ny)

        # Save intermediate file.
        np.savetxt(F_global_fn, F)

    MPI.Finalize()
