"""Interpolation from Telemac to Cellular Automaton.

Todo: Docstrings.

"""
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

    # Split domains as CA.
    # Takes all triangles with at least one vertex inside sub-domain.
    # Apply interpolation on sub-domains.
    # Done.

    # Mesh partitioning by domain decomposition.
    if rank == 0:

        # Intermediate file names.
        x_global_fn = './tmp_tel2ca/x_global.txt'
        y_global_fn = './tmp_tel2ca/y_global.txt'
        f_global_fn = './tmp_tel2ca/f_global.txt'
        tri_global_fn = './tmp_tel2ca/tri_global.txt'
        X_global_fn = './tmp_tel2ca/X_global.txt'
        Y_global_fn = './tmp_tel2ca/Y_global.txt'
        F_global_fn = './tmp_tel2ca/F_global.txt'

        # Load intermediate files.
        x = np.loadtxt(x_global_fn)
        y = np.loadtxt(y_global_fn)
        f = np.loadtxt(f_global_fn)
        tri = np.loadtxt(tri_global_fn, dtype = int)
        X = np.loadtxt(X_global_fn)
        Y = np.loadtxt(Y_global_fn)

        # Cellular Automaton grid size.
        DX = X[1] - X[0]

        # Domain decomposition (Cellular Automaton grid). Convert X, Y to 2D
        # arrays before splitting, then convert split arrays to contiguous
        # (required for MPI communication) 1D arrays.
        X, Y = np.meshgrid(X, Y, indexing = 'ij')
        nx, ny = X.shape
        mpi_nx, mpi_ny = ca_mpi.get_domain_decomposition(nx, ny, nproc)
        X_list = ca_mpi.mpi_split_array(X, mpi_nx, mpi_ny)
        Y_list = ca_mpi.mpi_split_array(Y, mpi_nx, mpi_ny)
        for i in range(nproc):
            X_list[i] = np.ascontiguousarray(X_list[i][:, 0])
            Y_list[i] = np.ascontiguousarray(Y_list[i][0, :])

        # Domain decomposition (Telemac grid). Take all triangles with at least
        # one vertex inside the Cellular Automaton sub-domain.
        x_list = []
        y_list = []
        f_list = []
        tri_list = []
        for i in range(nproc):
            # Cellular Automaton sub-domain bounding box.
            X0 = X_list[i][0] - DX
            X1 = X_list[i][-1] + DX
            Y0 = Y_list[i][0] - DX
            Y1 = Y_list[i][-1] + DX
            # Global indices of nodes inside the Cellular Automaton sub-domain.
            nod_glo = np.argwhere((x >= X0) * (x <= X1) * (y >= Y0) * (y <= Y1))
            nod_glo = nod_glo.reshape(-1)
            # Global indices of triangles with at least one vertex inside the
            # Cellular Automaton sub-domain.
            tmp = np.zeros(tri.shape[0], dtype = bool)
            for j in range(tri.shape[0]):
                tmp[j] = np.intersect1d(tri[j, :], nod_glo).shape[0] > 0
            tri_glo = tri[tmp, :]
            # Local nodes with global indices (including some outside the
            # Cellular Automaton sub-domain but that are vertices of triangles
            # partly inside).
            nod_glo = np.unique(tri_glo)
            # Local node coordinates and field values.
            x_list.append(x[nod_glo])
            y_list.append(y[nod_glo])
            f_list.append(f[nod_glo])
            # Local connectivity table with local indices.
            tri_loc = np.zeros(tri_glo.shape, dtype = int)
            for j in range(tri_loc.shape[0]):
                for k in range(tri_loc.shape[1]):
                    tri_loc[j, k] = int(np.argwhere(tri_glo[j, k] == nod_glo))
            tri_list.append(tri_loc)

    # Mesh partitioning for primary processor.
    if rank == 0:

        # Primary processor partition.
        x_loc = x_list[0]
        y_loc = y_list[0]
        f_loc = f_list[0]
        tri_loc = tri_list[0]
        X_loc = X_list[0]
        Y_loc = Y_list[0]

        # Send partition data to secondary processors.
        for i in range(1, nproc):
            npoin_loc = x_list[i].shape[0]
            nelem_loc = tri_list[i].shape[0]
            nx_loc = X_list[i].shape[0]
            ny_loc = Y_list[i].shape[0]
            comm.send(npoin_loc, dest = i, tag = 500)
            comm.send(nelem_loc, dest = i, tag = 501)
            comm.send(nx_loc, dest = i, tag = 502)
            comm.send(ny_loc, dest = i, tag = 503)
            comm.Send([x_list[i], MPI.FLOAT], dest = i, tag = 504)
            comm.Send([y_list[i], MPI.FLOAT], dest = i, tag = 505)
            comm.Send([f_list[i], MPI.FLOAT], dest = i, tag = 506)
            comm.Send([tri_list[i], MPI.INT], dest = i, tag = 507)
            comm.Send([X_list[i], MPI.FLOAT], dest = i, tag = 508)
            comm.Send([Y_list[i], MPI.FLOAT], dest = i, tag = 509)

    # Mesh partitioning for secondary processors.
    if rank > 0:

        # Receive partition data from primary processor.
        npoin_loc = comm.recv(source = 0, tag = 500)
        nelem_loc = comm.recv(source = 0, tag = 501)
        nx_loc = comm.recv(source = 0, tag = 502)
        ny_loc = comm.recv(source = 0, tag = 503)
        x_loc = np.empty(npoin_loc, dtype = float)
        y_loc = np.empty(npoin_loc, dtype = float)
        f_loc = np.empty(npoin_loc, dtype = float)
        tri_loc = np.empty((nelem_loc, 3), dtype = int)
        X_loc = np.empty(nx_loc, dtype = float)
        Y_loc = np.empty(ny_loc, dtype = float)
        comm.Recv([x_loc, MPI.FLOAT], source = 0, tag = 504)
        comm.Recv([y_loc, MPI.FLOAT], source = 0, tag = 505)
        comm.Recv([f_loc, MPI.FLOAT], source = 0, tag = 506)
        comm.Recv([tri_loc, MPI.INT], source = 0, tag = 507)
        comm.Recv([X_loc, MPI.FLOAT], source = 0, tag = 508)
        comm.Recv([Y_loc, MPI.FLOAT], source = 0, tag = 509)

    # Interpolate.
    F_loc = interpolation(x_loc, y_loc, f_loc, tri_loc, X_loc, Y_loc)

    # Global mesh reconstruction for secondary processors.
    if rank > 0:

        # Send partition data to primary processor.
        comm.Send([F_loc, MPI.FLOAT], dest = 0, tag = 510)

    # Global mesh reconstruction for primary processor.
    if rank == 0:

        # Primary processor partition.
        F_list = [F_loc]

        # Receive partition data from secondary processors.
        for i in range(1, nproc):
            nx_loc = X_list[i].shape[0]
            ny_loc = Y_list[i].shape[0]
            F_list.append(np.empty((nx_loc, ny_loc), dtype = float))
            comm.Recv([F_list[i], MPI.FLOAT], source = i, tag = 510)

        # Reconstruction.
        F = ca_mpi.mpi_aggregate_array(F_list, mpi_nx, mpi_ny)

        # Save intermediate file.
        np.savetxt(F_global_fn, F)