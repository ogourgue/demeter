"""Calculate coverage from Cellular Automaton to Telemac.

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
def voronoi_coverage(X, Y, STATE, x, y, tri):
    """Calculate coverage over Voronoi neighborhoods.

    Args:
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        STATE (numPy array): Cellular Automaton state.
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        tri (NumPy array): Telemac grid connectivity table.

    """
    # Voronoi array.
    vor = voronoi(X, Y, x, y, tri)

    # Coverage.
    cov = np.zeros(x.shape)
    for i in range(x.shape[0]):
        cov[i] = np.mean(STATE[vor == i])

    return cov

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
"""if __name__ == '__main__':

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
        np.savetxt(F_global_fn, F)"""