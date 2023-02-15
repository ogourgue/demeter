"""Calculate coverage from Cellular Automaton to Telemac.

Todo: Docstrings.

"""
import os

import numpy as np
from mpi4py import MPI

from demeter import ca2tel
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
    vor = ca2tel.voronoi(X, Y, x, y, tri)

    # Coverage.
    cov = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if np.sum(vor == i) == 0:
            cov[i] = np.nan
        else:
            cov[i] = np.mean(STATE[vor == i])

    return cov

################################################################################
if __name__ == '__main__':

    # Global intermediate file names.
    X_global_fn = './tmp_ca2tel/ca_x_global.txt'
    Y_global_fn = './tmp_ca2tel/ca_y_global.txt'
    STATE_global_fn = './tmp_ca2tel/state_global.txt'
    x_global_fn = './tmp_ca2tel/tel_x_global.txt'
    y_global_fn = './tmp_ca2tel/tel_y_global.txt'
    tri_global_fn = './tmp_ca2tel/tri_global.txt'
    cov_global_fn = './tmp_ca2tel/cov_global.txt'
    bb_fn = './tmp_ca2tel/bb.txt'
    npoin_fn = './tmp_ca2tel/npoin.txt'

    # Test if mesh partitioning files are already available.
    ready = False
    if os.path.isfile(bb_fn):
        ready = True

    # Compute mesh partitioning by domain decomposition.
    if rank == 0 and not ready:

        # Load intermediate files.
        X = np.loadtxt(X_global_fn)
        Y = np.loadtxt(Y_global_fn)
        x = np.loadtxt(x_global_fn)
        y = np.loadtxt(y_global_fn)
        tri = np.loadtxt(tri_global_fn, dtype = int)

        # Cellular Automaton grid size.
        DX = X[1] - X[0]

        # Domain decomposition (Cellular Automaton grid). Convert X, Y to 2D
        # arrays before splitting, then convert all arrays back to 1D.
        X, Y = np.meshgrid(X, Y, indexing = 'ij')
        nx, ny = X.shape
        mpi_nx, mpi_ny = ca_mpi.get_domain_decomposition(nx, ny, nproc)
        X_list = ca_mpi.mpi_split_array(X, mpi_nx, mpi_ny, contiguous = False)
        Y_list = ca_mpi.mpi_split_array(Y, mpi_nx, mpi_ny, contiguous = False)
        for i in range(nproc):
            X_list[i] = X_list[i][:, 0]
            Y_list[i] = Y_list[i][0, :]
        X = X[:, 0]
        Y = Y[0, :]

        # Domain decomposition (Telemac grid). Take all triangles with at least
        # one vertex inside the Cellular Automaton sub-domain.
        x_list = []
        y_list = []
        tri_list = []
        glo2loc = [] # Global to local node index conversion.
        ext2loc = [] # Local extended to original local node index conversion.
        for i in range(nproc):
            # Cellular Automaton sub-domain bounding box.
            X0 = X_list[i][0] - DX
            X1 = X_list[i][-1] + DX
            Y0 = Y_list[i][0] - DX
            Y1 = Y_list[i][-1] + DX
            # Global indices of nodes inside the Cellular Automaton sub-domain.
            nod_glo = np.argwhere((x >= X0) * (x <= X1) *
                                  (y >= Y0) * (y <= Y1)).reshape(-1)
            glo2loc.append(nod_glo)
            # Global indices of triangles with at least one vertex inside the
            # Cellular Automaton sub-domain.
            tmp = np.zeros(tri.shape[0], dtype = bool)
            for j in range(3):
                tmp[np.isin(tri[:, j], nod_glo)] = True
            tri_glo = tri[tmp, :]
            # Local nodes with global indices (including some outside the
            # Cellular Automaton sub-domain but that are vertices of triangles
            # partly inside).
            nod_glo_extended = np.unique(tri_glo)
            ext2loc.append(np.intersect1d(nod_glo, nod_glo_extended,
                                          return_indices = True)[2])
            # Local node coordinates.
            x_list.append(x[nod_glo_extended])
            y_list.append(y[nod_glo_extended])
            # Local connectivity table with local indices.
            tri_loc = np.zeros(tri_glo.shape, dtype = int)
            for j in range(tri_loc.shape[0]):
                for k in range(tri_loc.shape[1]):
                    tri_loc[j, k] = int(np.argwhere(tri_glo[j, k] ==
                                                    nod_glo_extended))
            tri_list.append(tri_loc)

        # Extended domain decomposition (Cellular automaton grid). Cover all
        # local Telemac grid nodes (including some outside the original
        # Cellular Automaton sub-domain but that are vertices of triangles
        # partly inside).
        X_list = []
        Y_list = []
        bb = np.zeros((nproc, 4))
        for i in range(nproc):
            # Bounding box coordinates.
            if x_list[i].shape[0] > 0:
                xmin = np.min(x_list[i])
                xmax = np.max(x_list[i])
                ymin = np.min(y_list[i])
                ymax = np.max(y_list[i])
            # Bounding box indices.
            if x_list[i].shape[0] > 0:
                try:imin = int(np.argwhere(X <= xmin)[-1])
                except:imin = 0
                try:imax = int(np.argwhere(X >= xmax)[0])
                except:imax = len(X) - 1
                try:jmin = int(np.argwhere(Y <= ymin)[-1])
                except:jmin = 0
                try:jmax = int(np.argwhere(Y >= ymax)[0])
                except:jmax = len(Y) - 1
            # Extended Cellular Automaton coordinates.
            if x_list[i].shape[0] > 0:
                X_list.append(np.ascontiguousarray(X[imin:imax + 1]))
                Y_list.append(np.ascontiguousarray(Y[jmin:jmax + 1]))
            else:
                X_list.append(np.array([]))
                Y_list.append(np.array([]))
            # Add bounding box indices to bounding box array.
            if x_list[i].shape[0] > 0:
                bb[i, 0] = imin
                bb[i, 1] = imax
                bb[i, 2] = jmin
                bb[i, 3] = jmax

        # Save local intermediate files.
        for i in range(nproc):

            # Local intermediate file names.
            X_local_fn = './tmp_ca2tel/ca_x_local_%d.txt' % i
            Y_local_fn = './tmp_ca2tel/ca_y_local_%d.txt' % i
            x_local_fn = './tmp_ca2tel/tel_x_local_%d.txt' % i
            y_local_fn = './tmp_ca2tel/tel_y_local_%d.txt' % i
            tri_local_fn = './tmp_ca2tel/tri_local_%d.txt' % i
            glo2loc_local_fn = './tmp_ca2tel/glo2loc_local_%d.txt' % i
            ext2loc_local_fn = './tmp_ca2tel/ext2loc_local_%d.txt' % i

            # Save local intermediate files.
            np.savetxt(X_local_fn, X_list[i])
            np.savetxt(Y_local_fn, Y_list[i])
            np.savetxt(x_local_fn, x_list[i])
            np.savetxt(y_local_fn, y_list[i])
            np.savetxt(tri_local_fn, tri_list[i], fmt = '%d')
            np.savetxt(glo2loc_local_fn, glo2loc[i], fmt = '%d')
            np.savetxt(ext2loc_local_fn, ext2loc[i], fmt = '%d')

        # Save global intermediate files.
        np.savetxt(bb_fn, bb, fmt = '%d')
        np.savetxt(npoin_fn, np.array([x.shape[0]]), fmt = '%d')

    # Primary processor decomposes input variable for each partition and sends
    # partitions to secondary processors.
    if rank == 0:

        # Load global input variable.
        STATE = np.loadtxt(STATE_global_fn, dtype = int)

        # Load bounding box file.
        bb = np.loadtxt(bb_fn, dtype = int)

        # Input variable for each partition.
        STATE_list = []
        for i in range(nproc):
            if np.sum(bb[i, :]) > 0:
                imin = bb[i, 0]
                imax = bb[i, 1]
                jmin = bb[i, 2]
                jmax = bb[i, 3]
                STATE_list.append(np.ascontiguousarray(STATE[imin:imax + 1,
                                                             jmin:jmax + 1]))
            else:
                STATE_list.append(np.zeros((0, 0)))

        # Primary processor partition.
        STATE_loc = STATE_list[0]

        # Send partitions to secondary processors.
        for i in range(1, nproc):
            nx_loc = STATE_list[i].shape[0]
            ny_loc = STATE_list[i].shape[1]
            comm.send(nx_loc, dest = i, tag = 600)
            comm.send(ny_loc, dest = i, tag = 601)
            comm.Send([STATE_list[i], MPI.INT], dest = i, tag = 602)

    # Secondary processors receive input variable partitions from primary
    # processor.
    if rank > 0:

        # Receive partition from primary processor.
        nx_loc = comm.recv(source = 0, tag = 600)
        ny_loc = comm.recv(source = 0, tag = 601)
        STATE_loc = np.empty((nx_loc, ny_loc), dtype = int)
        comm.Recv([STATE_loc, MPI.INT], source = 0, tag = 602)

    # Local intermediate file names.
    X_local_fn = './tmp_ca2tel/ca_x_local_%d.txt' % rank
    Y_local_fn = './tmp_ca2tel/ca_y_local_%d.txt' % rank
    x_local_fn = './tmp_ca2tel/tel_x_local_%d.txt' % rank
    y_local_fn = './tmp_ca2tel/tel_y_local_%d.txt' % rank
    tri_local_fn = './tmp_ca2tel/tri_local_%d.txt' % rank

    # Load local mesh partition.
    if STATE_loc.shape != (0, 0):
        X_loc = np.loadtxt(X_local_fn)
        Y_loc = np.loadtxt(Y_local_fn)
        x_loc = np.loadtxt(x_local_fn)
        y_loc = np.loadtxt(y_local_fn)
        tri_loc = np.loadtxt(tri_local_fn, dtype = int)
    else:
        X_loc = np.array([])
        Y_loc = np.array([])
        x_loc = np.array([])
        y_loc = np.array([])
        tri_loc = np.array([], dtype = int)

    # Voronoi coverage.
    cov_loc = voronoi_coverage(X_loc, Y_loc, STATE_loc, x_loc, y_loc, tri_loc)

    # Secondary processors send output variable partitions to primary processor.
    if rank > 0:

        # Send partition to primary processor.
        npoin_loc = cov_loc.shape[0]
        comm.send(npoin_loc, dest = 0, tag = 603)
        comm.Send([cov_loc, MPI.FLOAT], dest = 0, tag = 604)

    # Primary processor receives partitions from secondary processors and
    # reconstructs output variable.
    if rank == 0:

        # Primary processor partition.
        cov_list = [cov_loc]

        # Receive partition from secondary processors.
        for i in range(1, nproc):
            npoin_loc = comm.recv(source = i, tag = 603)
            cov_list.append(np.empty(npoin_loc, dtype = float))
            comm.Recv([cov_list[i], MPI.FLOAT], source = i, tag = 604)

        # Load number of Telemac grid nodes.
        npoin = np.loadtxt(npoin_fn, dtype = int)

        # Reconstruction.
        cov = np.zeros(npoin)
        for i in range(nproc):
            # Load local intermediate files.
            glo2loc_local_fn = './tmp_ca2tel/glo2loc_local_%d.txt' % i
            ext2loc_local_fn = './tmp_ca2tel/ext2loc_local_%d.txt' % i
            if cov_list[i].shape[0] > 0:
                glo2loc = np.loadtxt(glo2loc_local_fn, dtype = int)
                ext2loc = np.loadtxt(ext2loc_local_fn, dtype = int)
            else:
                glo2loc = np.array([], dtype = int)
                ext2loc = np.array([], dtype = int)
            # Reconstruction
            cov[glo2loc] = cov_list[i][ext2loc]

        # Save intermediate file.
        np.savetxt(cov_global_fn, cov)

    MPI.Finalize()
