"""Diffusion operator.

Todo: Docstrings.

"""
import os
import sys

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from mpi4py import MPI

# Base class of MPI communicators.
comm = MPI.COMM_WORLD

# Number of MPI processes.
nproc = comm.Get_size()

# Rank of mpi process.
rank = comm.Get_rank()

################################################################################
def diffusion(x, y, f, tri, nu, dt, t, ghost = None):
    """ P1 finite element diffusion operator with mass lumping.

    Mass lumping allows for easier implementation of the linear matrix equation
    solver in parallel.

    Args:
        x (NumPy array): Grid node x-coordinates.
        y (NumPy array): Grid node y-coordinates.
        f (NumPy array): Grid node field values.
        tri (NumPy array): Triangle connectivity table.
        nu (float): Diffusion coefficient (m^2/yr).
        dt (float): Time step (yr)
        t (float): Duration of diffusion (yr).
        ghost (NumPy array, optional): Internal array for parallel computing.
            Default to None.
    """
    # Number of grid nodes.
    npoin = len(x)

    # Number of triangles.
    ntri = len(tri)

    # Initialize Jacobian determinant matrix.
    jac = np.zeros(ntri)

    # Initialize matrix A.
    a = scipy.sparse.lil_matrix((npoin, npoin))

    # Initialize matrix b.
    b = scipy.sparse.lil_matrix((npoin, npoin))

    # Compute Jacobian determinant matrix.
    for i in range(len(tri)):
        # Local coordinates.
        x0 = x[tri[i, 0]]
        x1 = x[tri[i, 1]]
        x2 = x[tri[i, 2]]
        y0 = y[tri[i, 0]]
        y1 = y[tri[i, 1]]
        y2 = y[tri[i, 2]]
        # Jacobian determinant.
        jac[i] = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    # Feed matrix A.
    for i in range(ntri):
        # Local matrix (mass lumping).
        a_loc = jac[i] / 6 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # Global indices of local nodes.
        i0 = tri[i, 0]
        i1 = tri[i, 1]
        i2 = tri[i, 2]
        # Feed matrix A.
        a[i0, i0] += a_loc[0, 0]
        a[i1, i1] += a_loc[1, 1]
        a[i2, i2] += a_loc[2, 2]

    # Feed matrix b.
    for i in range(ntri):
        # Local coordinates.
        x0 = x[tri[i, 0]]
        x1 = x[tri[i, 1]]
        x2 = x[tri[i, 2]]
        y0 = y[tri[i, 0]]
        y1 = y[tri[i, 1]]
        y2 = y[tri[i, 2]]
        # Shape function derivatives.
        phi0x = (y1 - y2) / jac[i]
        phi1x = (y2 - y0) / jac[i]
        phi2x = (y0 - y1) / jac[i]
        phi0y = (x2 - x1) / jac[i]
        phi1y = (x0 - x2) / jac[i]
        phi2y = (x1 - x0) / jac[i]
        # Local matrix b.
        b_loc = np.zeros((3, 3))
        b_loc[0, 0] = -.5 * jac[i] * (phi0x * phi0x + phi0y * phi0y)
        b_loc[0, 1] = -.5 * jac[i] * (phi0x * phi1x + phi0y * phi1y)
        b_loc[0, 2] = -.5 * jac[i] * (phi0x * phi2x + phi0y * phi2y)
        b_loc[1, 0] = -.5 * jac[i] * (phi1x * phi0x + phi1y * phi0y)
        b_loc[1, 1] = -.5 * jac[i] * (phi1x * phi1x + phi1y * phi1y)
        b_loc[1, 2] = -.5 * jac[i] * (phi1x * phi2x + phi1y * phi2y)
        b_loc[2, 0] = -.5 * jac[i] * (phi2x * phi0x + phi2y * phi0y)
        b_loc[2, 1] = -.5 * jac[i] * (phi2x * phi1x + phi2y * phi1y)
        b_loc[2, 2] = -.5 * jac[i] * (phi2x * phi2x + phi2y * phi2y)
        # Global indices of local nodes.
        i0 = tri[i, 0]
        i1 = tri[i, 1]
        i2 = tri[i, 2]
        # Feed matrix b.
        b[i0, i0] += b_loc[0, 0]
        b[i0, i1] += b_loc[0, 1]
        b[i0, i2] += b_loc[0, 2]
        b[i1, i0] += b_loc[1, 0]
        b[i1, i1] += b_loc[1, 1]
        b[i1, i2] += b_loc[1, 2]
        b[i2, i0] += b_loc[2, 0]
        b[i2, i1] += b_loc[2, 1]
        b[i2, i2] += b_loc[2, 2]

    # Convert to sparse matrices.
    a = scipy.sparse.csr_matrix(a)
    b = scipy.sparse.csr_matrix(b)

    # Initialize diffused array.
    f1 = f.copy()

    # Initialize current time.
    ti = 0

    # Diffusion loop.
    while ti < t - dt:
        # For each time step, the array to diffuse is the diffused array from
        # the previous time step.
        f0 = f1.copy()
        # Solve linear matrix equation.
        f1 = scipy.sparse.linalg.spsolve(a, a.dot(f0) + nu * dt * b.dot(f0))
        # Update current time.
        ti += dt
        # Update values on ghost nodes in parallel mode.
        if nproc > 1:
            f1 = ghost_update(f1, ghost)

    # Finale time step (t - ti <= dt).
    f0 = f1.copy()
    f1 = scipy.sparse.linalg.spsolve(a, a.dot(f0) + nu * (t - ti) * b.dot(f0))

    return f1

################################################################################
def ghost_update(f1_loc, ghost_loc):
    """
    Todo: Docstrings.

    """
    # Values to transfer.
    f1_ghost_loc = f1_loc[ghost_loc[:, 0]]
    ghost_loc = np.ascontiguousarray(ghost_loc[:, 1:])

    # Secondary processors.
    if rank > 0:

        # Send partition data to primary processor.
        npoin_ghost_loc = f1_ghost_loc.shape[0]
        comm.send(npoin_ghost_loc, dest = 0, tag = 200)
        comm.Send([f1_ghost_loc, MPI.FLOAT], dest = 0, tag = 201)
        comm.Send([ghost_loc, MPI.INT], dest = 0, tag = 202)

    # Primary processor.
    if rank == 0:

        # Primary processor partition.
        f1_ghost_list = [f1_ghost_loc]
        ghost_list = [ghost_loc]

        # Receive partition from secondary processors.
        for i in range(1, nproc):
            npoin_ghost_loc = comm.recv(source = i, tag = 200)
            f1_ghost_list.append(np.empty(npoin_ghost_loc, dtype = float))
            ghost_list.append(np.empty((npoin_ghost_loc, 2), dtype = int))
            comm.Recv([f1_ghost_list[i], MPI.FLOAT], source = i, tag = 201)
            comm.Recv([ghost_list[i], MPI.INT], source = i, tag = 202)

        # Prepare partition data to send back.
        f1_ghost_tmp = np.empty((nproc, nproc), dtype = object)
        ghost_tmp = np.empty((nproc, nproc), dtype = object)
        for i in range(nproc):
            for j in range(nproc):
                if i != j:
                    ind = np.argwhere(ghost_list[i][:, 0] == j).reshape(-1)
                    if ind.shape[0] > 0:
                        f1_ghost_tmp[i, j] = f1_ghost_list[i][ind]
                        ghost_tmp[i, j] = ghost_list[i][:, 1][ind]
        # Merge temporary sub-arrays.
        f1_ghost_tmp_list = []
        ghost_tmp_list = []
        for i in range(nproc):
            f1_ghost_tmp_list.append([])
            ghost_tmp_list.append([])
            for j in range(nproc):
                if ghost_tmp[j, i] is not None:
                    f1_ghost_tmp_list[i].append(f1_ghost_tmp[j, i])
                    ghost_tmp_list[i].append(ghost_tmp[j, i])
        f1_ghost_list = []
        ghost_list = []
        for i in range(nproc):
            f1_ghost_list.append(f1_ghost_tmp_list[i][0])
            ghost_list.append(ghost_tmp_list[i][0])
            for j in range(1, len(ghost_tmp_list[i])):
                f1_ghost_list[i] = np.append(f1_ghost_list[i],
                                             f1_ghost_tmp_list[i][j])
                ghost_list[i] = np.append(ghost_list[i], ghost_tmp_list[i][j])

        # Primary processor partition.
        f1_ghost_loc = f1_ghost_list[0]
        ghost_loc = ghost_list[0]

        # Send partition data to secondary processors.
        for i in range(1, nproc):
            npoin_ghost_loc = f1_ghost_list[i].shape[0]
            comm.send(npoin_ghost_loc, dest = i, tag = 203)
            comm.Send([f1_ghost_list[i], MPI.FLOAT], dest = i, tag = 204)
            comm.Send([ghost_list[i], MPI.INT], dest = i, tag = 205)

    # Secondary processors.
    if rank > 0:

        # Receive partition data from primary processor.
        npoin_ghost_loc = comm.recv(source = 0, tag = 203)
        f1_ghost_loc = np.empty(npoin_ghost_loc, dtype = float)
        ghost_loc = np.empty(npoin_ghost_loc, dtype = int)
        comm.Recv([f1_ghost_loc, MPI.FLOAT], source = 0, tag = 204)
        comm.Recv([ghost_loc, MPI.INT], source = 0, tag = 205)

    # Update values.
    f1_loc[ghost_loc] = f1_ghost_loc

    return f1_loc


################################################################################
if __name__ == '__main__':

    # Parameter input.
    nu = float(sys.argv[1])
    dt = float(sys.argv[2])
    t = float(sys.argv[3])

    # Mesh partitioning by domain decomposition.
    if rank == 0:

        # Intermediate file names.
        x_global_fn = './tmp_diffusion/x_global.txt'
        y_global_fn = './tmp_diffusion/y_global.txt'
        f_global_fn = './tmp_diffusion/f_global.txt'
        f1_global_fn = './tmp_diffusion/f1_global.txt'
        tri_global_fn = './tmp_diffusion/tri_global.txt'
        metis_fn = './tmp_diffusion/metis.txt'

        # Load intermediate files.
        x = np.loadtxt(x_global_fn)
        y = np.loadtxt(y_global_fn)
        f = np.loadtxt(f_global_fn)
        tri = np.loadtxt(tri_global_fn, dtype = 'int')

        # Write mesh file for Metis.
        # Metis requires that node indices start at 1 (not 0 as in Demeter).
        tri1 = tri + 1
        file = open(metis_fn, 'w')
        file.write('%d\n' % tri.shape[0])
        for i in range(tri.shape[0]):
            file.write('%d %d %d\n' % (tri1[i, 0], tri1[i, 1], tri1[i, 2]))
        file.close()

        # Run Metis (domain decomposition).
        os.system('mpmetis ' + metis_fn + ' %d' % nproc)

        # Import node partitioning.
        npart = np.loadtxt(metis_fn + '.npart.%d' % nproc, dtype = int)

        # Global node indices for each partition (without ghost nodes).
        glo_list = []
        for i in range(nproc):
            glo_list.append(np.argwhere(npart == i).reshape(-1))

        # Number of real nodes for each partition (without ghost nodes).
        npoin_real_list = []
        for i in range(nproc):
            npoin_real_list.append(glo_list[i].shape[0])

        # Connectivity table for each partition with global node indices.
        # A triangle belongs to a partition if at least one of its vertices
        # belongs to that partition. Some triangles belong to several
        # partitions.
        tri_glo_list = []
        for i in range(nproc):
            ind = np.argwhere(np.sum(npart[tri] == i, axis = 1) > 0).reshape(-1)
            tri_glo_list.append(tri[ind, :])

        # Update global node indices for each partition (including ghost nodes).
        for i in range(nproc):
            # Node indices in connectivity table.
            ind_tri = np.unique(tri_glo_list[i])
            # Ghost node indices.
            ind_ghost = ind_tri[np.in1d(ind_tri, glo_list[i],
                                        assume_unique = True, invert = True)]
            # Add ghost nodes.
            glo_list[i] = np.append(glo_list[i], ind_ghost)

        # Number of ghost nodes for each partition.
        npoin_ghost_list = []
        for i in range(nproc):
            npoin_ghost_list.append(glo_list[i].shape[0] - npoin_real_list[i])

        # Ghost information for each partition. First column gives local node
        # indices that are ghost nodes on other partitions. Second column gives
        # the processor indices on which local nodes are ghost nodes (if a local
        # node is a ghost on several processors, one row for each processor).
        # Third column gives local node indices on ghost processors.
        tmp = np.empty((nproc, nproc), dtype = object)
        for i in range(nproc):
            for j in range(nproc):
                if i != j:
                    res = np.intersect1d(glo_list[i][:npoin_real_list[i]],
                                         glo_list[j][-npoin_ghost_list[j]:],
                                         assume_unique = True,
                                         return_indices = True)
                    tmp[i, j] = np.zeros((res[0].shape[0], 3), dtype = int)
                    # Local node indices that are ghost on other partitions.
                    tmp[i, j][:, 0] = res[1]
                    # Processor index on which local nodes are ghost nodes.
                    tmp[i, j][:, 1] = j
                    # Local node indices on ghost processors.
                    tmp[i, j][:, 2] = res[2] + npoin_real_list[j]
        # Merge temporary sub-arrays.
        tmp_list = []
        for i in range(nproc):
            tmp_list.append([])
            for j in range(nproc):
                if tmp[i, j] is not None:
                    if tmp[i, j].shape[0] > 0:
                        tmp_list[i].append(tmp[i, j])
        ghost_list = []
        for i in range(nproc):
            ghost_list.append(tmp_list[i][0])
            for j in range(1, len(tmp_list[i])):
                ghost_list[i] = np.append(ghost_list[i], tmp_list[i][j],
                                          axis = 0)

        # Connectivity table for each partition with local node indices.
        tri_list = []
        for i in range(nproc):
            tri_list.append(np.zeros(tri_glo_list[i].shape, dtype = int))
            for j in range(tri_list[i].shape[0]):
                for k in range(3):
                    tri_list[i][j, k] = np.argwhere(glo_list[i] ==
                                                    tri_glo_list[i][j, k])

        # Coordinates and variable for each partition.
        x_list = []
        y_list = []
        f_list = []
        for i in range(nproc):
            x_list.append(x[glo_list[i]])
            y_list.append(y[glo_list[i]])
            f_list.append(f[glo_list[i]])

    # Mesh partitioning for primary processor.
    if rank == 0:

        # Primary processor partition.
        x_loc = x_list[0]
        y_loc = y_list[0]
        f_loc = f_list[0]
        tri_loc = tri_list[0]
        ghost_loc = ghost_list[0]

        # Send partition data to secondary processors.
        for i in range(1, nproc):
            npoin_loc = x_list[i].shape[0]
            nelem_loc = tri_list[i].shape[0]
            npoin_ghost_loc = ghost_list[i].shape[0]
            comm.send(npoin_loc, dest = i, tag = 100)
            comm.send(nelem_loc, dest = i, tag = 101)
            comm.send(npoin_ghost_loc, dest = i, tag = 102)
            comm.Send([x_list[i], MPI.FLOAT], dest = i, tag = 103)
            comm.Send([y_list[i], MPI.FLOAT], dest = i, tag = 104)
            comm.Send([f_list[i], MPI.FLOAT], dest = i, tag = 105)
            comm.Send([tri_list[i], MPI.INT], dest = i, tag = 106)
            comm.Send([ghost_list[i], MPI.INT], dest = i, tag = 107)

    # Mesh partitioning for secondary processors.
    if rank > 0:

        # Receive partition data from primary processor.
        npoin_loc = comm.recv(source = 0, tag = 100)
        nelem_loc = comm.recv(source = 0, tag = 101)
        npoin_ghost_loc = comm.recv(source = 0, tag = 102)
        x_loc = np.empty(npoin_loc, dtype = float)
        y_loc = np.empty(npoin_loc, dtype = float)
        f_loc = np.empty(npoin_loc, dtype = float)
        tri_loc = np.empty((nelem_loc, 3), dtype = int)
        ghost_loc = np.empty((npoin_ghost_loc, 3), dtype = int)
        comm.Recv([x_loc, MPI.FLOAT], source = 0, tag = 103)
        comm.Recv([y_loc, MPI.FLOAT], source = 0, tag = 104)
        comm.Recv([f_loc, MPI.FLOAT], source = 0, tag = 105)
        comm.Recv([tri_loc, MPI.INT], source = 0, tag = 106)
        comm.Recv([ghost_loc, MPI.INT], source = 0, tag = 107)

    # Diffusion.
    f1_loc = diffusion(x_loc, y_loc, f_loc, tri_loc, nu, dt, t, ghost_loc)

    # Global mesh reconstruction for secondary processors.
    if rank > 0:

        # Send partition data to primary processor.
        comm.Send([f1_loc, MPI.FLOAT], dest = 0, tag = 108)

    # Global mesh reconstruction for primary processor.
    if rank == 0:

        # Primary processor partition.
        f1_list = [f1_loc]

        # Receive partition data from secondary processors.
        for i in range(1, nproc):
            npoin_loc = x_list[i].shape[0]
            f1_list.append(np.empty(npoin_loc, dtype = float))
            comm.Recv([f1_list[i], MPI.FLOAT], source = i, tag = 108)

        # Remove data from ghost nodes.
        for i in range(nproc):
            f1_list[i] = f1_list[i][:npoin_real_list[i]]

        # Reconstruction.
        npoin = x.shape[0]
        f1 = np.zeros(npoin)
        for i in range(nproc):
            f1[npart == i] = f1_list[i]

        # Save intermediate file.
        np.savetxt(f1_global_fn, f1)