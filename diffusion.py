"""Diffusion operator.

Todo: Docstrings.

"""
import os
import sys

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from mpi4py import MPI

################################################################################
# Base class of MPI communicators.
comm = MPI.COMM_WORLD

# Number of MPI processes.
nproc = comm.Get_size()

# Rank of mpi process.
rank = comm.Get_rank()

################################################################################
def diffusion(x, y, f, tri, nu, dt, t):
    """ Diffusion operator.

    Args:
        x (NumPy array): Grid node x-coordinates.
        y (NumPy array): Grid node y-coordinates.
        f (NumPy array): Grid node field values.
        tri (NumPy array): Triangle connectivity table.
        nu (float): Diffusion coefficient (m^2/yr).
        dt (float): Time step (yr)
        t (float): Duration of diffusion (yr).

    Todo:
        MPI implementation.

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
        # Local matrix.
        a_loc = jac[i] / 24 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        # Global indices of local nodes.
        i0 = tri[i, 0]
        i1 = tri[i, 1]
        i2 = tri[i, 2]
        # Feed matrix A.
        a[i0, i0] += a_loc[0, 0]
        a[i0, i1] += a_loc[0, 1]
        a[i0, i2] += a_loc[0, 2]
        a[i1, i0] += a_loc[1, 0]
        a[i1, i1] += a_loc[1, 1]
        a[i1, i2] += a_loc[1, 2]
        a[i2, i0] += a_loc[2, 0]
        a[i2, i1] += a_loc[2, 1]
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

    # Finale time step (t - ti <= dt).
    f0 = f1.copy()
    f1 = scipy.sparse.linalg.spsolve(a, a.dot(f0) + nu * (t - ti) * b.dot(f0))

    return f1

################################################################################
if __name__ == '__main__':

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

        # Other input data.
        nu = float(sys.argv[1])
        dt = float(sys.argv[2])
        t = float(sys.argv[3])

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

        ########################################################################
        # Everything below must be adjusted.
        # On each partition, we need the partition nodes and the ghost nodes.
        # Globally: npart gives the partition, another array gives the partition
        # on which the node is a ghost (-1 if it is not a ghost).
        # Locally: x, y, f, tri on partition nodes (including ghost nodes), a
        # list of local ghost indices, and tri_loc.
        ########################################################################

        # Split variables for each partition.
        x_list = []
        y_list = []
        f_list = []
        for i in range(nproc):
            x_list.append(x[npart == i])
            y_list.append(y[npart == i])
            f_list.append(f[npart == i])

        # Global to local indices. Entry i gives local partition index of global
        # node i.
        gloloc = np.zeros(x.shape, dtype = int)
        ind = np.zeros(nproc, dtype = int) - 1
        for i in range(x.shape[0]):
            ind[npart[i]] += 1
            gloloc[i] = ind[npart[i]]

        # Split triangles for each partiton. A triangle belongs to a partition
        # if at least one of its vertices belongs to that partition. Some
        # triangles belong to several partitions.
        tri_list = []

        print(npart[:10])
        print(tri[tri[:, 0] == 0, :])

        for i in range(nproc):
            # Connectivity table for partition i, with global node indices.
            tri_glo = np.argwhere(np.sum(npart[tri] == i, axis = 1) > 0)
            tri_glo = tri_glo.reshape(-1)
            # Convert global to local indices.
            tri_list.append(gloloc[tri[tri_glo, :]])

        # Check partitioning with figures.
        import matplotlib.pyplot as plt
        for i in range(nproc):
            plt.figure()
            plt.plot(x_list[i], y_list[i], '.')
            plt.savefig('test_%d.png' % i)
            plt.close()

        # Diffusion.
        f1 = diffusion(x, y, f, tri, nu, dt, t)

        # Save intermediate file.
        np.savetxt(f1_global_fn, f1)