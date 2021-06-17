"""Diffusion operator.

Todo: Docstrings.

"""
import sys

import numpy as np
import mpi4py.MPI

from demeter import diffusion

# Base class of MPI communicators.
comm = mpi4py.MPI.COMM_WORLD

# Number of MPI processes.
nproc = comm.Get_size()

# Rank of mpi process.
rank = comm.Get_rank()

if __name__ == '__main__':

    if rank == 0:

        # Intermediate file names.
        x_global_fn = './tmp_diffusion/x_global.txt'
        y_global_fn = './tmp_diffusion/y_global.txt'
        f_global_fn = './tmp_diffusion/f_global.txt'
        f1_global_fn = './tmp_diffusion/f1_global.txt'
        tri_global_fn = './tmp_diffusion/tri_global.txt'

        # Load intermediate files.
        x = np.loadtxt(x_global_fn)
        y = np.loadtxt(y_global_fn)
        f = np.loadtxt(f_global_fn)
        tri = np.loadtxt(tri_global_fn, dtype = 'int')

        # Other input data.
        nu = float(sys.argv[1])
        dt = float(sys.argv[2])
        t = float(sys.argv[3])

        # Diffusion.
        f1 = diffusion.diffusion(x, y, f, tri, nu, dt, t)

        # Save intermediate file.
        np.savetxt(f1_global_fn, f1)