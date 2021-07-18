"""Update cellular automaton state.

Todo: Docstrings.

"""
import sys

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
def run(state, p_est, p_die, r_exp, nt, mpi_nx = 1, mpi_ny = 1):
    """Update cellular automaton state.

    Args:
        state (NumPy array): Initial cellular automaton state.
        p_est (NumPy array): Probability of establishment.
        p_die (NumPy array): Probability of die-back.
        r_exp (NumPy array): Lateral expansion rate (number of grid cells per
            run).
        nt (int): Number of iterations.
        mpi_nx (nt, optional): Number of partitions along x-direction (parallel
            computing). Default to 1.
        mpi_ny (nt, optional): Number of partitions along y-direction (parallel
            computing). Default to 1.

    """
    # Rescale probabilities for each iteration.
    p_est = 1 - (1 - p_est) ** (1 / nt)
    p_die = 1 - (1 - p_die) ** (1 / nt)
    p_exp = r_exp / nt

    # Loop over iterations.
    for i in range(nt):
        # Number of neighbors.
        nn = get_number_neighbors(state, mpi_nx, mpi_ny)
        # Update probability of expansion. Multiplied by .25 so that number-of-
        # neighbor contribution is 1 on average.
        p_exp_nn = p_exp * nn * .25
        # Establishment and expansion.
        state = update_state(0, 1, state, 1 - (1 - p_est) * (1 - p_exp_nn))
        # Die-back.
        state = update_state(1, 0, state, p_die)

    return state

################################################################################
def get_number_neighbors(state, mpi_nx = 1, mpi_ny = 1):
    """Calculate number of vegetated cells among 8 neighboring cells.

    Args:
        state (NumPy array): Cellular automaton state.
        mpi_nx (nt, optional): Number of partitions along x-direction (parallel
            computing). Default to 1.
        mpi_ny (nt, optional): Number of partitions along y-direction (parallel
            computing). Default to 1.

    """
    # Initialize number of neighbors.
    nn = np.zeros(state.shape, dtype = int)

    # Calculate number of neighbors.
    nn[:, :-1] += (state[:, 1:] == 1) # North.
    nn[:-1, :-1] += (state[1:, 1:]  == 1) # North-East.
    nn[:-1, :] += (state[1:, :] == 1) # East.
    nn[:-1, 1:] += (state[1:, :-1] == 1) # South-East.
    nn[:, 1:] += (state[:, :-1] == 1) # South.
    nn[1:, 1:] += (state[:-1, :-1] == 1) # South-West.
    nn[1:, :] += (state[:-1, :] == 1) # West.
    nn[1:, :-1] += (state[:-1, 1:]  == 1) # North-West.

    # MPI communication.
    mpi_i, mpi_j = ca_mpi.mpi_rank_to_indices(rank, mpi_nx)
    nx, ny = state.shape

    # North.
    if mpi_j > 0:
        tmp = np.ascontiguousarray(state[:, 0])
        comm.Send([tmp, MPI.INT], dest = rank - mpi_nx, tag = 400)
    if mpi_j < mpi_ny - 1:
        state_north = np.empty(nx, dtype = int)
        comm.Recv([state_north, MPI.INT], source = rank + mpi_nx, tag = 400)
        nn[:, -1] += (state_north == 1)

    # East.
    if mpi_i > 0:
        tmp = state[0, :]
        comm.Send([tmp, MPI.INT], dest = rank - 1, tag = 402)
    if mpi_i < mpi_nx - 1:
        state_east = np.empty(ny, dtype = int)
        comm.Recv([state_east, MPI.INT], source = rank + 1, tag = 402)
        nn[-1, :] += (state_east == 1)

    # South.
    if mpi_j < mpi_ny - 1:
        tmp = np.ascontiguousarray(state[:, -1])
        comm.Send([tmp, MPI.INT], dest = rank + mpi_nx, tag = 404)
    if mpi_j > 0:
        state_south = np.empty(nx, dtype = int)
        comm.Recv([state_south, MPI.INT], source = rank - mpi_nx, tag = 404)
        nn[:, 0] += (state_south == 1)

    # West.
    if mpi_i < mpi_nx - 1:
        tmp = state[-1, :]
        comm.Send([tmp, MPI.INT], dest = rank + 1, tag = 406)
    if mpi_i > 0:
        state_west = np.empty(ny, dtype = int)
        comm.Recv([state_west, MPI.INT], source = rank - 1, tag = 406)
        nn[0, :] += (state_west == 1)

    # North-East.
    if mpi_i > 0 and mpi_j > 0:
        comm.send(state[0, 0], dest = rank - mpi_nx - 1, tag = 401)
    if mpi_i < mpi_nx - 1 and mpi_j < mpi_ny - 1:
        state_north_east = comm.recv(source = rank + mpi_nx + 1, tag = 401)
        nn[-1, -1] += (state_north_east == 1)
    if mpi_i < mpi_nx - 1:
        nn[-1, :-1] += (state_east[1:] == 1)
    if mpi_j < mpi_ny - 1:
        nn[:-1, -1] += (state_north[1:] == 1)

    # South-East.
    if mpi_i > 0 and mpi_j < mpi_ny - 1:
        comm.send(state[0, -1], dest = rank + mpi_nx - 1, tag = 403)
    if mpi_i < mpi_nx - 1 and mpi_j > 0:
        state_south_east = comm.recv(source = rank - mpi_nx + 1, tag = 403)
        nn[-1, 0] += (state_south_east == 1)
    if mpi_i < mpi_nx - 1:
        nn[-1, 1:] += (state_east[:-1] == 1)
    if mpi_j > 0:
        nn[:-1, 0] += (state_south[1:] == 1)

    # South-West.
    if mpi_i < mpi_nx - 1 and mpi_j < mpi_ny - 1:
        comm.send(state[-1, -1], dest = rank + mpi_nx + 1, tag = 405)
    if mpi_i > 0 and mpi_j > 0:
        state_south_west = comm.recv(source = rank - mpi_nx - 1, tag = 405)
        nn[0, 0] += (state_south_west == 1)
    if mpi_i > 0:
        nn[0, 1:] += (state_west[:-1] == 1)
    if mpi_j > 0:
        nn[1:, 0] += (state_south[:-1] == 1)

    # North-West.
    if mpi_i < mpi_nx - 1 and mpi_j > 0:
        comm.send(state[-1, 0], dest = rank - mpi_nx + 1, tag = 407)
    if mpi_i > 0 and mpi_j < mpi_ny - 1:
        state_north_west = comm.recv(source = rank + mpi_nx - 1, tag = 407)
        nn[0, -1] += (state_north_west == 1)
    if mpi_i > 0:
        nn[0, :-1] += (state_west[1:] == 1)
    if mpi_j < mpi_ny - 1:
        nn[1:, -1] += (state_north[:-1] == 1)

    return nn

################################################################################
def update_state(i, j, state, p):
    """Update cellular automaton state for transition i to j based on
    probability p.

    Args:
        i (int): State before transition.
        j (int): State after transition.
        state (NumPy array): Cellular automaton state before transition.
        p (NumPy array): Probability to transition from state i to j.

    """
    # Indices where cellular automaton state is i.
    ind = (state == i)

    # Update state only if there are cells to update.
    if np.sum(ind) > 0:

        # Generate random numbers for cells where cellular automaton state is i.
        test = np.array(np.random.rand(np.sum(ind)))

        # Test probabilities where cellular automaton state is i.
        tmp = state[ind].copy()
        tmp[p[ind] > test] = j

        # Update cellular automaton state.
        state[ind] = tmp

    return state

################################################################################
if __name__ == '__main__':

    # Parameter input.
    nt = int(sys.argv[1])
    seed = int(sys.argv[2])

    # Mesh partitioning by domain decomposition.
    if rank == 0:

        # Intermediate file names.
        state_0_global_fn = './tmp_cellular_automaton/state_0_global.txt'
        state_1_global_fn = './tmp_cellular_automaton/state_1_global.txt'
        p_est_global_fn = './tmp_cellular_automaton/p_est_global.txt'
        p_die_global_fn = './tmp_cellular_automaton/p_die_global.txt'
        r_exp_global_fn = './tmp_cellular_automaton/r_exp_global.txt'

        # Load intermediate files.
        state_0 = np.loadtxt(state_0_global_fn, dtype = int)
        p_est = np.loadtxt(p_est_global_fn)
        p_die = np.loadtxt(p_die_global_fn)
        r_exp = np.loadtxt(r_exp_global_fn)

        # Domain decomposition.
        nx, ny = state_0.shape
        mpi_nx, mpi_ny = ca_mpi.get_domain_decomposition(nx, ny, nproc)
        state_0_list = ca_mpi.mpi_split_array(state_0, mpi_nx, mpi_ny)
        p_est_list = ca_mpi.mpi_split_array(p_est, mpi_nx, mpi_ny)
        p_die_list = ca_mpi.mpi_split_array(p_die, mpi_nx, mpi_ny)
        r_exp_list = ca_mpi.mpi_split_array(r_exp, mpi_nx, mpi_ny)

        # Generate random seeds for all processes (required for
        # reproducibility).
        np.random.seed(seed)
        seed_list = np.random.randint(2 ** 32, size = nproc)

    # Mesh partitioning for primary processor.
    if rank == 0:

        # Primary processor partition.
        seed_loc = seed_list[0]
        state_0_loc = state_0_list[0]
        p_est_loc = p_est_list[0]
        p_die_loc = p_die_list[0]
        r_exp_loc = r_exp_list[0]

        # Send partition data to secondary processors.
        for i in range(1, nproc):
            nx_loc = state_0_list[i].shape[0]
            ny_loc = state_0_list[i].shape[1]
            comm.send(mpi_nx, dest = i, tag = 300)
            comm.send(mpi_ny, dest = i, tag = 301)
            comm.send(nx_loc, dest = i, tag = 302)
            comm.send(ny_loc, dest = i, tag = 303)
            comm.send(seed_list[i], dest = i, tag = 304)
            comm.Send([state_0_list[i], MPI.INT], dest = i, tag = 305)
            comm.Send([p_est_list[i], MPI.FLOAT], dest = i, tag = 306)
            comm.Send([p_die_list[i], MPI.FLOAT], dest = i, tag = 307)
            comm.Send([r_exp_list[i], MPI.FLOAT], dest = i, tag = 308)

    # Mesh partitioning for secondary processors.
    if rank > 0:

        # Receive partition data from primary processor.
        mpi_nx = comm.recv(source = 0, tag = 300)
        mpi_ny = comm.recv(source = 0, tag = 301)
        nx_loc = comm.recv(source = 0, tag = 302)
        ny_loc = comm.recv(source = 0, tag = 303)
        seed_loc = comm.recv(source = 0, tag = 304)
        state_0_loc = np.empty((nx_loc, ny_loc), dtype = int)
        p_est_loc = np.empty((nx_loc, ny_loc), dtype = float)
        p_die_loc = np.empty((nx_loc, ny_loc), dtype = float)
        r_exp_loc = np.empty((nx_loc, ny_loc), dtype = float)
        comm.Recv([state_0_loc, MPI.INT], source = 0, tag = 305)
        comm.Recv([p_est_loc, MPI.FLOAT], source = 0, tag = 306)
        comm.Recv([p_die_loc, MPI.FLOAT], source = 0, tag = 307)
        comm.Recv([r_exp_loc, MPI.FLOAT], source = 0, tag = 308)

    # Run Cellular Automaton.
    np.random.seed(seed_loc)
    state_1_loc = run(state_0_loc, p_est_loc, p_die_loc, r_exp_loc, nt, mpi_nx,
                      mpi_ny)

    # Global mesh reconstruction for secondary processors.
    if rank > 0:

        # Send partition data to primary processor.
        comm.Send([state_1_loc, MPI.INT], dest = 0, tag = 307)

    # Global mesh reconstruction for primary processor.
    if rank == 0:

        # Primary processor partition.
        state_1_list = [state_1_loc]

        # Receive partition data from secondary processors.
        for i in range(1, nproc):
            nx_loc = state_0_list[i].shape[0]
            ny_loc = state_0_list[i].shape[1]
            state_1_list.append(np.empty((nx_loc, ny_loc), dtype = int))
            comm.Recv([state_1_list[i], MPI.INT], source = i, tag = 307)

        # Reconstruction.
        state_1 = ca_mpi.mpi_aggregate_array(state_1_list, mpi_nx, mpi_ny)

        # Save intermediate file.
        np.savetxt(state_1_global_fn, state_1, fmt = '%d')