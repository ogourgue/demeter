"""Calculate Cellular Automaton domain decomposition parameters.

Todo: Docstrings.

"""
import numpy as np

################################################################################
def get_domain_decomposition(nx, ny, nproc):
    """Calculate optimal numbers of partitions in x- and y-direction.

    Args:
        nx (int): Number of grid cells in x-direction.
        ny (int): Number of grid cells in y-direction.
        nproc (int): Number of MPI processes.

    """
    # Initialization.
    mpi_nx = []
    mpi_ny = []

    # Find all possible pairs.
    for i in range(1, nproc + 1):
        if i * int((nproc / i)) == nproc:
            mpi_nx.append(i)
            mpi_ny.append(int(nproc / i))

    # Find optimal pair, so that sub-domains are as squared as possible.
    mpi_nx = np.array(mpi_nx)
    mpi_ny = np.array(mpi_ny)
    ind = np.argmin(np.abs(mpi_nx - mpi_ny))
    mpi_nx, mpi_ny = np.sort([mpi_nx[ind], mpi_ny[ind]])

    return mpi_nx, mpi_ny

################################################################################
def mpi_split_array(array, mpi_nx, mpi_ny, contiguous = True):
    """Split a global array into list of partition arrays.

    Args:
        array (NumPy array): Array to split.
        mpi_nx (int): Number of partitions in x-direction.
        mpi_ny (int): Number of partitions in y-direction.

    """
    # Partition arrays organized as a two-order list.
    tmp = np.array_split(array, mpi_nx)
    for i in range(mpi_nx):
        tmp[i] = np.array_split(tmp[i], mpi_ny, axis = 1)

    # Convert two-order list into one-order list.
    array_list = []
    for j in range(mpi_ny):
        for i in range(mpi_nx):
            if contiguous:
                array_list.append(np.ascontiguousarray(tmp[i][j]))
            else:
                array_list.append(tmp[i][j])

    return array_list

################################################################################
def mpi_aggregate_array(array_list, mpi_nx, mpi_ny):
    """Aggregate list of partition arrays into global array.

    Args:
        array_list (list of NumPy arrays): Partition arrays to aggregate.
        mpi_nx (int): Number of partitions in x-direction.
        mpi_ny (int): Number of partitions in y-direction.

    """
    tmp = [None] * mpi_ny
    for i in range(mpi_ny):
        tmp[i] = np.concatenate(array_list[i * mpi_nx: (i + 1) * mpi_nx])
    array = np.concatenate(tmp, axis = 1)

    return array

################################################################################
def mpi_rank_to_indices(rank, mpi_nx):
    """Calculate MPI partition indices for a given rank.

    Args:
        rank (int): MPI rank.
        mpi_nx (int): Number of partitions in x-direction.

    """
    mpi_j = int(np.floor(rank / float(mpi_nx)))
    mpi_i = rank - mpi_j * mpi_nx

    return mpi_i, mpi_j