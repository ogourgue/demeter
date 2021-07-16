"""Telemac to Cellular Automaton coupling functions.

Todo: Docstrings.

"""
################################################################################
def interpolation(x, y, f, tri, X, Y, nproc = 1):
    """Interpolate a Telemac variable on a Cellular Automaton grid.

    Args:
        x (NumPy array): Telemac grid node x-coordinates.
        y (NumPy array): Telemac grid node y-coordinates.
        f (NumPy array): Telemac variable to interpolate.
        tri (NumPy array): Telemac grid connectivity table.
        X (NumPy array): Cellular Automaton grid cell x-coordinates (1D).
        Y (NumPy array): Cellular Automaton grid cell y-coordinates (1D).
        nproc (int, optional): Number of MPI processes. Default to 1 (no MPI).

    """
    if nproc <= 1:

        ################
        # Serial mode. #
        ################

        # Call run_ca function.
        from demeter import tel2ca_interpolation
        F = tel2ca_interpolation.interpolation(x, y, f, tri, X, Y)

    else:

        #################
        # Parallel mode #
        #################

        print('Todo.')