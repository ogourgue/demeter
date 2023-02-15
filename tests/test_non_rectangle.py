import shutil

import numpy as np

from demeter import cellular_automaton
from demeter import tel2ca
from demeter import telemac


# Cellular automaton grid cell size.
DX = 1

# Number fo processors.
NPROC = 4


############
# Telemac. #
############

# Import grids.
tel = telemac.Telemac('./data/test_non_rectangle_grid.slf')


#######################
# Cellular automaton. #
#######################

# Grid parameters.
x0 = np.min(tel.x) + .5 * DX
y0 = np.min(tel.y) + .5 * DX
nx = int((np.max(tel.x) - np.min(tel.x)) / DX)
ny = int((np.max(tel.y) - np.min(tel.y)) / DX)

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, DX)


##################################
# Telemac to cellular automaton. #
##################################

# Interpolate x-coordinates.
x = tel2ca.interpolation(tel.x, tel.y, tel.x, tel.tri, ca.x, ca.y,
                         nproc = NPROC)

# Test.
assert np.mean(np.abs(ca.x - x[:, -1])) < 1e-9

# Delete intermediate files.
shutil.rmtree('./tmp_tel2ca')
