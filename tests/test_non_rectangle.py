import shutil

import numpy as np

from demeter import ca2tel
from demeter import cellular_automaton
from demeter import tel2ca
from demeter import telemac

# Random seed.
np.random.seed(0)

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

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny)))


##################################
# Telemac to cellular automaton. #
##################################

# Determine which cellular automaton cells are inside Telemac grid.
inside = tel2ca.interpolation(tel.x, tel.y, np.ones(tel.npoin), tel.tri, ca.x, ca.y, nproc = NPROC)

# Test.
assert np.sum(np.isnan(inside)) == 3601


###########################
# Run cellular automaton. #
###########################

# Initial state.
state = np.zeros((nx, ny))
state[np.isnan(inside)] = -1

# Append initial time and state.
ca.append_times(0)
ca.append_state(state)

# Update probabilities.
ca.update_probabilities(.5, 0, 0)

# Run cellular automaton.
ca.run(1, nproc = NPROC)

# Tests final state.
state = ca.state[-1, :, :]
assert np.mean(state[np.isnan(inside)]) == -1
assert np.around(np.mean(state[np.isfinite(inside)]), decimals = 6) == .491796


##################################
# Cellular automaton to Telemac. #
##################################

# Compute coverage on Telemac grid.
cov = ca2tel.voronoi_coverage(ca.x, ca.y, ca.state[-1, :, :], tel.x, tel.y, tel.tri, nproc = NPROC)

# Test.
assert np.around(np.mean(cov), decimals = 6) == .493506


#############
# Finalize. #
#############

# Delete intermediate files.
shutil.rmtree('./tmp_tel2ca')
shutil.rmtree('./tmp_ca2tel')
