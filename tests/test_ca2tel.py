import shutil

import numpy as np

from demeter import ca2tel
from demeter import cellular_automaton
from demeter import telemac


# Random seed.
np.random.seed(0)


#################
# Telemac grid. #
#################

# Create Telemac instance.
tel = telemac.Telemac('test_ca2tel_grid.slf')

# Telemac grid.
x = tel.x
y = tel.y
tri = tel.tri


############################
# Cellular automaton grid. #
############################

# Grid parameters.
x0 = 0
y0 = 0
nx = 100
ny = 100
dx = 1

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx, with_age = True)

# Cellular automaton grid.
X = ca.x
Y = ca.y


########################
# Vegetation dynamics. #
########################

# Vegetation parameters.
p_est = .01
p_die = .1
r_exp = 1
nt = 10

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))
ca.append_age(np.zeros((nx, ny), dtype = int))

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Number of iterations (/yr).
ca_nt = cellular_automaton.number_iterations(r_exp, dx)

# Loop over years.
for i in range(nt):
    ca.run(ca_nt)
    ca.append_times(i + 1)


###########################################
# Cellular automaton to Telemac (serial). #
###########################################

# Final cellular automaton state and age.
STATE = ca.state[-1, :, :]
AGE = ca.age[-1, :, :]

# Coverage.
cov = ca2tel.voronoi_coverage(X, Y, STATE, x, y, tri)

# Age.
age = ca2tel.voronoi_age(X, Y, STATE, AGE, x, y, tri)

"""# Save.
np.save('test_ca2tel_cov.npy', cov)
np.save('test_ca2tel_age.npy', age)"""

# Tests.
assert np.array_equal(cov, np.load('test_ca2tel_cov.npy'))
assert np.array_equal(age, np.load('test_ca2tel_age.npy'))


#############################################
# Cellular automaton to Telemac (parallel). #
#############################################

# Final cellular automaton state and age.
STATE = ca.state[-1, :, :]
AGE = ca.age[-1, :, :]

# Coverage.
cov = ca2tel.voronoi_coverage(X, Y, STATE, x, y, tri, nproc = 4)

# Age.
age = ca2tel.voronoi_age(X, Y, STATE, AGE, x, y, tri, nproc = 4)

# Tests.
assert np.array_equal(cov, np.load('test_ca2tel_cov.npy'))
assert np.array_equal(age, np.load('test_ca2tel_age.npy'))

# Delete intermediate files.
shutil.rmtree('./tmp_ca2tel')