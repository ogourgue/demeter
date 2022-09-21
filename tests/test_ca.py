import numpy as np
from demeter import cellular_automaton
from demeter import cellular_automaton_functions as fct

# Grid parameters.
x0 = 0
y0 = 0
nx = 10
ny = 10
dx = 1

# Random seed.
np.random.seed(0)


##################
# Establishment. #
##################

# Vegetation parameters.
p_est = .01
p_die = 0
r_exp = 0
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Loop over years.
for i in range(nt):
    ca.run(1)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Test.
assert np.array_equal(ca.state[-1, :, :], ref)


#############
# Die-back. #
#############

# Vegetation parameters.
p_est = 0
p_die = .1
r_exp = 0
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.ones((nx, ny), dtype = int))

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Loop over years.
for i in range(nt):
    ca.run(1)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]])

# Test.
assert np.array_equal(ca.state[-1, :, :], ref)


##############
# Expansion. #
##############

# Vegetation parameters.
p_est = 0
p_die = 0
r_exp = 5
nt = 1

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Initial state.
state_0 = np.zeros((nx, ny), dtype = int)
state_0[5, 5] = 1

# Append initial time and state.
ca.append_times(0)
ca.append_state(state_0)

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Number of iterations (/yr).
ca_nt = cellular_automaton.number_iterations(r_exp, dx)

# Loop over years.
for i in range(nt):
    ca.run(ca_nt)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 1, 1]])

# Test.
assert ca_nt == 10
assert np.array_equal(ca.state[-1, :, :], ref)


#############################################
# All processes, homogeneous probabilities. #
#############################################

# Vegetation parameters.
p_est = .01
p_die = .1
r_exp = 1
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Number of iterations (/yr).
ca_nt = cellular_automaton.number_iterations(r_exp, dx)

# Loop over years.
for i in range(nt):
    ca.run(ca_nt)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# Test.
assert np.array_equal(ca.state[-1, :, :], ref)


###############################################
# All processes, heterogeneous probabilities. #
###############################################

# Vegetation parameters.
p_est = .1
p_die = 1
r_exp = .5
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))

# Coordinates on the entire grid.
x, y = np.meshgrid(ca.x, ca.y, indexing = 'ij')

# Establishment only on the left (x < 5), die-back only on the bottom (y < 5).
p_est *= fct.step_tanh(x, 5, 6)
p_die *= fct.step_tanh(y, 5, 6)

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Number of iterations (/yr).
ca_nt = cellular_automaton.number_iterations(r_exp, dx)

# Loop over years.
for i in range(nt):
    ca.run(ca_nt)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

# Test.
assert np.array_equal(ca.state[-1, :, :], ref)


#########################################################
# All processes, heterogeneous probabilities, with age. #
#########################################################

# Vegetation parameters.
p_est = .1
p_die = 1
r_exp = .5
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx, with_age = True)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))
ca.append_age(np.zeros((nx, ny), dtype = int))

# Coordinates on the entire grid.
x, y = np.meshgrid(ca.x, ca.y, indexing = 'ij')

# Establishment only on the left (x < 5), die-back only on the bottom (y < 5).
p_est *= fct.step_tanh(x, 5, 6)
p_die *= fct.step_tanh(y, 5, 6)

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Number of iterations (/yr).
ca_nt = cellular_automaton.number_iterations(r_exp, dx)

# Loop over years.
for i in range(nt):
    ca.run(ca_nt)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [3, 4, 0, 3, 2, 0, 0, 0, 0, 0],
                [8, 5, 6, 1, 3, 0, 0, 0, 0, 0],
                [7, 6, 2, 4, 6, 0, 0, 0, 0, 0],
                [8, 3, 4, 5, 5, 1, 0, 0, 0, 0],
                [7, 6, 5, 7, 10, 0, 0, 0, 0, 0],
                [2, 10, 9, 7, 4, 2, 1, 0, 0, 0],
                [8, 9, 9, 7, 3, 1, 1, 0, 0, 0]])

# Test.
assert np.array_equal(ca.age[-1, :, :], ref)
