import numpy as np
from demeter import cellular_automaton
from demeter import cellular_automaton_functions as fct


# Random seed.
np.random.seed(0)

# Grid parameters.
x0 = 0
y0 = 0
nx = 10
ny = 10
dx = 1

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
    ca.run(ca_nt, nproc = 4)
    ca.append_times(i + 1)

# Final state.
ref = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]])

# Test.
assert np.array_equal(ca.state[-1, :, :], ref)