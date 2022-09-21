import os
import numpy as np
from demeter import cellular_automaton

# Grid parameters.
x0 = 0
y0 = 0
nx = 10
ny = 10
dx = 1

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))

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


####################################################
# Cellular automaton state (class implementation). #
####################################################

# Export all time steps.
ca.export('test_all.bin')

# Export last time step.
ca.export('test_last.bin', step = -1)

# Test importing all time steps.
test = cellular_automaton.CellularAutomaton.from_file('test_all.bin')
assert np.array_equal(ca.state, test.state)

# Test importing last time step from all-time-step file.
test = cellular_automaton.CellularAutomaton.from_file('test_all.bin', step = -1)
assert np.array_equal(ca.state[-1, :, :], test.state[0, :, :])

# Test importing last time step from last-time-step file.
test = cellular_automaton.CellularAutomaton.from_file('test_last.bin')
assert np.array_equal(ca.state[-1, :, :], test.state[0, :, :])

# Delete files.
os.remove('test_all.bin')
os.remove('test_last.bin')


######################################################
# Cellular automaton state, function implementation. #
######################################################

# Export.
filename = 'test.bin'
times = ca.times
state = ca.state
cellular_automaton.export_state(filename, x0, y0, nx, ny, dx, times, state)

# Test importing all time steps.
state = cellular_automaton.import_state('test.bin')
assert np.array_equal(ca.state, state)

# Test importing last time step.
state = cellular_automaton.import_state('test.bin', step = -1)
assert np.array_equal(ca.state[-1, :, :], state[0, :, :])

# Delete file.
os.remove('test.bin')