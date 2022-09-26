import os
import numpy as np
from demeter import cellular_automaton

# Grid parameters.
x0 = 0
y0 = 0
nx = 10
ny = 10
dx = 1

# Vegetation parameters.
p_est = .01
p_die = 0
r_exp = 0
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx, with_age = True)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))
ca.append_age(np.zeros((nx, ny), dtype = int))

# Update probabilities.
ca.update_probabilities(p_est, p_die, r_exp)

# Loop over years.
for i in range(nt):
    ca.run(1)
    ca.append_times(i + 1)


#################################
# State (class implementation). #
#################################

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


###################################
# State, function implementation. #
###################################

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


###############################
# Age (class implementation). #
###############################

# Export all time steps.
ca.export('test_all.bin')
ca.export_age('test_all_age.bin')

# Export last time step.
ca.export('test_last.bin', step = -1)
ca.export_age('test_last_age.bin', step = -1)

# Test importing all time steps.
fn = 'test_all.bin'
fn_age = 'test_all_age.bin'
test = cellular_automaton.CellularAutomaton.from_file(fn, fn_age)
assert np.array_equal(ca.age, test.age)

# Test importing last time step from all-time-step file.
fn = 'test_all.bin'
fn_age = 'test_all_age.bin'
test = cellular_automaton.CellularAutomaton.from_file(fn, fn_age, step = -1)
assert np.array_equal(ca.age[-1, :, :], test.age[0, :, :])

# Test importing last time step from last-time-step file.
fn = 'test_last.bin'
fn_age = 'test_last_age.bin'
test = cellular_automaton.CellularAutomaton.from_file(fn, fn_age)
assert np.array_equal(ca.age[-1, :, :], test.age[0, :, :])

# Delete files.
os.remove('test_all.bin')
os.remove('test_all_age.bin')
os.remove('test_last.bin')
os.remove('test_last_age.bin')


#################################
# Age, function implementation. #
#################################

# Export.
filename = 'test.bin'
times = ca.times
age = ca.age
cellular_automaton.export_age(filename, x0, y0, nx, ny, dx, times, age)

# Test importing all time steps.
age = cellular_automaton.import_age('test.bin')
assert np.array_equal(ca.age, age)

# Test importing last time step.
age = cellular_automaton.import_age('test.bin', step = -1)
assert np.array_equal(ca.age[-1, :, :], age[0, :, :])

# Delete file.
os.remove('test.bin')