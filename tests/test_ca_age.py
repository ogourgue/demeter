import numpy as np
from demeter import cellular_automaton

# Grid parameters.
x0 = 0
y0 = 0
nx = 10
ny = 10
dx = 1

# Number fo years.
nt = 10

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx, with_age = True)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny), dtype = int))
ca.append_age(np.zeros((nx, ny), dtype = int))

# Loop over years.
for i in range(nt):
    ca.state[-1, i, i] = 1
    ca.run(1)

# Tests.
for i in range(nt):
    assert ca.state[-1, i, i] == 1
    assert ca.age[-1, i, i] == 10 - i


# Todo: Test with randome establishment.