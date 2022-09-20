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
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, dx)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny)))

# Loop over years.
for i in range(nt):
    ca.state[-1, i, i] = 1
    ca.run(1)

# Tests.
for i in range(nt):
    assert ca.state[-1, i, i] == 1
