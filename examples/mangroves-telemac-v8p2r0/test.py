import numpy as np

from demeter import cellular_automaton
from demeter import telemac

################################################################################
################################################################################
# This test only works for a simulation of 5 years on 4 processors. ############
################################################################################
################################################################################

############
# Telemac. #
############

# Reference.
ref = telemac.Telemac('./ref_tel.slf')

# Output.
out = telemac.Telemac('./out_tel.slf')

# Tests.
assert out.u is None
assert out.v is None
assert out.s is None
assert out.t is None
assert out.r is None
assert out.m is None
assert np.allclose(ref.b, out.b)
assert np.allclose(ref.th, out.th)
assert np.allclose(ref.jb, out.jb)
assert np.allclose(ref.cov, out.cov)


#######################
# Cellular automaton. #
#######################

# Reference.
ref = cellular_automaton.CellularAutomaton.from_file('./ref_ca_state.bin', './ref_ca_age.bin')

# Output.
out = cellular_automaton.CellularAutomaton.from_file('./out_ca_state.bin', './out_ca_age.bin')

# Tests.
assert np.array_equal(ref.state, out.state)
assert np.array_equal(ref.age, out.age)
