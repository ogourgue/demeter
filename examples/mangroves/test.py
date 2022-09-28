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
assert np.array_equal(ref.u, out.u)
assert np.array_equal(ref.v, out.v)
assert np.array_equal(ref.s, out.s)
assert np.array_equal(ref.b, out.b)
assert np.array_equal(ref.t, out.t)
assert np.array_equal(ref.r, out.r)
assert np.array_equal(ref.m, out.m)
assert np.array_equal(ref.th, out.th)
assert np.array_equal(ref.jb, out.jb)
assert np.array_equal(ref.cov, out.cov)


#######################
# Cellular automaton. #
#######################

# Reference.
ref = cellular_automaton.CellularAutomaton.from_file('./ref_ca_state.bin',
                                                     './ref_ca_age.bin')

# Output.
out = cellular_automaton.CellularAutomaton.from_file('./out_ca_state.bin',
                                                     './out_ca_age.bin')

# Tests.
assert np.array_equal(ref.state, out.state)
assert np.array_equal(ref.age, out.age)
