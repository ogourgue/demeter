import os
import shutil
import sys

import numpy as np
import scipy.interpolate as interpolate

from demeter import ca2tel
from demeter import cellular_automaton
from demeter import cellular_automaton_functions as fct
from demeter import tel2ca
from demeter import telemac

################################################################################
################################################################################
# Global parameters. ###########################################################
################################################################################
################################################################################

# Random seed.
np.random.seed(0)

# Number of years.
NYEAR = 5

# Number of processors.
NPROC = int(sys.argv[1])

###################################
# Telemac (hydro-morphodynamics). #
###################################

# Tidal range.
TR = 5

# Initial topography profile.
X0 = [-2000, -1000, 2000]
B0 = [-TR, -.5 * TR, .25 * TR]

# Rigid bed level.
R0 = -.5 * TR

# Initial bottom surface noise range.
NR = TR / 10

# Dry bulk density.
RHO = 500

# Soil diffusivity.
NU = 100

# Relative time step for diffusion.
NU_DT = 1

# Number of tidal cycles in one morphological year.
N = 1

# Sea level rise rate.
SLRR = .005

# Critical bottom shear stress for plants (N/m2).
TAUC = .3

############################################
# Cellular automaton (vegetation dynamics) #
############################################

# Grid cell size.
DX = 5

# Background probability of establishment.
P = 1e-2

# Background probability of die-back.
Q = 1

# Background lateral expansion rate (m/yr).
R = 2

# Pivot (x50) probability values for the hydroperiod (establishment, die-back,
# expansion).
TH50 = [.3, .4, .35]

# X95 probability values for the hydroperiod (establishment, die-back,
# expansion).
TH95 = [.25, .45, .3]

# Pivot (x50) probability values for the exceeding bottom shear impulse
# (establishment, die-back, expansion).
JB50 = [.03, .04, .035]

# X95 probability values for the exceeding bottom shear impulse (establishment,
# die-back, expansion).
JB95 = [.02, .05, .025]

# Time between two outputs (yr).
CA_NYEAR = 5

################################################################################
################################################################################
# Pre-processing. ##############################################################
################################################################################
################################################################################

############
# Telemac. #
############

# Import geometry.
tel = telemac.Telemac('./TIGER_2km_50m.slf')

# Initialize bottom surface.
b = np.interp(tel.x, X0, B0).reshape((1, tel.npoin))

# Add random noise on initial bottom surface.
# Random noise is generated on a coarser grid, then interpolated on finer grid.
tel_coarse = telemac.Telemac('./TIGER_2km_200m.slf')
noise = NR * (np.random.random(tel_coarse.npoin) - .5)
b += interpolate.griddata((tel_coarse.x, tel_coarse.y), noise, (tel.x, tel.y))

# Initialize other Telemac variables.
u = np.zeros((1, tel.npoin))
v = np.zeros((1, tel.npoin))
s = np.zeros((1, tel.npoin))
t = np.zeros((1, 1, tel.npoin))
r = np.minimum(b, R0)
m = ((b - r) * RHO).reshape((1, 1, 1, tel.npoin))
th = np.zeros((1, tel.npoin))
jb = np.zeros((1, tel.npoin))
cov = np.zeros((1, tel.npoin))
age = np.zeros((1, tel.npoin))

# Add times, variables and parameters to the Telemac instance.
tel.add_times([0])
tel.add_parameter(RHO, 'layers mud concentration')
tel.add_variable(u, 'velocity u')
tel.add_variable(v, 'velocity v')
tel.add_variable(s, 'free surface')
tel.add_variable(b, 'bottom')
tel.add_variable(t, 'coh sediment')
tel.add_variable(r, 'rigid bed')
tel.add_variable(m, 'mass mud')
tel.add_variable(th, 'hydroperiod')
tel.add_variable(jb, 'exceeding bottom shear impulse')
tel.add_variable(cov, 'coverage')
tel.add_variable(age, 'age')

# Create directory to store intermediate output files.
if os.path.isdir('./tmp'):
    shutil.rmtree('./tmp')
os.mkdir('./tmp')

# Initialize Telemac output instance.
tel.export('./tmp.slf', step = -1)
vnames = ['bottom', 'hydroperiod', 'ebs impulse', 'coverage', 'age']
tel_out = telemac.Telemac('./tmp.slf', vnames = vnames)

#######################
# Cellular automaton. #
#######################

# Grid parameters.
x0 = np.min(tel.x) + .5 * DX
y0 = np.min(tel.y) + .5 * DX
nx = int((np.max(tel.x) - np.min(tel.x)) / DX)
ny = int((np.max(tel.y) - np.min(tel.y)) / DX)

# Create cellular automaton.
ca = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, DX, with_age = True)

# Append initial time and state.
ca.append_times(0)
ca.append_state(np.zeros((nx, ny)))
ca.append_age(np.zeros((nx, ny)))

# Initialize Cellular Automaton output instance.
ca_out = cellular_automaton.CellularAutomaton(x0, y0, nx, ny, DX,
                                              with_age = True)
ca_out.append_times(ca.times[0])
ca_out.append_state(ca.state[0, :, :])
ca_out.append_age(ca.age[0, :, :])

################################################################################
################################################################################
# Main loop. ###################################################################
################################################################################
################################################################################

# Inter-annual loop.
for year in range(NYEAR):

    ############
    # Telemac. #
    ############

    # Reset coupling variables.
    tel.reset_variable('hydroperiod')
    tel.reset_variable('exceeding bottom shear impulse')

    # Intra-annual loop.
    for i in range(N):

        # Export next initial condition.
        tel.export('./tmp.slf', step = -1)

        # Run telemac.
        os.system('telemac2d.py ./t2d_input.cas --ncsize=%d' % NPROC)

        # Import results.
        t2d = telemac.Telemac('./out_t2d.slf', step = -1)
        gai = telemac.Telemac('./out_gai.slf', step = -1,
                              vnames = ['rigid bed', 'lay1 mass mud1'])

        # Append times and variables to the Telemac instance.
        tel.append_times(t2d.times[-1])
        tel.append_variable(t2d.u, 'velocity u')
        tel.append_variable(t2d.v, 'velocity v')
        tel.append_variable(t2d.s, 'free surface')
        tel.append_variable(t2d.b, 'bottom')
        tel.append_variable(t2d.t, 'coh sediment')
        tel.append_variable(gai.r, 'rigid bed')
        tel.append_variable(gai.m, 'mass mud')
        tel.append_variable(t2d.th, 'hydroperiod')
        tel.append_variable(t2d.jb, 'exceeding bottom shear impulse')

        # Diffuse bottom surface change.
        tel.diffuse_bottom(NU, NU_DT, t = 1 / N, nproc = NPROC)

        # Set intermediate output files aside.
        os.replace('./out_t2d.slf', './tmp/out_t2d_%03d_%02d.slf' % (year, i))
        os.replace('./out_gai.slf', './tmp/out_gai_%03d_%02d.slf' % (year, i))

        # Apply sea level rise to the next initial condition.
        tel.s[-1, :] += SLRR / N

        # Coverage not updated between two steps of intra-annual loop.
        # If not final step, add dummy coverage variable before removing
        # previous time step.
        if i < N - 1:
            tel.append_variable(tel.cov, 'coverage')
            tel.append_variable(tel.cov, 'age')

        # Remove previous time step.
        tel.remove_time_step()

    # Append times and variables to Telemac output instance.
    tel_out.append_times(tel.times[-1])
    tel_out.append_variable(tel.b[-1, :].reshape((1, -1)), 'bottom')
    tel_out.append_variable(tel.th[-1, :].reshape((1, -1)), 'hydroperiod')
    tel_out.append_variable(tel.jb[-1, :].reshape((1, -1)),
                            'exceeding bottom shear impulse')

    ##################################
    # Telemac to cellular automaton. #
    ##################################

    # Hydroperiod (dimensionless).
    # th = .2 --> flooded 20% of the time.
    th = tel2ca.interpolation(tel.x, tel.y,
                              tel.th[-1, :] / (N * (12 * 60 + 25) * 60),
                              tel.tri, ca.x, ca.y, nproc = NPROC)

    # Exceeding bottom shear impulse (dimensionless).
    # jb = .2 --> bottom shear stress 1x above critical value, 20% of time, or
    #             bottom shear stress 2x above critical value, 10% of time, etc.
    jb = tel2ca.interpolation(tel.x, tel.y,
                              tel.jb[-1, :] / (TAUC * N * (12 * 60 + 25) * 60),
                              tel.tri, ca.x, ca.y, nproc = NPROC)

    #######################
    # Cellular automaton. #
    #######################

    # Establishment probability.
    p_est = P * fct.step_tanh(th, TH50[0], TH95[0]) * \
                fct.step_tanh(jb, JB50[0], JB95[0])

    # Die-back probability.
    p_die = Q * (1 - (1 - fct.step_tanh(th, TH50[1], TH95[1])) * \
                     (1 - fct.step_tanh(jb, JB50[1], JB95[1])))

    # Lateral expansion rate.
    r_exp = R * fct.step_tanh(th, TH50[2], TH95[2]) * \
                fct.step_tanh(jb, JB50[2], JB95[2])

    # Update probabilities.
    ca.update_probabilities(p_est, p_die, r_exp)

    # Number of iterations (/yr).
    ca_nt = cellular_automaton.number_iterations(R, DX)

    # Run cellular automaton.
    ca.run(ca_nt, nproc = NPROC)

    # Append time.
    ca.append_times(year + 1)

    # Remove previous time step.
    ca.remove_time_step()

    # Append times, state and age to the Cellular Automaton output instance.
    if (year + 1) % CA_NYEAR == 0:
        ca_out.append_times(ca.times[-1])
        ca_out.append_state(ca.state[-1, :, :])
        ca_out.append_age(ca.age[-1, :, :])

    ##################################
    # Cellular automaton to Telemac. #
    ##################################

    # Coverage.
    cov = ca2tel.voronoi_coverage(ca.x, ca.y, ca.state[-1, :, :], tel.x, tel.y,
                                  tel.tri, nproc = NPROC)

    # Age.
    age = ca2tel.voronoi_age(ca.x, ca.y, ca.state[-1, :, :], ca.age[-1, :, :],
                             tel.x, tel.y, tel.tri, nproc = NPROC)

    # Append variables to the Telemac and Telemac output instances.
    tel.append_variable(cov.reshape((1, -1)), 'coverage')
    tel.append_variable(age.reshape((1, -1)), 'age')
    tel_out.append_variable(cov.reshape((1, -1)), 'coverage')
    tel_out.append_variable(age.reshape((1, -1)), 'age')

################################################################################
################################################################################
# Finalization. ################################################################
################################################################################
################################################################################

# Export Telemac results and et last intermediate output files aside.
tel_out.export('./out_tel.slf')

# Set last intermediate output files aside.
os.replace('./tmp/out_t2d_%03d_%02d.slf' % (NYEAR - 1, N - 1), './last_t2d.slf')
os.replace('./tmp/out_gai_%03d_%02d.slf' % (NYEAR - 1, N - 1), './last_gai.slf')

# Export Cellular Automaton results.
ca_out.export('out_ca_state.bin')
ca_out.export_age('out_ca_age.bin')

# Remove last initial condition file.
os.remove('./tmp.slf')

# Remove intermediate output files.
shutil.rmtree('./tmp')
shutil.rmtree('./tmp_ca2tel')
shutil.rmtree('./tmp_diffusion')
shutil.rmtree('./tmp_tel2ca')