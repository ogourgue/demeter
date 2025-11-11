import os
import shutil

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from demeter import cellular_automaton
from demeter import telemac

# Number of tidal cycles in one morphological year.
N = 1

# Critical bottom shear stress for plants (N/m2).
TAUC = .3

# Sea level rise rate (mm/yr).
SLRR = .005

# Time between two Cellular Automaton outputs (yr).
DT = 5

# Delete previous figures.
if os.path.isdir('./Figures'):
    shutil.rmtree('./Figures')
os.mkdir('./Figures')

# Import Telemac output file.
vnames = ['bottom', 'hydroperiod', 'ebs impulse', 'coverage', 'age']
tel = telemac.Telemac('./out_tel.slf', vnames = vnames)
x = tel.x
y = tel.y
tri = tel.ikle - 1
b = tel.b
th = tel.th / (N * (12 * 60 + 25) * 60) * 100
jb = tel.jb / (N * (12 * 60 + 25) * 60 * TAUC) * 100
cov = tel.cov * 100
age = tel.age

# Figures.
for i in range(0, b.shape[0], DT):
    plt.figure()
    plt.tripcolor(x, y, tri, b[i, :] - i * SLRR, vmin = -2.5, vmax = 2.5)
    plt.axvline(x = -1000, color = 'k', linestyle = '--')
    plt.axvline(x = 0, color = 'k', linestyle = '--')
    plt.axis('scaled')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation = 'horizontal', label = 'Bottom elevation (m MSL)')
    plt.title('Year %d' % i)
    plt.tight_layout()
    plt.savefig('./Figures/BottomElevation%03d.png' % i, dpi = 600)
    plt.close()

for i in range(DT, th.shape[0], DT):
    plt.figure()
    plt.tripcolor(x, y, tri, th[i, :], vmin = 0, vmax = 100)
    plt.axvline(x = -1000, color = 'k', linestyle = '--')
    plt.axvline(x = 0, color = 'k', linestyle = '--')
    plt.axis('scaled')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation = 'horizontal', label = 'Hydroperiod (%)')
    plt.title('Year %d' % i)
    plt.tight_layout()
    plt.savefig('./Figures/Hydroperiod%03d.png' % i, dpi = 600)
    plt.close()

for i in range(DT, jb.shape[0], DT):
    plt.figure()
    plt.tripcolor(x, y, tri, jb[i, :], vmin = 0, vmax = 10)
    plt.axvline(x = -1000, color = 'k', linestyle = '--')
    plt.axvline(x = 0, color = 'k', linestyle = '--')
    plt.axis('scaled')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation = 'horizontal',
                 label = 'Exceeding bottom shear impulse (%)')
    plt.title('Year %d' % i)
    plt.tight_layout()
    plt.savefig('./Figures/ExceddingBottomShearImpulse%03d.png' % i, dpi = 600)
    plt.close()

for i in range(0, cov.shape[0], DT):
    plt.figure()
    plt.tripcolor(x, y, tri, cov[i, :], vmin = 0, vmax = 100, cmap = 'Greens')
    plt.axvline(x = -1000, color = 'k', linestyle = '--')
    plt.axvline(x = 0, color = 'k', linestyle = '--')
    plt.axis('scaled')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation = 'horizontal', label = 'Vegetation coverage (%)')
    plt.title('Year %d' % i)
    plt.tight_layout()
    plt.savefig('./Figures/Coverage%03d.png' % i, dpi = 600)
    plt.close()

for i in range(0, age.shape[0], DT):
    plt.figure()
    plt.tripcolor(x, y, tri, age[i, :], vmin = 0, vmax = age.shape[0] - 1,
                  cmap = 'Reds')
    plt.axvline(x = -1000, color = 'k', linestyle = '--')
    plt.axvline(x = 0, color = 'k', linestyle = '--')
    plt.axis('scaled')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation = 'horizontal', label = 'Vegetation age (yr)')
    plt.title('Year %d' % i)
    plt.tight_layout()
    plt.savefig('./Figures/Age%03d.png' % i, dpi = 600)
    plt.close()