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

# Delete previous difference figures.
if os.path.isdir('./Diff'):
    shutil.rmtree('./Diff')
os.mkdir('./Diff')

# Import Telemac output file.
vnames = ['bottom', 'hydroperiod', 'ebs impulse', 'coverage']
out = telemac.Telemac('./out_tel.slf', vnames = vnames)
ref = telemac.Telemac('./ref_tel.slf', vnames = vnames)
x = out.x
y = out.y
tri = out.ikle - 1
b_diff = out.b - ref.b
th_diff = (out.th - ref.th) / (N * (12 * 60 + 25) * 60) * 100
jb_diff = (out.jb - ref.jb) / (N * (12 * 60 + 25) * 60 * TAUC) * 100
cov_diff = (out.cov - ref.cov) * 100

# Figures.
plt.figure()
vmax = np.max(np.abs(b_diff))
plt.tripcolor(x, y, tri, b_diff[-1, :], cmap = 'RdBu', vmin = -vmax, vmax = vmax)
plt.axvline(x = -1000, color = 'k', linestyle = '--')
plt.axvline(x = 0, color = 'k', linestyle = '--')
plt.axis('scaled')
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.xticks([])
plt.yticks([])
plt.colorbar(orientation = 'horizontal', label = 'Bottom elevation (m)')
plt.title('Final year')
plt.tight_layout()
plt.savefig('./Diff/BottomElevation.png', dpi = 600)
plt.close()

plt.figure()
vmax = np.max(np.abs(th_diff))
plt.tripcolor(x, y, tri, th_diff[-1, :], cmap = 'RdBu', vmin = -vmax, vmax = vmax)
plt.axvline(x = -1000, color = 'k', linestyle = '--')
plt.axvline(x = 0, color = 'k', linestyle = '--')
plt.axis('scaled')
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.xticks([])
plt.yticks([])
plt.colorbar(orientation = 'horizontal', label = 'Hydroperiod (%)')
plt.title('Final year')
plt.tight_layout()
plt.savefig('./Diff/Hydroperiod.png', dpi = 600)
plt.close()

plt.figure()
vmax = np.max(np.abs(jb_diff))
plt.tripcolor(x, y, tri, jb_diff[-1, :], cmap = 'RdBu', vmin = -vmax, vmax = vmax)
plt.axvline(x = -1000, color = 'k', linestyle = '--')
plt.axvline(x = 0, color = 'k', linestyle = '--')
plt.axis('scaled')
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.xticks([])
plt.yticks([])
plt.colorbar(orientation = 'horizontal', label = 'Exceeding bottom shear impulse (%)')
plt.title('Final year')
plt.tight_layout()
plt.savefig('./Diff/ExceedingBottomShearImpulse.png', dpi = 600)
plt.close()

plt.figure()
vmax = np.max(np.abs(cov_diff))
plt.tripcolor(x, y, tri, cov_diff[-1, :], cmap = 'RdBu', vmin = -vmax, vmax = vmax)
plt.axvline(x = -1000, color = 'k', linestyle = '--')
plt.axvline(x = 0, color = 'k', linestyle = '--')
plt.axis('scaled')
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.xticks([])
plt.yticks([])
plt.colorbar(orientation = 'horizontal', label = 'Vegetation coverage (%)')
plt.title('Final year')
plt.tight_layout()
plt.savefig('./Diff/Coverage.png', dpi = 600)
plt.close()