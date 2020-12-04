#!/usr/bin/env python3

"""
@author: slevy
"""

import ivim_fitting
import numpy as np
import _pickle as pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.2e' % x))
fmt = mticker.FuncFormatter(g)



# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================

dataFolder = str(Path.home())+"/job/data/zurich/3T/20201109_phantomGel"

diffDirs = ["phase", "read", "slice"]


# ======================================================================================================================

# plt.rcParams.update({'axes.labelsize': 'large'})
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.name"] = "Times New Roman"
# plt.rcParams["font.style"]  = "italic"



fig, axes = plt.subplots(2, len(diffDirs), figsize=(17, 9.5), num="Definition of IVIM protocol")
plt.subplots_adjust(wspace=0.2, left=0.05, right=0.97, hspace=0.35, bottom=0.1, top=0.95)

# Phantom data analysis
# ----------------------------------------------------------------------------------------------------------------------
i_case = 0
for i_dir in range(len(diffDirs)):

    data = pickle.load(open(dataFolder+"/"+diffDirs[i_dir]+"/phantomData_analysis.pickle", "rb"))
    bvals_unique = np.unique(data[i_case]["b-values"])

    axes[0, i_dir].bar(bvals_unique, -100*data[i_case]["D across reps across all voxels"][:, 1]/np.mean(data[i_case]["D across reps across all voxels"][2:, 0]), width=45, color="tab:blue")
    axes[0, i_dir].set(xlabel='b-value (s/mm$^2$)', ylabel='', title='Diffusion direction = '+diffDirs[i_dir]+' encoding')
    axes[0, i_dir].xaxis.label.set_size(15)
    axes[0, i_dir].axhline(y=50, color="gray", lw=0.75)


axes[0, 0].set_ylabel("COV across repetitions (%)", fontsize=15)

# Look at the evolution of the proportion of perfusion-related signal in the total signal as a function of b-value
# ----------------------------------------------------------------------------------------------------------------------
# define variations range for Fivim, Dstar and D
F_range = [.06, .07, .08, .09, .1, .11, .12, .13, .14, .15]
Dstar_range = [9.0e-3, 9.5e-3, 10.5e-3, 11e-3, 12e-3, 12.5e-3, 13e-3, 13.5e-3, 14.0e-3, 14.5e-3]
D_range = np.linspace(0.3e-3,  1.7e-3, num=10)

# set fixed parameters
bvals = np.linspace(0, 1000, 1000)
S0 = 1

# first plot D=0.3e-3, Dstar=11.5e-3 and F varies
D = 0.3e-3
Dstar=11.5e-3
axes[1, 0].set(xlabel='b-values (s/mm$^2$)', ylabel='Perfusion-related signal / total signal (%)', title=r'f$_{IVIM}$=6 - 15%, D$^*$=11.5x10$^{-3}$mm²/s, D=0.3x10$^{-3}$mm²/s')
axes[1, 0].axhline(y=1, color="gray", lw=0.75)
for i_F in range(len(F_range)):
    axes[1, 0].plot(bvals, 100 * (ivim_fitting.ivim_1pool_model(bvals, S0, D, F_range[i_F], Dstar) - (S0 - F_range[i_F]) * np.exp(-bvals * D)) / ivim_fitting.ivim_1pool_model(bvals, S0, D, F_range[i_F], Dstar), label=r"f$_{{IVIM}}$={:.1f}%".format(100*F_range[i_F]))
axes[1, 0].legend()

# second plot D=0.3e-3, F=.15 and Dstar varies
D = 0.3e-3
F = .15
axes[1, 1].set(xlabel='b-values (s/mm$^2$)', ylabel='', title=r'f$_{IVIM}$=15%, D$^*$=9.0 - 14.5 x10$^{-3}$mm²/s, D=0.3 x10$^{-3}$mm²/s')
axes[1, 1].axhline(y=1, color="gray", lw=0.7)
for i_Dstar in range(len(Dstar_range)):
    axes[1, 1].plot(bvals, 100 * (ivim_fitting.ivim_1pool_model(bvals, S0, D, F, Dstar_range[i_Dstar]) - (S0 - F) * np.exp(-bvals * D)) / ivim_fitting.ivim_1pool_model(bvals, S0, D, F, Dstar_range[i_Dstar]), label=r"D$^*$={:.1f}$\times${}".format(Dstar_range[i_Dstar]/1e-3, fmt(1e-3)))
axes[1, 1].legend()

# third plot F=.15 and Dstar=11.5e-3 and D varies
F = .15
Dstar=11.5e-3
axes[1, 2].set(xlabel='b-values (s/mm$^2$)', ylabel='', title=r'f$_{IVIM}$=15%, D$^*$=11.5x10$^{-3}$mm²/s, D=0.3 - 1.7 x10$^{-3}$mm²/s')
axes[1, 2].axhline(y=1, color="gray", lw=0.7)
for i_D in range(len(D_range)):
    axes[1, 2].plot(bvals, 100 * (ivim_fitting.ivim_1pool_model(bvals, S0, D_range[i_D], F, Dstar) - (S0 - F) * np.exp(-bvals * D_range[i_D])) / ivim_fitting.ivim_1pool_model(bvals, S0, D_range[i_D], F, Dstar), label=r"D={:.1f}$\times${}".format(D_range[i_D]/1e-3, fmt(1e-3)))
axes[1, 2].legend()

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
font.set_size(11)
for ax in axes[1, :]:
    ax.title.set_fontproperties(font)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.set_ylim([0, 5])


# plt.show()
fig.savefig("fig_protocolDefinition.jpeg")
print(">>> Saved to: fig_protocolDefinition.jpeg")
plt.close(fig)



