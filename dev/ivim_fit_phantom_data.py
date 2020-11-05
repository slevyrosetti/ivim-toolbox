from msct_image import Image
import numpy as np
import os
import ivim_fitting
import matplotlib.pyplot as plt


def fit_D_only(x, y):
    polyfit = np.poly1d(np.polyfit(x, y, 1))

    sum_squared_error = np.sum(np.square(y - polyfit(x)))

    sum_squared_deviation_from_mean = np.sum(np.square(y - np.mean(y)))

    r2 = 1 - sum_squared_error / sum_squared_deviation_from_mean

    return polyfit, r2, sum_squared_error

def get_r2(y, y_est):

    sum_squared_error = np.sum(np.square(y - y_est))

    sum_squared_deviation_from_mean = np.sum(np.square(y - np.mean(y)))

    return 1 - sum_squared_error / sum_squared_deviation_from_mean


# ===================== PARAMS =========================
wd = '/Users/slevy/data/ivim/flowflow/180608_speed_variations/slice'
data_folders = ['speed0', 'speed1', 'speed2', 'speed4']
dwi_fname = 'dwi_topup_mean.nii.gz'
bval_fname = 'bval_rl_mean.txt'
mask_fname = 'mask_homogeneous_zone.nii.gz'
low_bval_thr = 200
high_bval_thr = 500
ndirs = 1
# ======================================================
os.chdir(wd)

# load data
mask = Image(mask_fname).data
bvals = np.loadtxt(data_folders[0]+'/'+bval_fname, delimiter=None)
dwi = np.zeros(mask.shape + (len(bvals), len(data_folders)))
bvals_all_folders = np.zeros((len(bvals), len(data_folders)))
for i_folder in range(len(data_folders)):

    dwi[..., i_folder] = Image(data_folders[i_folder]+'/'+dwi_fname).data
    bvals_all_folders[:, i_folder] = np.loadtxt(data_folders[i_folder]+'/'+bval_fname, delimiter=None)

# check if all b-values are the same
if np.sum(np.tile(bvals, (bvals_all_folders.shape[1], 1)).T == bvals_all_folders) < np.size(bvals_all_folders):
    print("WARNING: All b-values are not equal for all folders.")

# extract mean value within ROI
# Sroi = np.zeros((len(bvals)/ndirs, ndirs))
# for i_dir in range(ndirs):
#     for i_vol in range(i_dir, len(bvals), ndirs):
#         dwi_i_b = dwi[:, :, :, i_vol]
#         Sroi[i_vol/ndirs, i_dir] = np.mean(dwi_i_b[mask > 0])
Sroi = np.zeros((len(bvals), len(data_folders)))  # bvals X speeds
for i_folder in range(len(data_folders)):
    for i_vol in range(bvals_all_folders.shape[0]):
        dwi_i_b = dwi[:, :, :, i_vol, i_folder]
        Sroi[i_vol, i_folder] = np.mean(dwi_i_b[mask > 0])

# normalize signal by first b-value signal
Sroi_norm = np.divide(Sroi, np.broadcast_to(Sroi[0, :], Sroi.shape))


# fit
for i_folder in range(len(data_folders)):

    # prepare IVIM fit object
    ivim_fit = ivim_fitting.IVIMfit(bvals=bvals_all_folders[:, i_folder],
                                    voxels_values=np.array([Sroi_norm[:, i_folder]]),
                                    voxels_idx=tuple([np.array([0]), np.array([0]), np.array([0])]),
                                    oplots_folder=wd + "/" + data_folders[i_folder],
                                    multithreading=0)
    # run fit
    ivim_fit.run_fit()

# fit D to each data
poly_fitD = []
for i_folder in range(len(data_folders)):
    fit_x = bvals_all_folders[:, i_folder]
    fit_y = Sroi[:, i_folder]
    p_highb, r2, sum_squared_error = fit_D_only(fit_x[(fit_x >= 500) & (fit_x <= 1000)], np.log(fit_y[(fit_x >= 500) & (fit_x <= 1000)]))
    poly_fitD.append(p_highb)

# plot
font = {'family': 'normal',
        'size'  : 18}
plt.rc('font', **font)

# fig, ax = plt.subplots(figsize=(12, 10))
# fig2, ax2 = plt.subplots(figsize=(12, 10))
# colors = plt.cm.jet(np.linspace(0, 1, ndirs))
# # colors = ['r', 'b', 'g', 'k', 'y', 'c', 'orange']
# xp = np.linspace(bvals_uniques[0], bvals_uniques[-1], 1000)
# for i_dir in range(ndirs):
#
#     Sroi_dir_i = Sroi[:, i_dir]
#     p_highb, r2, sum_squared_error = fit_D_only(bvals_uniques[(bvals_uniques >= 500) & (bvals_uniques <= 1000)], np.log(Sroi_dir_i[(bvals_uniques >= 500) & (bvals_uniques <= 1000)]))
#     r2_all = get_r2(np.log(Sroi_dir_i), p_highb(bvals_uniques))
#
#     xp = np.linspace(bvals_uniques[0], bvals_uniques[-1], 1000)
#     ax.plot(bvals_uniques, np.log(Sroi_dir_i), '+', label='data', color=colors[i_dir], markersize=3.)
#     ax.plot(xp, p_highb(xp), '-', label='fit 500 $\leq$ b $\leq$ 1000', linewidth=0.5, color=colors[i_dir])
#
#     xp = np.linspace(bvals_uniques[0], bvals_uniques[bvals_uniques == low_bval_thr], 1000)
#     ax2.plot(bvals_uniques[(0 <= bvals_uniques) & (bvals_uniques <= low_bval_thr)], np.log(Sroi_dir_i[(0 <= bvals_uniques) & (bvals_uniques <= low_bval_thr)]), '.', label='data', color=colors[i_dir])
#     ax2.plot(xp, p_highb(xp), '-', label='fit 500 $\leq$ b $\leq$ 1000', linewidth=0.5, color=colors[i_dir])
#
# ax.grid()
# ax.set(xlabel='b-values (s/mm$^2$)', ylabel='ln(S)', title='Self-diffusion due to thermal motion')
# # ax.legend()
# ax2.grid(which='both')
# ax2.set(xlabel='b-values(s/mm$^2$)', ylabel='ln(S)', title='Self-diffusion due to thermal motion')
# # ax2.legend()
# # ax2.annotate('D = %.2E mm$^2$/s\nR$^2$ = %.9f' % (-p_highb.c[0], r2_all), xycoords='figure fraction', xy=(0.6, 0.7))
#
# fig.savefig("plot_fit500_1000_lnS_vs_b.png")
# fig2.savefig("plot_fit500_1000_lnS_vs_b_zoom_low_bvals.png")
# plt.show(block=True)
fig, ax = plt.subplots(num="Fit D", figsize=(12, 10))
cmap = plt.cm.jet
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap(np.repeat(np.linspace(0, 1, len(data_folders)), 2)))
# colors = plt.cm.jet(np.linspace(0, 1, len(data_folders)))
# colors = ['r', 'b', 'g', 'k', 'y', 'c', 'orange']
xp = np.linspace(bvals[0], bvals[-1], 1000)

for i_folder in range(len(data_folders)):

    # color = '%02x%02x%02x' % tuple(colors[i_folder][0:-1])
    plot = ax.plot(bvals, np.log(Sroi[:, i_folder]), '.-', label=data_folders[i_folder], markersize=8.)
    ax.plot(xp, poly_fitD[i_folder](xp), '-', linewidth=0.5, color=plot[0].get_color())

ax.grid()
ax.set(xlabel='b-values (s/mm$^2$)', ylabel='ln(S)', title='DW signal for different pump speeds')
ax.legend()
fig.savefig("plot_DWIsignal_for_different_pump_speeds.png")

print('**** Done ****')





