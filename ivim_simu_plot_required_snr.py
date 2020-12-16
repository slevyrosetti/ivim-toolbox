#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This program produces the figures plotting the results of the program "ivim_simu_compute_required_snr.py" which computes the required SNR for an estimation of FivimXDstar within 10% error margins.



Created on Mon Jul  8 19:21:50 2019

@author: slevy
"""

import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse


def main(ifname, ofname):
    """Main."""

    # load data
    minSNR, mean_err_for_minSNR, minSNR_diverging, true_params_values, F_range, Dstar_range, D_range, n_noise_simu = pickle.load(open(ifname, "rb"))
    n_sample = len(F_range)
    print(minSNR[0,0,:])
    print(np.sum(minSNR_diverging))

    # convert dictionaries array to array
    # true_params_values = np.zeros((n_sample, n_sample, n_sample), dtype=dict)  # true values for F, Dstar, D
    true_params_values_reshape = np.zeros((3, len(F_range), len(Dstar_range), len(D_range)))
    for i_F in range(len(F_range)):
        for i_Dstar in range(len(Dstar_range)):
            for i_D in range(len(D_range)):

                # # TEMPORARY
                # true_params_values[i_F, i_Dstar, i_D] = {'S0': 600, 'D': D_range[i_D], 'Fivim': F_range[i_F], 'Dstar': Dstar_range[i_Dstar]}

                # convert dictionaries array to array
                true_params_values_reshape[0, i_F, i_Dstar, i_D] = true_params_values[i_F, i_Dstar, i_D]["Fivim"]
                true_params_values_reshape[1, i_F, i_Dstar, i_D] = true_params_values[i_F, i_Dstar, i_D]["Dstar"]
                true_params_values_reshape[2, i_F, i_Dstar, i_D] = true_params_values[i_F, i_Dstar, i_D]["D"]

    # average across all D
    mean_minSNR_acrossD = np.mean(minSNR[true_params_values_reshape[0, :, :, :] > 0])
    # mean_err_FDstar_acrossD = np.mean(mean_err_FDstar_for_minSNR[true_params_values_reshape[0, :, :, :] > 0])

    # arbitrary boundaries for ISMRM abstracts
    minSNR_Dr = minSNR[:, :, 0]
    minSNR_Da = minSNR[:, :, 1]
    mask = np.ones(minSNR.shape, dtype=bool)
    mask[0, 0, :] = 0
    vmin_snr = np.min(minSNR[mask])
    vmax_snr = np.max(minSNR[mask])
    vmax_param = [55, 308, 23, 10]  # Fivim, Dstar, D, Fivim.Dstar
    vmin_param = [1.9, 0.5, 0, 6.2]  # Fivim, Dstar, D, Fivim.Dstar

    # display min required SNR if Fivim>= and D*>= mm2/s
    print('For D=%.3e mm2/s, if F>=%.4f and D*>=%.4e mm2/s, median required SNR = %.1f' % (D_range[0], F_range[1], Dstar_range[1], np.mean(minSNR_Dr[1:, 1:])))
    print('For D=%.3e mm2/s, if F>=%.4f and D*>=%.4e mm2/s, median required SNR = %.1f' % (D_range[1], F_range[1], Dstar_range[1], np.mean(minSNR_Da[1:, 1:])))
    print('For D=%.3e mm2/s, if F=%.4f and D*=%.4e mm2/s, median required SNR = %.1f' % (D_range[0], F_range[4], Dstar_range[2], minSNR[4, 2, 0]))
    print('For D=%.3e mm2/s, if F=%.4f and D*=%.4e mm2/s, median required SNR = %.1f' % (D_range[1], F_range[4], Dstar_range[2], minSNR[4, 2, 1]))

    plt.rcParams.update({'font.size': 25})
    fs_labels = 22

    F_range = 100*F_range # display Fivim in %
    for i_D in range(len(D_range)):

        minSNR_iD = minSNR[:, :, i_D]
        vmax_snr_iD = np.max(minSNR_iD[minSNR_iD != minSNR_iD[0, 0]])

        # plot estimation error on simulated params -------------------------------------------------------------
        fig_snr = plt.figure(figsize=(12, 11))
        ax_snr = plt.gca()
        fig_snr.suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        # minimum required SNR -------------------------------------------------------------
        ax_snr.set_aspect('equal', 'box')
        p_snr = ax_snr.pcolormesh(minSNR_iD, cmap="jet", linewidth=2, edgecolors='white', norm=colors.LogNorm(), vmin=10, vmax=vmax_snr)  # vmin=vmin_snr, vmax=vmax_snr, )
        ax_snr.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax_snr.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax_snr.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax_snr.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax_snr.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax_snr.set_xlabel('D$^*$ (mm$^2$/s)')
        ax_snr.set_ylabel('f$_{IVIM}$ (%)')
        # ax_snr.set_title('Minimum required SNR to get an error < 10%% on f$_{IVIM}$.D$^*$ \n(Mean = %.1f, Mean across all D = %.1f)\n' % (np.mean(minSNR_iD[idx_nonnulF_iD]), mean_minSNR_acrossD))
        ax_snr.set_title('(Median [min-max]= %d [%d - %.3e])' % (np.median(minSNR_iD), np.min(minSNR_iD), np.max(minSNR_iD)))
        divider_snr = make_axes_locatable(ax_snr)
        cax_snr = divider_snr.append_axes("right", size="5%", pad=0.05)
        cbar_snr = fig_snr.colorbar(p_snr, cax=cax_snr, ticks=LogLocator(subs=range(10)))
        cbar_snr.ax.yaxis.set_offset_position('left')
        cbar_snr.ax.tick_params(length=31, width=2)
        cbar_snr.update_ticks()
        cbar_snr.ax.set_ylabel('SNR')
        fig_snr.savefig(ofname + '_minSNR_D' + str(i_D) + '.png', transparent=True)
        plt.close(fig_snr)

        # corresponding estimation error
        fig_param, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(25, 20))
        fig_param.suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        # Fivim ----
        ax1.set_aspect('equal', 'box')
        # mean_err_on_Fivim_for_minSNR = mean_err_for_minSNR[0, :, :, :]
        p1 = ax1.pcolormesh(mean_err_for_minSNR[0, :, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=np.min(mean_err_for_minSNR[0, :, :, :]), vmax=np.max(mean_err_for_minSNR[0, :, :, :])) #, vmin=vmin_param[0], vmax=vmax_param[0])
        ax1.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax1.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax1.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        # ax1.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        ax1.set_ylabel('f$_{IVIM}$ (%)', fontsize=fs_labels)
        # ax2.set_title('Corresponding f$_{IVIM}$.D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)\n' % (np.mean(err_FDstar_iD[idx_nonnulF_iD]), mean_err_FDstar_acrossD))
        ax1.set_title('f$_{IVIM}$\n%.1f [%.1f - %.1f] %%' % (np.median(mean_err_for_minSNR[0, :, :, i_D]), np.min(mean_err_for_minSNR[0, :, :, i_D]), np.max(mean_err_for_minSNR[0, :, :, i_D])))
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig_param.colorbar(p1, cax=cax1)
        cbar1.ax.tick_params(length=25, width=1.5)
        cbar1.ax.yaxis.set_offset_position('left')
        cbar1.update_ticks()
        cbar1.ax.set_ylabel('estimation error (%)')

       # Dstar ----
        ax2.set_aspect('equal', 'box')
        # mean_err_on_Dstar_for_minSNR = mean_err_for_minSNR[1, :, :, :]
        p2 = ax2.pcolormesh(mean_err_for_minSNR[1, :, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=np.min(mean_err_for_minSNR[1, :, :, :]), vmax=np.max(mean_err_for_minSNR[1, :, :, :]))
        ax2.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax2.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax2.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        # ax2.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        # ax2.set_ylabel('f$_{IVIM}$ (%)', fontsize=fs_labels)
        # ax2.set_title('Corresponding f$_{IVIM}$.D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)\n' % (np.mean(err_FDstar_iD[idx_nonnulF_iD]), mean_err_FDstar_acrossD))
        ax2.set_title('D$^*$\n%.1f [%.1f - %.1f] %%' % (np.median(mean_err_for_minSNR[1, :, :, i_D]), np.min(mean_err_for_minSNR[1, :, :, i_D]), np.max(mean_err_for_minSNR[1, :, :, i_D])))
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig_param.colorbar(p2, cax=cax2)
        cbar2.ax.tick_params(length=25, width=2)
        cbar2.ax.yaxis.set_offset_position('left')
        cbar2.update_ticks()
        cbar2.ax.set_ylabel('estimation error (%)')

       # F.Dstar ----
        ax3.set_aspect('equal', 'box')
        # mean_err_on_FDstar_for_minSNR = mean_err_for_minSNR[3, :, :, :]
        p3 = ax3.pcolormesh(mean_err_for_minSNR[3, :, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=np.min(mean_err_for_minSNR[3, :, :, :]), vmax=np.max(mean_err_for_minSNR[3, :, :, :]))
        ax3.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax3.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        ax3.set_ylabel('f$_{IVIM}$ (%)', fontsize=fs_labels)
        # ax2.set_title('Corresponding f$_{IVIM}$.D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)\n' % (np.mean(err_FDstar_iD[idx_nonnulF_iD]), mean_err_FDstar_acrossD))
        ax3.set_title('f$_{IVIM}$.D$^*$\n%.1f [%.1f - %.1f] %%' % (np.median(mean_err_for_minSNR[3, :, :, i_D]), np.min(mean_err_for_minSNR[3, :, :, i_D]), np.max(mean_err_for_minSNR[3, :, :, i_D])))
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = fig_param.colorbar(p3, cax=cax3)
        cbar3.ax.tick_params(length=25, width=1.5)
        cbar3.ax.yaxis.set_offset_position('left')
        cbar3.update_ticks()
        cbar3.ax.set_ylabel('estimation error (%)')

       # D ----
        ax4.set_aspect('equal', 'box')
        # mean_err_on_D_for_minSNR = mean_err_for_minSNR[2, :, :, :]
        p4 = ax4.pcolormesh(mean_err_for_minSNR[2, :, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=np.min(mean_err_for_minSNR[2, :, :, :]), vmax=np.max(mean_err_for_minSNR[2, :, :, :]))
        ax4.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax4.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax4.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax4.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax4.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax4.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        # ax4.set_ylabel('f$_{IVIM}$ (%)', fontsize=fs_labels)
        # ax2.set_title('Corresponding f$_{IVIM}$.D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)\n' % (np.mean(err_FDstar_iD[idx_nonnulF_iD]), mean_err_FDstar_acrossD))
        ax4.set_title('D\n%.1f [%.1f - %.1f] %%' % (np.median(mean_err_for_minSNR[2, :, :, i_D]), np.min(mean_err_for_minSNR[2, :, :, i_D]), np.max(mean_err_for_minSNR[2, :, :, i_D])))
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        cbar4 = fig_param.colorbar(p4, cax=cax4)
        cbar4.ax.tick_params(length=25, width=1.5)
        cbar4.ax.yaxis.set_offset_position('left')
        cbar4.update_ticks()
        cbar4.ax.set_ylabel('estimation error (%)')

        plt.subplots_adjust(wspace=0.0, hspace=0.3, top=0.9, left=0.07, right=0.93)

        fig_param.savefig(ofname + '_params_D' + str(i_D) + '.png', transparent=True)
        plt.close(fig_param)

    print('=== Done ===')


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program produces the figures plotting the results of the program \"ivim_simu_compute_required_snr.py\" which computes the required SNR for an estimation of FivimXDstar within 10% error margins.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-input', dest='ifname', help='Result file produced by program \"ivim_simu_compute_required_snr.py\".', type=str, required=True)
    requiredArgs.add_argument('-oname', dest='ofname', help='Base name for output plots.', type=str, required=True)

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # print citation
    print('\n\n'
          '\n****************************** <3 Thank you for using our toolbox! <3 ***********************************'
          '\n********************************* PLEASE CITE THE FOLLOWING PAPER ***************************************'
          '\nLévy S, Rapacchi S, Massire A, et al. Intravoxel Incoherent Motion at 7 Tesla to quantify human spinal '
          '\ncord perfusion: limitations and promises. Magn Reson Med. 2020;00:1-20. https://doi.org/10.1002/mrm.28195'
          '\n*********************************************************************************************************'
          '\n\n')

    # run main
    main(ifname=args.ifname, ofname=args.ofname)

