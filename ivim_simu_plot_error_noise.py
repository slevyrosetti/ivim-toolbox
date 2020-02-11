#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plot fitting error computed from Monte-Carlo simulations with program "ivim_simu_plot_error_noise".



Created on Mon Jul  8 16:25:15 2019

@author: slevy
"""

import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator
import argparse


def main(ifname, ofname):
    """Main."""

    # load data
    sim_params_values, true_params_values, F_range, Dstar_range, D_range, n_noise_simu, snr = pickle.load(open(ifname, "rb"))
    n_sample = len(F_range)

    # convert dictionaries array to array
    sim_params_values_reshape = np.zeros((3, len(F_range), len(Dstar_range), len(D_range), n_noise_simu))
    true_params_values_reshape = np.zeros((3, len(F_range), len(Dstar_range), len(D_range), n_noise_simu))
    for i_F in range(n_sample):
        for i_Dstar in range(n_sample):
            for i_D in range(len(D_range)):
                for i_noise in range(n_noise_simu):
                    i_flat = i_F * len(Dstar_range) * len(D_range) * n_noise_simu + i_Dstar * len(D_range) * n_noise_simu + i_D * n_noise_simu + i_noise
                    # simulated values
                    sim_params_values_reshape[0, i_F, i_Dstar, i_D, i_noise] = sim_params_values[i_flat]["Fivim"]
                    sim_params_values_reshape[1, i_F, i_Dstar, i_D, i_noise] = sim_params_values[i_flat]["Dstar"]
                    sim_params_values_reshape[2, i_F, i_Dstar, i_D, i_noise] = sim_params_values[i_flat]["D"]
                    # true values
                    true_params_values_reshape[0, i_F, i_Dstar, i_D, i_noise] = true_params_values[i_flat]["Fivim"]
                    true_params_values_reshape[1, i_F, i_Dstar, i_D, i_noise] = true_params_values[i_flat]["Dstar"]
                    true_params_values_reshape[2, i_F, i_Dstar, i_D, i_noise] = true_params_values[i_flat]["D"]

    # compute estimation error in percentage for each simulation
    Fivim_est_err_percent = 100*np.divide(np.abs(sim_params_values_reshape[0, :, :, :, :] - true_params_values_reshape[0, :, :, :, :]), true_params_values_reshape[0, :, :, :, :])
    Dstar_est_err_percent = 100*np.divide(np.abs(sim_params_values_reshape[1, :, :, :, :] - true_params_values_reshape[1, :, :, :, :]), true_params_values_reshape[1, :, :, :, :])
    D_est_err_percent = 100*np.divide(np.abs(sim_params_values_reshape[2, :, :, :, :] - true_params_values_reshape[2, :, :, :]), true_params_values_reshape[2, :, :, :, :])
    true_FDstar = np.multiply(true_params_values_reshape[0, :, :, :, :], true_params_values_reshape[1, :, :, :, :])
    sim_FDstar = np.multiply(sim_params_values_reshape[0, :, :, :, :], sim_params_values_reshape[1, :, :, :, :])
    FDstar_est_err_percent = 100*np.divide(np.abs(sim_FDstar - true_FDstar), true_FDstar)

    # average estimation errors across simulations
    mean_sim_Fivim_est_err_percent = np.mean(Fivim_est_err_percent, axis=3)
    mean_sim_Dstar_est_err_percent = np.mean(Dstar_est_err_percent, axis=3)
    mean_sim_D_est_err_percent = np.mean(D_est_err_percent, axis=3)
    mean_sim_FDstar_est_err_percent = np.mean(FDstar_est_err_percent, axis=3)
    # STD
    std_sim_Fivim_est_err_percent = np.std(Fivim_est_err_percent, axis=3)
    std_sim_Dstar_est_err_percent = np.std(Dstar_est_err_percent, axis=3)
    std_sim_D_est_err_percent = np.std(D_est_err_percent, axis=3)
    std_sim_FDstar_est_err_percent = np.std(FDstar_est_err_percent, axis=3)

    # for Fivim=0, Fivim*Dstar=0 then percentage error is infinite --> set it to 0
    mean_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] == 0] = 0
    mean_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] == 0] = 0
    mean_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] == 0] = 0
    std_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] == 0] = 0
    std_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] == 0] = 0
    std_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] == 0] = 0

    # display some specific values for 2019 ISMRM abstract
    print('Global mean estimation error on F*Dstar = %.4e mm$^2$/s\n\t\t\t\t  = %3f %%' % (np.mean(np.abs(sim_FDstar - true_FDstar)), np.mean(FDstar_est_err_percent[true_FDstar > 0])))

    print('For D = %.3e, f_ivim = %.2f, D* = %.3e, estimation errors (%%) are the following:\n- f_ivim: %.1f\n- D*: %.1f\n- f_ivim.D*: %.1f\n- D: %.1f' % (D_range[0], F_range[4], Dstar_range[2], mean_sim_Fivim_est_err_percent[4, 2, 0], mean_sim_Dstar_est_err_percent[4, 2, 0], mean_sim_FDstar_est_err_percent[4, 2, 0], mean_sim_D_est_err_percent[5, 2, 0]))

    print('For D = %.3e, f_ivim = %.2f, D* = %.3e, estimation errors (%%) are the following:\n- f_ivim: %.1f\n- D*: %.1f\n- f_ivim.D*: %.1f\n- D: %.1f' % (D_range[1], F_range[4], Dstar_range[2], mean_sim_Fivim_est_err_percent[4, 2, 1], mean_sim_Dstar_est_err_percent[4, 2, 1], mean_sim_FDstar_est_err_percent[4, 2, 1], mean_sim_D_est_err_percent[5, 2, 1]))

    print('[Weighted average of 3 directions] For D = %.3e, f_ivim = %.2f, D* = %.3e, estimation errors (%%) are the following:\n- f_ivim: %.1f\n- D*: %.1f\n- f_ivim.D*: %.1f\n- D: %.1f' % ((2.*D_range[0]+D_range[1])/3., F_range[4], Dstar_range[2], (2.*mean_sim_Fivim_est_err_percent[4, 2, 0]+mean_sim_Fivim_est_err_percent[4, 2, 1])/3., (2.*mean_sim_Dstar_est_err_percent[4, 2, 0]+mean_sim_Dstar_est_err_percent[4, 2, 1])/3., (2.*mean_sim_FDstar_est_err_percent[4, 2, 0]+mean_sim_FDstar_est_err_percent[4, 2, 1])/3., (2.*mean_sim_D_est_err_percent[5, 2, 0]+mean_sim_D_est_err_percent[5, 2, 1])/3.))


    # save maps for each D value
    # # (estimation error in mm2/s)
    # mean_estimation_err_by_D = np.mean(np.mean(estimation_err, axis=0), axis=0)
    # vmin, vmax = np.min(estimation_err), np.max(estimation_err)
    # for i_D in range(estimation_err.shape[2]):
    #     # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3))
    #     plt.figure()
    #     plt.pcolormesh(estimation_err[:, :, i_D], cmap="jet", vmin=vmin, vmax=vmax, linewidth=2, edgecolors='white')
    #     # plt.tick_params(direction='out')
    #     plt.tick_params(axis='both', which='both', length=0)  # remove ticks line
    #     plt.xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5], np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     plt.yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5], np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     plt.xlabel('D$^*$ (mm$^2$/s)')
    #     plt.ylabel('f$_{IVIM}$ (fraction)')
    #     plt.title('D = %.4e mm$^2$/s\nMean estimation error on f$_{IVIM}$.D$^*$ = %.2e mm$^2$/s (%.2e mm$^2$/s for all D)' % (D_range[i_D], mean_estimation_err_by_D[i_D], np.mean(estimation_err)))
    #     cbar = plt.colorbar()
    #     cbar.formatter.set_powerlimits((0, 0))
    #     cbar.ax.yaxis.set_offset_position('left')
    #     cbar.update_ticks()
    #     cbar.ax.set_ylabel('f$_{IVIM}$.D$^*$ estimation error (mm$^2$/s)')
    #     plt.savefig(ofname + '_D' + str(round(D_range[i_D], 4)) + '.png')
    #     plt.close()

    # # (estimation error in percent)
    # # estimation_err_percent[2, 0, :] = 0
    # vmin_percent, vmax_percent = np.min(estimation_err_percent[true_FDstar > 0]), np.max(estimation_err_percent[true_FDstar > 0])
    # for i_D in range(len(D_range)):
    #     # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3))
    #     plt.figure()
    #     plt.pcolormesh(estimation_err_percent[:, :, i_D], cmap="jet", vmin=vmin_percent, vmax=vmax_percent, linewidth=2, edgecolors='white')
    #     # plt.tick_params(direction='out')
    #     plt.tick_params(axis='both', which='both', length=0)  # remove ticks line
    #     plt.xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5],
    #                np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     plt.yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5],
    #                np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     plt.xlabel('D$^*$ (mm$^2$/s)')
    #     plt.ylabel('f$_{IVIM}$ (fraction)')
    #     plt.title('D = %.4e mm$^2$/s\nMean estimation error on f$_{IVIM}$.D$^*$ = %.1f %% (%.1f %% for all D)' % (D_range[i_D], np.mean(estimation_err_percent[true_FDstar[:, :, i_D] > 0, i_D]), np.mean(estimation_err_percent[true_FDstar > 0])))
    #     cbar = plt.colorbar()
    #     cbar.ax.yaxis.set_offset_position('left')
    #     cbar.update_ticks()
    #     cbar.ax.set_ylabel('f$_{IVIM}$.D$^*$ estimation error (%)')
    #     plt.savefig(ofname + '_percent_D' + str(round(D_range[i_D], 4)) + '.png')
    #     plt.close()

    # # plot true params values for sanity check purpose
    # vmin_true, vmax_true = np.min(true_FDstar), np.max(true_FDstar)
    # for i_D in range(len(D_range)):
    #     plt.figure()
    #     plt.pcolormesh(true_FDstar[:, :, i_D], cmap="jet", vmin=vmin_true, vmax=vmax_true, linewidth=2, edgecolors='white')
    #     # plt.tick_params(direction='out')
    #     plt.tick_params(axis='both', which='both', length=0)  # remove ticks line
    #     plt.xticks([0.5, n_sample / 2 - 0.5, n_sample - 0.5], np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     plt.yticks([0.5, n_sample / 2 - 0.5, n_sample - 0.5], np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     plt.xlabel('D$^*$ (mm$^2$/s)')
    #     plt.ylabel('f$_{IVIM}$ (fraction)')
    #     plt.title('D = %.4e mm$^2$/s' % D_range[i_D])
    #     cbar = plt.colorbar()
    #     cbar.ax.yaxis.set_offset_position('left')
    #     cbar.update_ticks()
    #     cbar.ax.set_ylabel('f$_{IVIM}$.D$^*$ (mm$^2$/s)')
    #     plt.savefig(ofname + '_true_' + str(round(D_range[i_D], 4)) + '.png')
    #     plt.close()

    vmin_F_percent, vmax_F_percent = np.min(mean_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0]), np.max(mean_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    vmin_Dstar_percent, vmax_Dstar_percent = np.min(mean_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0]), np.max(mean_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    vmin_D_percent, vmax_D_percent = np.min(mean_sim_D_est_err_percent), np.max(mean_sim_D_est_err_percent)
    vmin_FDstar_percent, vmax_FDstar_percent = np.min(mean_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0]), np.max(mean_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    # STD
    vmin_std_F_percent, vmax_std_F_percent = np.min(std_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0]), np.max(std_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    vmin_std_Dstar_percent, vmax_std_Dstar_percent = np.min(std_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0]), np.max(std_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    vmin_std_D_percent, vmax_std_D_percent = np.min(std_sim_D_est_err_percent), np.max(std_sim_D_est_err_percent)
    vmin_std_FDstar_percent, vmax_std_FDstar_percent = np.min(std_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0]), np.max(std_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    # # arbitrary boundaries for ISMRM abstracts
    # vmax_Dr = [1164, 394, 40, 428]  # Fivim, Dstar, D, Fivim.Dstar
    # vmin_Dr = [2.7, 6.7, 4.4, 4.2]  # Fivim, Dstar, D, Fivim.Dstar
    # vmax_Da = [1118, 436, 18, 2220]  # Fivim, Dstar, D, Fivim.Dstar
    # vmin_Da = [7.1, 14.6, 2.2, 15.7]  # Fivim, Dstar, D, Fivim.Dstar
    # arbitrary boundaries for IVIM paper
    # for 2SS approach
    vmax_Dr = [600., 350., 75., 370.]  # Fivim, Dstar, D, Fivim.Dstar
    vmin_Dr = [10., 10., 0., 10.]  # Fivim, Dstar, D, Fivim.Dstar
    vmax_Da = [1000., 300., 30., 1000.]  # Fivim, Dstar, D, Fivim.Dstar
    vmin_Da = [10., 10., 0., 10.]  # Fivim, Dstar, D, Fivim.Dstar
    # for FULL approach
    vmax_Dr = [600., 350., 75., 370.]  # Fivim, Dstar, D, Fivim.Dstar
    vmin_Dr = [10., 10., 0., 10.]  # Fivim, Dstar, D, Fivim.Dstar
    vmax_Da = [1000., 300., 30., 2000.]  # Fivim, Dstar, D, Fivim.Dstar
    vmin_Da = [10., 10., 0., 10.]  # Fivim, Dstar, D, Fivim.Dstar


    mean_Fivim_est_err = np.mean(mean_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    mean_Dstar_est_err = np.mean(mean_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    mean_D_est_err = np.mean(mean_sim_D_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    mean_FDstar_est_err = np.mean(mean_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    # STD
    mean_std_Fivim_est_err = np.mean(std_sim_Fivim_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    mean_std_Dstar_est_err = np.mean(std_sim_Dstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    mean_std_D_est_err = np.mean(std_sim_D_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])
    mean_std_FDstar_est_err = np.mean(std_sim_FDstar_est_err_percent[true_params_values_reshape[0, :, :, :, 0] > 0])

    plt.rcParams.update({'font.size': 20})
    fs_labels = 16
    # display Fivim in %
    F_range = 100*F_range

    for i_D in range(len(D_range)):

        if i_D == 0:
            vmax = vmax_Dr
            vmin = vmin_Dr
        else:
            vmax = vmax_Da
            vmin = vmin_Da

        idx_nonnulF_iD = true_params_values_reshape[0, :, :, i_D, 0] > 0
        Fivim_est_err_percent_iD = mean_sim_Fivim_est_err_percent[:, :, i_D]
        Dstar_est_err_percent_iD = mean_sim_Dstar_est_err_percent[:, :, i_D]
        D_est_err_percent_iD = mean_sim_D_est_err_percent[:, :, i_D]
        FDstar_est_err_percent_iD = mean_sim_FDstar_est_err_percent[:, :, i_D]
        # STD
        std_Fivim_est_err_percent_iD = std_sim_Fivim_est_err_percent[:, :, i_D]
        std_Dstar_est_err_percent_iD = std_sim_Dstar_est_err_percent[:, :, i_D]
        std_D_est_err_percent_iD = std_sim_D_est_err_percent[:, :, i_D]
        std_FDstar_est_err_percent_iD = std_sim_FDstar_est_err_percent[:, :, i_D]

        # plot estimation error on simulated params -------------------------------------------------------------
        fig_sim = plt.subplots(2, 2, figsize=(17, 15))
        ax1, ax2 = fig_sim[1][0]
        ax3, ax4 = fig_sim[1][1]
        # Fivim ----
        ax1.set_aspect('equal', 'box')
        p1 = ax1.pcolormesh(mean_sim_Fivim_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=vmin[0], vmax=vmax[0], norm=colors.LogNorm())  #, vmin=vmin_F_percent, vmax=vmax_F_percent)
        ax1.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax1.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax1.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax1.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        ax1.set_ylabel('f$_{IVIM}$ (fraction)', fontsize=fs_labels)
        # ax1.set_title('f$_{IVIM}$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)' % (np.mean(Fivim_est_err_percent_iD[idx_nonnulF_iD]), mean_Fivim_est_err))
        ax1.set_title('f$_{IVIM}$ estimation error\n%.1f [%.1f - %.1f] %%' % (np.median(Fivim_est_err_percent_iD), np.min(Fivim_est_err_percent_iD), np.max(Fivim_est_err_percent_iD)))
        fig_sim[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim[0].colorbar(p1, ax=ax1)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('estimation error (%)')
        # Dstar ----
        ax2.set_aspect('equal', 'box')
        p2 = ax2.pcolormesh(mean_sim_Dstar_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=vmin[1], vmax=vmax[1], norm=colors.LogNorm()) #, vmin=vmin_Dstar_percent, vmax=vmax_Dstar_percent)
        ax2.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax2.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax2.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax2.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        ax2.set_ylabel('f$_{IVIM}$ (fraction)', fontsize=fs_labels)
        # ax2.set_title('D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)' % (np.mean(Dstar_est_err_percent_iD[idx_nonnulF_iD]), mean_Dstar_est_err))
        ax2.set_title('D$^*$ estimation error\n%.1f [%.1f - %.1f] %%' % (np.median(Dstar_est_err_percent_iD), np.min(Dstar_est_err_percent_iD), np.max(Dstar_est_err_percent_iD)))
        fig_sim[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim[0].colorbar(p2, ax=ax2)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('estimation error (%)')
        # D ----
        ax3.set_aspect('equal', 'box')
        p3 = ax3.pcolormesh(mean_sim_D_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=vmin[2], vmax=vmax[2])  #, vmin=vmin_D_percent, vmax=vmax_D_percent)
        ax3.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax3.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        ax3.set_ylabel('f$_{IVIM}$ (fraction)', fontsize=fs_labels)
        ax3.set_title('D estimation error\n%.1f [%.1f - %.1f] %%' % (np.median(D_est_err_percent_iD), np.min(D_est_err_percent_iD), np.max(D_est_err_percent_iD)))
        fig_sim[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim[0].colorbar(p3, ax=ax3)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('estimation error (%)')
        # Fivim*Dstar ----
        ax4.set_aspect('equal', 'box')
        p4 = ax4.pcolormesh(mean_sim_FDstar_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white', vmin=vmin[3], vmax=vmax[3], norm=colors.LogNorm()) #, vmin=0, vmax=100)  #, vmin=vmin_FDstar_percent, vmax=vmax_FDstar_percent)
        ax4.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax4.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax4.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax4.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax4.set_yticklabels(np.take(np.round(F_range, 1), [0, round(n_sample / 2), n_sample - 1]))
        ax4.set_xlabel('D$^*$ (mm$^2$/s)', fontsize=fs_labels)
        ax4.set_ylabel('f$_{IVIM}$ (fraction)', fontsize=fs_labels)
        ax4.set_title('f$_{IVIM}$.D$^*$ estimation error\n%.1f [%.1f - %.1f] %%' % (np.median(FDstar_est_err_percent_iD), np.min(FDstar_est_err_percent_iD), np.max(FDstar_est_err_percent_iD)))
        fig_sim[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim[0].colorbar(p4, ax=ax4)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('estimation error (%)')

        plt.subplots_adjust(wspace=0.4, top=0.9, left=0.07, right=0.93)

        fig_sim[0].savefig(ofname + '_sim_D' + str(i_D) + '.png')
        plt.close(fig_sim[0])



        # plot STD across simulations --------------------------------------------------------------------
        fig_sim_std = plt.subplots(2, 2, figsize=(17, 15))
        ax1, ax2 = fig_sim_std[1][0]
        ax3, ax4 = fig_sim_std[1][1]
        # Fivim ----
        p1 = ax1.pcolormesh(std_sim_Fivim_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white')  #, vmin=vmin_std_F_percent, vmax=vmax_std_F_percent)
        ax1.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax1.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax1.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax1.set_xlabel('D$^*$ (mm$^2$/s)')
        ax1.set_ylabel('f$_{IVIM}$ (fraction)')
        ax1.set_title('STD across simulations of f$_{IVIM}$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)' % (np.mean(std_Fivim_est_err_percent_iD[idx_nonnulF_iD]), mean_std_Fivim_est_err))
        fig_sim_std[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim_std[0].colorbar(p1, ax=ax1)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('STD across simulations on estimation error (%)')
        # Dstar ----
        p2 = ax2.pcolormesh(std_sim_Dstar_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white')  #, vmin=vmin_std_Dstar_percent, vmax=vmax_std_Dstar_percent)
        ax2.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax2.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax2.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax2.set_xlabel('D$^*$ (mm$^2$/s)')
        ax2.set_ylabel('f$_{IVIM}$ (fraction)')
        ax2.set_title('STD across simulations of D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)' % (np.mean(std_Dstar_est_err_percent_iD[idx_nonnulF_iD]), mean_std_Dstar_est_err))
        fig_sim_std[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim_std[0].colorbar(p2, ax=ax2)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('STD across simulations on estimation error (%)')
        # D ----
        p3 = ax3.pcolormesh(std_sim_D_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white')  #, vmin=vmin_std_D_percent, vmax=vmax_std_D_percent)
        ax3.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax3.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_xlabel('D$^*$ (mm$^2$/s)')
        ax3.set_ylabel('f$_{IVIM}$ (fraction)')
        ax3.set_title('STD across simulations of D estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)' % (np.mean(std_D_est_err_percent_iD[idx_nonnulF_iD]), mean_std_D_est_err))
        fig_sim_std[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim_std[0].colorbar(p3, ax=ax3)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('estimation error (%)')
        # Fivim*Dstar ----
        p4 = ax4.pcolormesh(std_sim_FDstar_est_err_percent[:, :, i_D], cmap="jet", linewidth=2, edgecolors='white')  #, vmin=vmin_std_FDstar_percent, vmax=vmax_std_FDstar_percent)
        ax4.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax4.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax4.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax4.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax4.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax4.set_xlabel('D$^*$ (mm$^2$/s)')
        ax4.set_ylabel('f$_{IVIM}$ (fraction)')
        ax4.set_title('STD across simulations of f$_{IVIM}$.D$^*$ estimation error\n(Mean = %.1f %%, Mean across all D = %.1f %%)' % (np.mean(std_FDstar_est_err_percent_iD[idx_nonnulF_iD]), mean_std_FDstar_est_err))
        fig_sim_std[0].suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_sim_std[0].colorbar(p4, ax=ax4)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('estimation error (%)')

        plt.subplots_adjust(wspace=0.4, top=0.9, left=0.07, right=0.93)

        fig_sim_std[0].savefig(ofname + '_sim_STD_D' + str(i_D) + '.png')
        plt.close(fig_sim_std[0])



        # plot true params values for sanity check purpose -------------------------------------------------------------
        fig_true, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        # Fivim ----
        p1 = ax1.pcolormesh(true_params_values_reshape[0, :, :, i_D, 0], cmap="jet", linewidth=2, edgecolors='white')
        # plt.tick_params(direction='out')
        ax1.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax1.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax1.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax1.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
        # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
        ax1.set_xlabel('D$^*$ (mm$^2$/s)')
        ax1.set_ylabel('f$_{IVIM}$ (fraction)')
        ax1.set_title('f$_{IVIM}$ (fraction)')
        fig_true.suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_true.colorbar(p1, ax=ax1)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('f$_{IVIM}$')
        # Dstar ----
        p2 = ax2.pcolormesh(true_params_values_reshape[1, :, :, i_D, 0], cmap="jet", linewidth=2, edgecolors='white')
        # plt.tick_params(direction='out')
        ax2.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax2.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax2.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax2.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
        # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
        ax2.set_xlabel('D$^*$ (mm$^2$/s)')
        ax2.set_ylabel('f$_{IVIM}$ (fraction)')
        ax2.set_title('D$^*$ (mm$^2$/s)')
        fig_true.suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_true.colorbar(p2, ax=ax2)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('D$^*$ (mm$^2$/s)')
        # D ----
        p3 = ax3.pcolormesh(true_params_values_reshape[2, :, :, i_D, 0], cmap="jet", linewidth=2, edgecolors='white')
        # plt.tick_params(direction='out')
        ax3.tick_params(axis='both', which='both', length=0)  # remove ticks line
        ax3.set_xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_xticklabels(np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        ax3.set_yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5])
        ax3.set_yticklabels(np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
        # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
        # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
        ax3.set_xlabel('D$^*$ (mm$^2$/s)')
        ax3.set_ylabel('f$_{IVIM}$ (fraction)')
        ax3.set_title('D (mm$^2$/s)')
        fig_true.suptitle('D = %.4e mm$^2$/s' % D_range[i_D])
        cbar = fig_true.colorbar(p3, ax=ax3)
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        cbar.ax.set_ylabel('D (mm$^2$/s)')

        plt.subplots_adjust(wspace=0.5, top=0.85)

        fig_true.savefig(ofname + '_true_D' + str(i_D) + '.png')
        plt.close(fig_true)


    # fake figure to get a big colorbar (DO NOT FORGET TO CHANGE THE vmin_Dr AND vmax_Dr IF YOU NEED Da BOUNDARIES)
    for i_param in range(len(vmax_Dr)):

        # fake figure for colorbar
        fig_fake, ax_fake = plt.subplots(1, 1, figsize=(17, 15))
        if i_param == 2:
            p_fake = ax_fake.pcolormesh(mean_sim_Fivim_est_err_percent[:, :, 0], cmap="jet", linewidth=2, edgecolors='white', vmin=vmin_Dr[i_param], vmax=vmax_Dr[i_param])
            cbar = fig_fake.colorbar(p_fake, ax=ax_fake, aspect=30)
        else:
            p_fake = ax_fake.pcolormesh(mean_sim_Fivim_est_err_percent[:, :, 0], cmap="jet", linewidth=2, edgecolors='white', vmin=vmin_Dr[i_param], vmax=vmax_Dr[i_param], norm=colors.LogNorm())
            cbar = fig_fake.colorbar(p_fake, ax=ax_fake, aspect=30, ticks=LogLocator(subs=range(10)))

        cbar.ax.yaxis.set_offset_position('left')
        cbar.ax.tick_params(labelsize=17, length=29, width=2)
        # fig_fake.savefig(ofname + '_fake_param' + str(i_param) + '.png')
        plt.close(fig_fake)

    print('=== Done ===')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program produces the figures plotting the estimation errors from simulations run with program \"ivim_simu_compute_error_noise\".')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-input', dest='ifname', help='Result file produced by function \"ivim_simu_compute_error_noise\".', type=str, required=True)
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

