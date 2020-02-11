#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This program generates data from IVIM signal representation and fit them to get estimation error.


Created on Mon Jul  8 17:38:41 2019

@author: slevy
"""

import ivim_fitting
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
import os


def main(model, ofolder, bvals):
    """Main."""

    # get starting time and create output folder if does not exist yet
    start_time = time.strftime("%y%m%d%H%M%S")
    if not os.path.exists(ofolder):
        os.mkdir(ofolder)
        print "\nDirectory", ofolder, "created.\n"
    else:
        print "\nDirectory", ofolder, "already exists.\n"

    # set fit parameters
    ivim_fitting.approach = model
    bvals = np.array(map(int, bvals.split(',')))
    s0 = 600.

    # define variations range for Fivim, Dstar and D
    n_sample = 10
    F_range = np.linspace(start=0.01, stop=0.30, num=n_sample)
    Dstar_range = np.linspace(start=3.0e-3, stop=35e-3, num=n_sample)
    D_range = np.array([0.2e-3, 0.3e-3, 0.7e-3, 1.0e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2.0e-3, 2.5e-3, 2.9e-3])  #np.linspace(start=1e-4, stop=1e-3, num=n_sample)

    # for each parameter combination, simulate MRI signal
    true_params_values = np.zeros((n_sample**3), dtype=dict)  # true values for F, Dstar, D
    S_simu = np.zeros((n_sample**3, len(bvals)))
    for i_F in range(len(F_range)):
        for i_Dstar in range(len(Dstar_range)):
            for i_D in range(len(D_range)):

                # simulate
                # S_simu[i_F*n_sample**2+i_Dstar*n_sample+i_D, :] = ivim_fitting.ivim_lemke2010_model(bvals, s0, D_range[i_D], F_range[i_F], Dstar_range[i_Dstar], 51.6, 50., 235.)
                S_simu[i_F*n_sample**2+i_Dstar*n_sample+i_D, :] = ivim_fitting.ivim_1pool_model(bvals, s0, D_range[i_D], F_range[i_F], Dstar_range[i_Dstar])
                true_params_values[i_F*n_sample**2+i_Dstar*n_sample+i_D] = {'S0': s0, 'D': D_range[i_D], 'Fivim': F_range[i_F], 'Dstar': Dstar_range[i_Dstar]}

    # fit
    plots_idx = np.where(np.zeros((n_sample, n_sample, n_sample)) == 0)
    ivim_fit = ivim_fitting.IVIMfit(bvals=bvals, voxels_values=S_simu, voxels_idx=plots_idx, multithreading=1, model=model, save_plots=False)
    ivim_fit.run_fit(true_params_values)

    # # add true parameters values on fit plot
    # sct.printv('Add true IVIM parameters values to plots...', type='info')
    # args_func_edit_fit_plot = [(i_simu, ivim_fit, plots_idx, s0, true_params_values) for i_simu in range(n_sample**3)]
    # # pool = multiprocessing.Pool()
    # # pool.map(pool_map_warpper_edit_fit_plot, args_func_edit_fit_plot)
    # # pool.close()
    # # pool.join()
    # map(pool_map_warpper_edit_fit_plot, args_func_edit_fit_plot)
    # sct.printv('...Done.', type='info')

    # # compute estimation error of F*Dstar in percent
    # estimation_err = np.zeros((n_sample**3))
    # estimation_err_percent = np.zeros((n_sample**3))
    # for i_simu in range(n_sample**3):
    #     true_FDstar = true_params_values[i_simu]["Fivim"]*true_params_values[i_simu]["Dstar"]
    #     est_FDstar = ivim_fit.ivim_metrics_all_voxels[i_simu]["Fivim"]*ivim_fit.ivim_metrics_all_voxels[i_simu]["Dstar"]
    #     estimation_err[i_simu] = np.abs(true_FDstar-est_FDstar)
    #     if true_FDstar <= np.spacing(1):
    #         true_FDstar = np.spacing(1)
    #     estimation_err_percent[i_simu] = 100 * np.abs((true_FDstar - est_FDstar) / true_FDstar)
    # estimation_err = np.reshape(estimation_err, (n_sample, n_sample, n_sample))
    # estimation_err_percent = np.reshape(estimation_err_percent, (n_sample, n_sample, n_sample))
    # true_params_values_reshape = np.array(true_params_values, dtype=dict)
    # true_params_values_reshape = np.reshape(true_params_values, (n_sample, n_sample, n_sample))
    #
    # # plot estimation error
    # sct.printv('Mean estimation error on F*Dstar = '+str(round(np.mean(estimation_err), 10))+r' mm$^2$/s', type='info')
    #
    # # plot estimation error
    # sct.printv('Global mean estimation error on F*Dstar = %.4e mm$^2$/s\n\t\t\t\t\t\t\t\t\t= %3f %%' % (np.mean(estimation_err), np.mean(estimation_err_percent[true_FDstar > 0])), type='info')
    #
    # # save maps for each D value
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
    #     plt.savefig(ofolder + 'FDstar_err_D' + str(round(D_range[i_D], 4)) + '.png')
    #     plt.close()
    #
    # # (estimation error in percent)
    # # estimation_err_percent[2, 0, :] = 0
    # vmin_percent, vmax_percent = np.min(estimation_err_percent[true_FDstar > 0]), np.max(estimation_err_percent[true_FDstar > 0])
    # for i_D in range(len(D_range)):
    #     # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3))
    #     plt.figure()
    #     plt.pcolormesh(estimation_err_percent[:, :, i_D], cmap="jet", vmin=vmin_percent, vmax=vmax_percent, linewidth=2, edgecolors='white')
    #     # plt.tick_params(direction='out')
    #     plt.tick_params(axis='both', which='both', length=0)  # remove ticks line
    #     plt.xticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5], np.take(np.round(Dstar_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     plt.yticks([0.5, n_sample / 2 + 0.5, n_sample - 0.5], np.take(np.round(F_range, 3), [0, round(n_sample / 2), n_sample - 1]))
    #     # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(range(n_sample+1)))
    #     plt.xlabel('D$^*$ (mm$^2$/s)')
    #     plt.ylabel('f$_{IVIM}$ (fraction)')
    #     plt.title('D = %.4e mm$^2$/s\nMean estimation error on f$_{IVIM}$.D$^*$ = %.1f %% (%.1f %% for all D)' % (D_range[i_D], np.mean(estimation_err_percent[true_FDstar[:, :, i_D] > 0, i_D]), np.mean(estimation_err_percent[true_FDstar > 0])))
    #     cbar = plt.colorbar()
    #     cbar.ax.yaxis.set_offset_position('left')
    #     cbar.update_ticks()
    #     cbar.ax.set_ylabel('f$_{IVIM}$.D$^*$ estimation error (%)')
    #     plt.savefig(ofolder + 'FDstar_err_percent_D' + str(round(D_range[i_D], 4)) + '.png')
    #     plt.close()

    # save estimation error and true values
    # pickle.dump([estimation_err, true_params_values_reshape], open(ofolder+'/sim_results.pkl', 'w'))
    pickle.dump([ivim_fit.ivim_metrics_all_voxels, true_params_values, F_range, Dstar_range, D_range], open(ofolder+'/sim_results_'+start_time+'.pkl', 'w'))

    print('==> Simulations result file was saved as: '+ofolder+'/sim_results_'+start_time+'.pkl')


def edit_fit_plot(i_simu, ivim_fit, plots_idx, s0, true_params_values):
    """
    Load PNG plot for each simulatio and add the true parameters.
    :param i_simu: index of the simulation dataset
    :return:
    """

    plot = mpimg.imread(ivim_fit.ofit_dir + "/" + ivim_fit.plot_dir + "/z{}_y{}_x{}.png".format(plots_idx[2][i_simu], plots_idx[1][i_simu], plots_idx[0][i_simu]))
    fig = plt.imshow(plot)
    params_true_value_to_display = "TRUE VALUES\nS$_0$=%.1f\nf$_{IVIM}$=%.3f\nD$^*$=%.3e mm$^2$/s\nD=%.3e mm$^2$/s" % (s0, true_params_values[0, i_simu], true_params_values[1, i_simu],true_params_values[2, i_simu])
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
    fig.axes.text(.9, .10,
                  params_true_value_to_display,
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=fig.axes.transAxes,
                  size=8,
                  bbox=bbox_props)
    fig.axes.set_axis_off()
    fig.figure.savefig(ivim_fit.ofit_dir + "/" + ivim_fit.plot_dir + "/z{}_y{}_x{}.png".format(plots_idx[2][i_simu], plots_idx[1][i_simu], plots_idx[0][i_simu]), transparent=False, bbox_inches='tight', pad_inches=-0.15, dpi=250)
    plt.close()

def pool_map_warpper_edit_fit_plot(args_func):
    return edit_fit_plot(*args_func)


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program generates data from IVIM signal representation and fits them to get estimation error.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-model', dest='model', help='Fit approach: one-step or two-step.', type=str, required=True)
    requiredArgs.add_argument('-ofolder', dest='ofolder', help="Output directory for results.", type=str, required=True)

    optionalArgs.add_argument('-bval', dest='bvals', help='B-value distribution to fit.', type=str, required=False, default='5,10,15,20,30,50,75,100,125,150,200,250,600,700,800')
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
    main(model=args.model, ofolder=args.ofolder, bvals=args.bvals)

