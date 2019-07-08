#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This program generates data from IVIM representation, add Gaussian noise to match selected SNR and fit those data to get fit error.

Created on Mon Jul  8 16:03:43 2019

@author: slevy
"""

import ivim_fitting
import numpy as np
import cPickle as pickle
import argparse
import time
import os

def main(model, snr, ofolder, bvals):
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
    n_noise_simu = 1000

    # define variations range for Fivim, Dstar and D
    n_sample = 10
    F_range = np.linspace(start=0.01, stop=0.30, num=n_sample)
    Dstar_range = np.linspace(start=3.0e-3, stop=35e-3, num=n_sample)
    # D_range = np.array([0.2e-3, 0.4e-3, 0.7e-3, 1.0e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2.0e-3, 2.5e-3, 2.9e-3])  #np.linspace(start=1e-4, stop=1e-3, num=n_sample)
    D_range = np.array([0.3e-3, 1.5e-3])

    # for each parameter combination, simulate MRI signal
    true_params_values = np.zeros((len(F_range) * len(Dstar_range) * len(D_range) * n_noise_simu), dtype=dict)  # true values for F, Dstar, D
    S_simu = np.zeros((len(F_range) * len(Dstar_range) * len(D_range) * n_noise_simu, len(bvals)))
    for i_F in range(len(F_range)):
        for i_Dstar in range(len(Dstar_range)):
            for i_D in range(len(D_range)):

                # simulate
                # signal = ivim_fitting.ivim_lemke2010_model(bvals, s0, D_range[i_D], F_range[i_F], Dstar_range[i_Dstar], 51.6, 50., 235.)
                signal = ivim_fitting.ivim_1pool_model(bvals, s0, D_range[i_D], F_range[i_F], Dstar_range[i_Dstar])

                # add noise
                for i_noise_simu in range(n_noise_simu):
                    i_flat = i_F * len(Dstar_range) * len(D_range) * n_noise_simu + i_Dstar * len(D_range) * n_noise_simu + i_D * n_noise_simu + i_noise_simu

                    noise = np.random.normal(0, s0/snr, len(bvals))
                    S_simu[i_flat, :] = signal + noise
                    true_params_values[i_flat] = {'S0': s0, 'D': D_range[i_D], 'Fivim': F_range[i_F], 'Dstar': Dstar_range[i_Dstar]}

    # fit
    plots_idx = tuple([np.array(range(len(F_range)*len(Dstar_range)*len(D_range)*n_noise_simu)), np.zeros((len(F_range)*len(Dstar_range)*len(D_range)*n_noise_simu), dtype=int), np.zeros((len(F_range)*len(Dstar_range)*len(D_range)*n_noise_simu), dtype=int)])
    ivim_fit = ivim_fitting.IVIMfit(bvals=bvals, voxels_values=S_simu, voxels_idx=plots_idx, multithreading=1, model=model, save_plots=False)
    ivim_fit.run_fit(true_params_values)

    pickle.dump([ivim_fit.ivim_metrics_all_voxels, true_params_values, F_range, Dstar_range, D_range, n_noise_simu, snr], open(ofolder+'/sim_results_'+start_time+'.pkl', 'w'))

    print('==> Simulations result file was saved as: '+ofolder+'/sim_results_'+start_time+'.pkl')


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program generates data from IVIM representation, adds random Gaussian noise to match SNR specified by user, fits data to get IVIM params back and quantifies mean parameters estimation error across noise draws.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-model', dest='model', help='Fit approach: one-step or two-step.', type=str, required=True)
    requiredArgs.add_argument('-snr', dest='snr', help='Simulated SNR.', type=float, required=True)
    requiredArgs.add_argument('-ofolder', dest='ofolder', help="Output directory for results.", type=str, required=True)

    optionalArgs.add_argument('-bval', dest='bvals', help='B-value distribution to fit.', type=str, required=False, default='5,10,15,20,30,50,75,100,125,150,200,250,600,700,800')
    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # run main
    main(model=args.model, snr=args.snr, ofolder=args.ofolder, bvals=args.bvals)


