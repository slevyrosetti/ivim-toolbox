#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This program computes the required SNR to estimate the product of parameters FivimXDstar within 10% error margins, based on Monte-Carlo simulations.



Created on Mon Jul  8 18:32:11 2019

@author: slevy
"""

import ivim_fitting
import numpy as np
import cPickle as pickle
import math
import time
import argparse
import sys
import warnings
import os


def main(model, ofolder, bvals, condition, snr_init):
    """Main."""

    # set fit parameters
    ivim_fitting.approach = model
    bvals = np.array(map(int, bvals.split(',')))
    s0 = 600.
    n_noise_simu = 10

    # define variations range for Fivim, Dstar and D
    n_sample = 10
    F_range = np.linspace(start=0.01, stop=0.30, num=n_sample)
    Dstar_range = np.linspace(start=3.0e-3, stop=35e-3, num=n_sample)
    D_range = np.array([0.3e-3, 1.5e-3])

    # measure duration
    start_time = time.time()

    # for output
    start_time_str = time.strftime("%y%m%d%H%M%S")
    if not os.path.exists(ofolder):
        os.mkdir(ofolder)
        print "\nDirectory", ofolder, "created.\n"
    else:
        print "\nDirectory", ofolder, "already exists.\n"

    # for each parameter combination, find required SNR
    minSNR = np.zeros((len(F_range), len(Dstar_range), len(D_range)))
    mean_err_for_minSNR = np.zeros((4, len(F_range), len(Dstar_range), len(D_range)))  # F, Dstar, D, F.Dstar
    minSNR_diverging = np.zeros((len(F_range), len(Dstar_range), len(D_range)))
    true_params_values = np.zeros((len(F_range), len(Dstar_range), len(D_range)), dtype=dict)  # true values for F, Dstar, D
    for i_F in range(len(F_range)):  # does not run for Fivim=0 because cannot calculate estimation error in percentage for Fivim=0
        for i_Dstar in range(len(Dstar_range)):
            for i_D in range(len(D_range)):

                # simulate
                signal = ivim_fitting.ivim_1pool_model(bvals, s0, D_range[i_D], F_range[i_F], Dstar_range[i_Dstar])

                # search for the mininum SNR to get at leat 10% error on estimation
                snr = [snr_init]
                opt_snr_found, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err = 0, [], [], np.empty(4)
                while not opt_snr_found:

                    # add noise
                    print('Trying SNR = %d ...' % snr[-1])
                    # if float(s0)/snr[-1] < 1e-12:  # if SNR is too high, simulate perfect signal
                    #     sct.printv('/!\/!\/!\ SNR has grown too much (asking for noise_std = %.9f) ==> adding zero noise (perfect data) for Fivim=%.3f, D*=%.3e, D=%.3e' % (float(s0)/snr[-1], F_range[i_F], Dstar_range[i_Dstar], D_range[i_D]), type='warning')
                    #     S_simu = np.tile(signal, (n_noise_simu, 1))
                    #     minSNR_diverging[i_F, i_Dstar, i_D] = 1  # mark this parameters set
                    # else:
                    S_simu = np.zeros((n_noise_simu, len(signal)))
                    for i_noise_simu in range(n_noise_simu):
                        noise = np.random.normal(0, float(s0)/snr[-1], len(bvals))
                        S_simu[i_noise_simu, :] = signal + noise

                    # fit
                    plots_idx = tuple([np.array(range(n_noise_simu)), np.zeros((n_noise_simu), dtype=int), np.zeros((n_noise_simu), dtype=int)])
                    ivim_fit = ivim_fitting.IVIMfit(bvals=bvals, voxels_values=S_simu, voxels_idx=plots_idx, multithreading=1, model=model, save_plots=False)
                    ivim_fit.run_fit(verbose=0)

                    # compute estimation error on each parameter
                    mean_err = np.empty(4)  # F, Dstar, D, F.Dstar
                    F_est = np.array([simu["Fivim"] for simu in ivim_fit.ivim_metrics_all_voxels])
                    Dstar_est = np.array([simu["Dstar"] for simu in ivim_fit.ivim_metrics_all_voxels])
                    D_est = np.array([simu["D"] for simu in ivim_fit.ivim_metrics_all_voxels])
                    mean_err[0] = 100 * np.mean(np.abs(F_est - F_range[i_F])) / F_range[i_F]
                    mean_err[1] = 100 * np.mean(np.abs(Dstar_est - Dstar_range[i_Dstar])) / Dstar_range[i_Dstar]
                    mean_err[2] = 100 * np.mean(np.abs(D_est - D_range[i_D])) / D_range[i_D]
                    # F.Dstar
                    FDstar_est = np.array([simu["Fivim"]*simu["Dstar"] for simu in ivim_fit.ivim_metrics_all_voxels])
                    FDstar_true = F_range[i_F]*Dstar_range[i_Dstar]
                    mean_err[3] = 100*np.mean(np.abs(FDstar_est - FDstar_true))/FDstar_true
                    print('> Min error on F.D* = '+str(mean_err[3])+'%')

                    # update SNR
                    if condition == 'all':
                        update_snr_fct = update_SNR_v3
                    elif condition == 'FDstar':
                        update_snr_fct = update_SNR_v2
                    else:
                        sys.exit('ERROR: condition unknown.')
                    new_snr, opt_snr_found, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err = update_snr_fct(snr, mean_err, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err)
                    snr.append(new_snr)

                    # # check that SNR does not diverge
                    # if float(s0)/snr[-1] < 1e-12:
                    #     opt_snr_found = 1  # stop search
                    #     minSNR_diverging[i_F, i_Dstar, i_D] = 1  # mark this parameters set
                    #     snr_min_acceptable = np.nan  # set min SNR to NaN

                # save minimum SNR
                if minSNR_diverging[i_F, i_Dstar, i_D]:
                    warnings.warn('/!\/!\/!\ SNR diverging for Fivim=%.3f, D*=%.3e, D=%.3e ==> stopping search for this set of parameters at SNR=%d' % (F_range[i_F], Dstar_range[i_Dstar], D_range[i_D], snr[-1]))
                else:
                    print('==> Found minimum SNR yielding estimation error < 10%%: SNRmin = %d (error on F.D* = %.3f%%)' % (snr_min_acceptable, snr_min_acceptable_err[3]))
                    minSNR[i_F, i_Dstar, i_D] = snr_min_acceptable
                mean_err_for_minSNR[:, i_F, i_Dstar, i_D] = snr_min_acceptable_err[:]
                true_params_values[i_F, i_Dstar, i_D] = {'S0': s0, 'D': D_range[i_D], 'Fivim': F_range[i_F], 'Dstar': Dstar_range[i_Dstar]}
                print('==> %.3f%% done.\n' % (100*float(i_F*n_sample**2 + i_Dstar*n_sample + i_D)/(n_sample**3)))

    # save estimation error and true values
    pickle.dump([minSNR, mean_err_for_minSNR, minSNR_diverging, true_params_values, F_range, Dstar_range, D_range, n_noise_simu], open(ofolder+'/sim_results_'+start_time_str+'.pkl', 'w'))

    print('==> Result file was saved as: ' + ofolder + '/sim_results_' + start_time_str + '.pkl')

    total_elapsed_time = time.time() - start_time
    print('\n=== Done! TOTAL ELAPSED TIME: ' + str(int(round(total_elapsed_time))) + 's\n')


def update_SNR(snr, err, min_acceptable_err=10):
    """
    Kind of binary search for minimum SNR
    :param snr:
    :param err:
    :param stop_flag:
    :return:
    """

    stop_flag = []
    if err <= min_acceptable_err:
        # if estimation error is small then can decrease SNR
        if len(snr) == 1:
            new_snr = snr[-1] / 2.
        else:
            if snr[-2] > snr[-1]:
                new_snr = snr[-1] / 2.
            elif snr[-2] < snr[-1]:
                new_snr = (snr[-2] + snr[-1])/2

    else:
        # if estimation error is too large, then we need do increase SNR
        if len(snr) == 1:
            new_snr = snr[-1] * 2.
        else:
            if snr[-2] > snr[-1]:
                new_snr = (snr[-2] + snr[-1])/2
            elif snr[-2] < snr[-1]:
                new_snr = snr[-1] * 2.

    # if new snr is less than 1 above
    if (np.abs(new_snr - snr[-1]) < 1) and (new_snr > snr[-1]):
            new_snr = math.ceil(new_snr)

    # stop if approximately found
    if round(new_snr, 0) == snr[-1]:
        stop_flag = [snr[-1]]

    return round(new_snr, 0), stop_flag


def update_SNR_v2(snr, err, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err, min_acceptable_err=10):
    """
    Kind of binary search for minimum SNR with recording of min and max bounds.
    """

    opt_snr_found = 0

    # update the new bounds for optimal SNR
    if err[3] <= min_acceptable_err:
        if not snr_min_acceptable:
            snr_min_acceptable = snr[-1]
            snr_min_acceptable_err = err
        else:
            if snr[-1] < snr_min_acceptable:
                snr_min_acceptable = snr[-1]
                snr_min_acceptable_err = err

    elif err[3] > min_acceptable_err:
        if not snr_max_nonacceptable:
            snr_max_nonacceptable = snr[-1]
        else:
            if snr[-1] > snr_max_nonacceptable:
                snr_max_nonacceptable = snr[-1]

    # update SNR to be tested
    if not snr_min_acceptable:
        new_snr = snr_max_nonacceptable * 3.
    elif not snr_max_nonacceptable:
        new_snr = snr_min_acceptable / 2.
    else:
        if np.abs(snr_min_acceptable - snr_max_nonacceptable) <= 1:
            opt_snr_found = 1
            new_snr = snr[-1]
        else:
            new_snr = float(snr_min_acceptable + snr_max_nonacceptable) / 2.

    return round(new_snr, 0), opt_snr_found, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err


def update_SNR_v3(snr, err, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err, min_acceptable_err=10):
    """
    Kind of binary search for minimum SNR with recording of min and max bounds.
    v3: record acceptable SNR if estimation error on all parameters (Fivim, Dstar, D and Fivim.Dstar) is low enough
    """

    opt_snr_found = 0

    # update the new bounds for optimal SNR
    if np.all(err <= min_acceptable_err):
        if not snr_min_acceptable:
            snr_min_acceptable = snr[-1]
            snr_min_acceptable_err = err
        else:
            if snr[-1] < snr_min_acceptable:
                snr_min_acceptable = snr[-1]
                snr_min_acceptable_err = err

    elif np.any(err > min_acceptable_err):
        if not snr_max_nonacceptable:
            snr_max_nonacceptable = snr[-1]
        else:
            if snr[-1] > snr_max_nonacceptable:
                snr_max_nonacceptable = snr[-1]

    # update SNR to be tested
    if not snr_min_acceptable:
        new_snr = snr_max_nonacceptable * 3.
    elif not snr_max_nonacceptable:
        new_snr = snr_min_acceptable / 2.
    else:
        if np.abs(snr_min_acceptable - snr_max_nonacceptable) <= 1:
            opt_snr_found = 1
            new_snr = snr[-1]
        else:
            new_snr = float(snr_min_acceptable + snr_max_nonacceptable) / 2.

    return round(new_snr, 0), opt_snr_found, snr_min_acceptable, snr_max_nonacceptable, snr_min_acceptable_err


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program computes the required SNR to estimate the product of parameters FivimXDstar within 10% error margins, based on Monte-Carlo simulations.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-model', dest='model', help='Fit approach: one-step or two-step.', type=str, required=True)
    requiredArgs.add_argument('-ofolder', dest='ofolder', help="Output directory for results.", type=str, required=True)

    optionalArgs.add_argument('-bval', dest='bvals', help='B-value distribution to fit.', type=str, required=False, default='5,10,15,20,30,50,75,100,125,150,200,250,600,700,800')
    optionalArgs.add_argument('-condition', dest='condition', help='Estimation error condition on F.D* (\"FDstar\") or on all IVIM parameters (\"all\").', type=str, required=False, default='FDstar')
    optionalArgs.add_argument('-snr', dest='snr_init', help='Initial SNR to start search.', type=float, required=False, default=1500.)
    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # run main
    main(model=args.model, ofolder=args.ofolder, bvals=args.bvals, condition=args.condition, snr_init=args.snr_init)

