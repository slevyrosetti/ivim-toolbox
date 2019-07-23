#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Compute IVIM parameters maps fitting IVIM model voxel-wise and using parallel threading.


Created on Tue Jul  4 17:45:43 2017

@author: slevy
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
from lmfit import Model
import os
import nibabel as nib
import argparse


class IVIMfit:
    def __init__(self, bvals, voxels_values, voxels_idx, model, ofit_dir='', multithreading=True, save_plots=True):
        self.bvals = bvals
        self.voxels_values = voxels_values
        self.voxels_idx = voxels_idx
        self.ofit_dir = ofit_dir
        self.multithreading = multithreading
        self.ivim_metrics_all_voxels = {}
        self.plot_dir = ''
        self.save_plots = save_plots
        self.model = model

    def run_fit(self, true_params=[], verbose=1):
        """Run defined fit."""



        if self.save_plots:
            # create dedicated plot directory
            self.plot_dir = "%s_plots" % time.strftime("%y%m%d%H%M%S")
            os.mkdir(self.ofit_dir + "/" + self.plot_dir)

        n_vox_to_fit = self.voxels_values.shape[0]
        if verbose:
            print('=== ' + str(n_vox_to_fit) + ' voxels to be fitted with representation: '+self.model+' ===')

        # measure duration
        start_time = time.time()

        if true_params == []:
            fit_func_args = [(vox_val,
                              self.bvals,
                              self.ofit_dir + "/" + self.plot_dir + "/z{:d}_y{:d}_x{:d}.png".format(self.voxels_idx[2][vox], self.voxels_idx[1][vox], self.voxels_idx[0][vox]),
                              n_vox_to_fit,
                              {},
                              verbose) for vox, vox_val in enumerate(self.voxels_values)]
        else:
            fit_func_args = [(vox_val,
                              self.bvals,
                              self.ofit_dir + "/" + self.plot_dir + "/z{:d}_y{:d}_x{:d}.png".format(self.voxels_idx[2][vox], self.voxels_idx[1][vox], self.voxels_idx[0][vox]),
                              n_vox_to_fit,
                              true_params[vox],
                              verbose) for vox, vox_val in enumerate(self.voxels_values)]

        if self.multithreading:
            # ---- on all available workers ----
            pool = multiprocessing.Pool()
            # set each matching item into a tuple
            self.ivim_metrics_all_voxels = pool.map(fit_warpper, fit_func_args)
            pool.close()
            pool.join()
        else:
            # ---- on one worker ----
            # ivim_params_all_vox = map(fit_func, iter(values_voxels_to_fit), [bvals] * n_vox_to_fit, [ofolder + "/" + plot_dir+"/z{2}_y{1}_x{0}.png".format(*vox_coord) for vox_coord in np.array(idx_voxels_to_fit).T], [n_vox_to_fit] * n_vox_to_fit)
            self.ivim_metrics_all_voxels = map(fit_warpper, fit_func_args)

        elapsed_time = time.time() - start_time
        if verbose:
            print('\nFitting done! Elapsed time: ' + str(int(round(elapsed_time))) + 's\n')


def main(dwi_fname, bval_fname, mask_fname, model, ofolder, multithreading):
    """Main."""

    # define the fit approach as global variable to avoid having to define a fit_warpper function for each fit approach (needed in pool.map for multithreading)
    global approach
    approach = model

    # load data
    dwi = nib.load(dwi_fname).get_data()
    bvals = np.loadtxt(bval_fname, delimiter=None)
    mask_nii = nib.load(mask_fname)
    mask = mask_nii.get_data()
    
    # initialize outputs
    if not os.path.exists(ofolder):
        os.mkdir(ofolder)
        print "\nDirectory", ofolder, "created.\n"
    else:
        print "\nDirectory", ofolder, "already exists.\n"

    # prepare IVIM fit object
    ivim_fit = IVIMfit(bvals=bvals,
                       voxels_values=dwi[mask > 0, :],
                       voxels_idx=np.where(mask > 0),
                       ofit_dir=ofolder,
                       multithreading=multithreading,
                       model=model)
    # run fit
    ivim_fit.run_fit()

    # save params values to arrays
    S0_map = np.zeros(dwi.shape[0:3])
    D_map = np.zeros(dwi.shape[0:3])
    FivimXDstar_map = np.zeros(dwi.shape[0:3])
    AIC_map = np.zeros(dwi.shape[0:3])
    R2_map = np.zeros(dwi.shape[0:3])
    exception_map = np.zeros(dwi.shape[0:3])
    S0_map[ivim_fit.voxels_idx] = [voxel["S0"] for voxel in ivim_fit.ivim_metrics_all_voxels]
    D_map[ivim_fit.voxels_idx] = [voxel["D"] for voxel in ivim_fit.ivim_metrics_all_voxels]
    AIC_map[ivim_fit.voxels_idx] = [voxel["AIC"] for voxel in ivim_fit.ivim_metrics_all_voxels]
    R2_map[ivim_fit.voxels_idx] = [voxel["R2"] for voxel in ivim_fit.ivim_metrics_all_voxels]
    exception_map[ivim_fit.voxels_idx] = [voxel["exception"] for voxel in ivim_fit.ivim_metrics_all_voxels]
    if model != 'FivimXDstar':
        Fivim_map = np.zeros(dwi.shape[0:3])
        Dstar_map = np.zeros(dwi.shape[0:3])
        Fivim_map[ivim_fit.voxels_idx] = [voxel["Fivim"] for voxel in ivim_fit.ivim_metrics_all_voxels]
        Dstar_map[ivim_fit.voxels_idx] = [voxel["Dstar"] for voxel in ivim_fit.ivim_metrics_all_voxels]
        FivimXDstar_map = np.multiply(Fivim_map, Dstar_map)
    else:
        FivimXDstar_map[ivim_fit.voxels_idx] = [voxel["FivimXDstar"] for voxel in ivim_fit.ivim_metrics_all_voxels]
    if model == 'combine':
        S0init_map = np.zeros(dwi.shape[0:3])
        Dinit_map = np.zeros(dwi.shape[0:3])
        S0init_map[ivim_fit.voxels_idx] = [voxel["S0init"] for voxel in ivim_fit.ivim_metrics_all_voxels]
        Dinit_map[ivim_fit.voxels_idx] = [voxel["Dinit"] for voxel in ivim_fit.ivim_metrics_all_voxels]

    # save as NIFTI images
    S0_map_nii = nib.Nifti1Image(S0_map.copy(), mask_nii.affine, mask_nii.header); nib.save(S0_map_nii, ofolder+"/S0_map.nii.gz")
    D_map_nii = nib.Nifti1Image(D_map.copy(), mask_nii.affine, mask_nii.header); nib.save(D_map_nii, ofolder+"/D_map.nii.gz")
    FivimXDstar_map_nii = nib.Nifti1Image(FivimXDstar_map.copy(), mask_nii.affine, mask_nii.header); nib.save(FivimXDstar_map_nii, ofolder+"/FivimXDstar_map.nii.gz")
    AIC_map_nii = nib.Nifti1Image(AIC_map.copy(), mask_nii.affine, mask_nii.header); nib.save(AIC_map_nii, ofolder+"/AIC_map.nii.gz")
    R2_map_nii = nib.Nifti1Image(R2_map.copy(), mask_nii.affine, mask_nii.header); nib.save(R2_map_nii, ofolder+"/R2_map.nii.gz")
    exception_map_nii = nib.Nifti1Image(exception_map.copy(), mask_nii.affine, mask_nii.header); nib.save(exception_map_nii, ofolder+"/exception_map.nii.gz")
    if model != 'FivimXDstar':
        Fivim_map_nii = nib.Nifti1Image(Fivim_map.copy(), mask_nii.affine, mask_nii.header); nib.save(Fivim_map_nii, ofolder + "/Fivim_map.nii.gz")
        Dstar_map_nii = nib.Nifti1Image(Dstar_map.copy(), mask_nii.affine, mask_nii.header); nib.save(Dstar_map_nii, ofolder + "/Dstar_map.nii.gz")
    if model == '1shot_initD_noise':
        noise_map = np.zeros(dwi.shape[0:3])
        noise_map[ivim_fit.voxels_idx] = [voxel["noise"] for voxel in ivim_fit.ivim_metrics_all_voxels]
        noise_map_nii = nib.Nifti1Image(noise_map.copy(), mask_nii.affine, mask_nii.header); nib.save(noise_map_nii, ofolder + "/noise_map.nii.gz")
    if model == 'combine':
        S0init_map_nii = nib.Nifti1Image(S0init_map.copy(), mask_nii.affine, mask_nii.header); nib.save(S0init_map_nii, ofolder + "/S0init_map.nii.gz")
        Dinit_map_nii = nib.Nifti1Image(Dinit_map.copy(), mask_nii.affine, mask_nii.header); nib.save(Dinit_map_nii, ofolder + "/Dinit_map.nii.gz")

    print('==> Results are available in folder: '+ofolder)


def D_log_representation(x, lnS0, D):
    """Logarithm representation of the diffusion-weighted signal decay"""
    return lnS0 - x*D

def ivim_1pool_model(x, S0, D, Fivim, Dstar):
    """1-pool IVIM representation: ivim_1pool_model(x, amp, cen, wid)"""
    return S0 * np.exp(-x*D) * (Fivim*np.exp(-x*Dstar) + 1 - Fivim)

def ivim_1pool_model_with_noise(x, S0, D, Fivim, Dstar, noise):
    """1-pool IVIM representation: ivim_1pool_model(x, amp, cen, wid)"""
    return np.sqrt(np.square(S0 * np.exp(-x*D) * (Fivim*np.exp(-x*Dstar) + 1 - Fivim)) + noise**2)

def ivim_lemke2010_model(x, S0, D, Fivim, Dstar, TE, T2tiss, T2bl):
    """IVIM representation from Lemke et al., MRM 2010: ivim_lemke2010_model(x, amp, cen, wid)"""
    # return S0 * ( (1 - Fivim)*(1 - np.exp(-TR/T1tiss))*np.exp(-TE/T2tiss - x*D) + Fivim*(1 - np.exp(-TR/T1bl))*np.exp(-TE/T2bl - x*(D + Dstar)) ) / ( (1 - Fivim)*np.exp(-TE/T2tiss)*(1 - np.exp(TR/T1tiss)) + Fivim*np.exp(-TE/T2bl)*(1 - np.exp(-TR/T1bl)) )
    # remove terms related to TR as TR was long
    return S0 * ( (1 - Fivim)*np.exp(-TE/T2tiss - x*D) + Fivim*np.exp(-TE/T2bl - x*(D + Dstar)) ) / ( (1 - Fivim)*np.exp(-TE/T2tiss) + Fivim*np.exp(-TE/T2bl) )

def kurtosis_representation(x, S0, ADC, K):
    """Taylor expansion of the diffusion signal: """
    return S0 * np.exp(-x*ADC + ((x*ADC)**2)*K/6)


def get_r2(fit_res):
    """Compute the coefficient of determination of a fit model."

    :param fit_res: ModelResult structure (from lmfit) obtained after performing fit
    :return: R-squared of fit
    """
    sum_squared_error = np.sum(np.square(fit_res.residual))
    sum_squared_deviation_from_mean = np.sum(np.square(fit_res.data - np.mean(fit_res.data)))
    return 1. - sum_squared_error / sum_squared_deviation_from_mean

def plot_fit(bvals, S, fit_res):
    """plot final fit"""

    font = {'size': 20}
    plt.rc('font', **font)
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    plt.title('Model: ' + fit_res.model.name[6:-1] + ' - Algo: ' + fit_res.method + '\n')
    xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
    ax.plot(bvals, np.log(S), color='b', linestyle='', marker='.', markersize=8, label='data')
    ax.plot(xwide, np.log(fit_res.eval(x=xwide)), color='r', linestyle='-', linewidth=1, label='final fit')
    ax.grid(which='major', linestyle=':', alpha=0.9)
    ax.grid(which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()
    ax.legend(loc=1, prop={'size': 15})
    plt.xlabel('b-value (s/mm$^2$)', fontsize=24)
    plt.ylabel('ln(S)', fontsize=24)
    ax.set_xlim(xmin=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # add annotations
    if fit_res.approach == '1shot_initD_noise':
        params_value_to_display = "S$_0$=%.1f\nf$_{IVIM}$=%.3f\nD$^*$=%.3e mm$^2$/s\nD=%.3e mm$^2$/s\nNoise=%.1f\nFit stats\n   Chi$^2$=%.1f\n   AIC=%.3f\n   R$^2$=%.3f" % (
        fit_res.params["S0"].value, fit_res.params["Fivim"].value, fit_res.params["Dstar"].value,
        fit_res.params["D"].value, fit_res.params["noise"].value, fit_res.chisqr, fit_res.aic, get_r2(fit_res))
    elif fit_res.approach == 'FivimXDstar':
        params_value_to_display = "S$_0$=%.1f\nADC=%.3e mm$^2$/s\nK=%.3e/s\nFit stats\n   Chi$^2$=%.1f\n   AIC=%.3f\n   R$^2$=%.3f" % (
        fit_res.params["S0"].value, fit_res.params["ADC"].value, fit_res.params["K"].value, fit_res.chisqr,
        fit_res.aic, get_r2(fit_res))
    else:
        params_value_to_display = "S$_0$=%.1f\nf$_{IVIM}$=%.3f\nD$^*$=%.3e mm$^2$/s\nD=%.3e mm$^2$/s\nFit stats\n   Chi$^2$=%.1f\n   AIC=%.3f\n   R$^2$=%.3f" % (
        fit_res.params["S0"].value, fit_res.params["Fivim"].value, fit_res.params["Dstar"].value,
        fit_res.params["D"].value, fit_res.chisqr, fit_res.aic, get_r2(fit_res))
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(.01, .01,
            params_value_to_display,
            transform=ax.transAxes,
            size=15,
            bbox=bbox_props)

    return ax


def fit_1pool_separate(S, bvals, oplot_fname, n_vox_to_fit=1, bval_thr=500):
    """
    Fit IVIM 1-pool model: S0*( Fivim*exp(-x*Dstar) + 1-Fivim )*exp(-x*D)
    :param vox_coord: coordinates of the voxel to be fit
    :param dwi_data: diffusion-weighted 4D data (x, y, z, b-values)
    :param bvals: b-values acquired
    :param mask: binary mask defining the voxels to fit
    :param ofolder_plot: path of the output folder for plots
    :return:
    """

    ivim_params = {}

    # fit high b-values to get a first estimate of D
    p_highb = np.poly1d(np.polyfit(bvals[bvals >= bval_thr], np.log(S[bvals >= bval_thr]), 1))
    ivim_params["D"] = -p_highb.c[0]

    # fit S0, Fivim, D*
    ivim_model = Model(ivim_1pool_model)
    # set params initial values and bounds
    ivim_model.set_param_hint('S0', value=S[0], min=0, max=10 * S[0])
    ivim_model.set_param_hint('Fivim', value=.05, min=0, max=.99)
    ivim_model.set_param_hint('Dstar', value=0.01, min=0, max=.9)
    ivim_model.set_param_hint('D', value=ivim_params["D"], vary=False)  # fix D
    fitting_params = ivim_model.make_params()
    # run fit algo
    fit_res = ivim_model.fit(S, x=bvals, params=fitting_params)

    # save fit params
    ivim_params["S0"] = fit_res.params["S0"].value
    ivim_params["Fivim"] = fit_res.params["Fivim"].value
    ivim_params["Dstar"] = fit_res.params["Dstar"].value

    # plot and save fit
    fit_res.approach = '1pool_separate'
    ax = plot_fit(bvals, S, fit_res)
    ax.plot(np.linspace(0, np.max(bvals), np.max(bvals) * 2),
            -ivim_params["D"] * np.linspace(0, np.max(bvals), np.max(bvals) * 2) + p_highb.c[1], color='orange',
            linestyle='-', linewidth=1, label='fit b$\leq$' + str(bval_thr))
    plt.savefig(oplot_fname)
    plt.close()
    # plot final fit
    # fit_res.plot()
    # plt.savefig(oplot_fname)
    # plt.close()
    font = {'size': 20}
    plt.rc('font', **font)
    plt.figure(oplot_fname, figsize=(12, 9))
    ax = plt.gca()
    plt.title('Signal fit using biexponentional IVIM model\n')
    xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
    ax.plot(bvals, np.log(S), color='b', linestyle='', marker='.', markersize=8, label='data')
    ax.plot(xwide, -ivim_params["D"] * xwide + p_highb.c[1], color='orange', linestyle='-', linewidth=1,
            label='fit b$\leq$' + str(bval_thr))
    ax.plot(xwide, np.log(ivim_params["S0"] * np.exp(-xwide * ivim_params["D"]) * (
            ivim_params["Fivim"] * np.exp(-xwide * ivim_params["Dstar"]) + 1 - ivim_params["Fivim"])),
            color='r', linestyle='-', linewidth=1, label='fit all b')
    ax.grid(which='major', linestyle=':', alpha=0.9)
    ax.grid(which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()
    ax.legend()
    plt.xlabel('b-value (s/mm$^2$)', fontsize=24)
    plt.ylabel('ln(S)', fontsize=24)
    ax.set_xlim(xmin=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(oplot_fname)
    plt.close()

    # display progress by counting number of plots in directory
    plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
    n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
    print(str(100 * n_voxel_done / n_vox_to_fit) + '% of voxels done.')

    return ivim_params

def fit_1pool_1shot(S, bvals, oplot_fname, n_vox_to_fit=1):
    """
    Fit IVIM 1-pool model: S0*( Fivim*exp(-x*Dstar) + 1-Fivim )*exp(-x*D)
    :param vox_coord: coordinates of the voxel to be fit
    :param dwi_data: diffusion-weighted 4D data (x, y, z, b-values)
    :param bvals: b-values acquired
    :param mask: binary mask defining the voxels to fit
    :param ofolder_plot: path of the output folder for plots
    :return:
    """

    ivim_params = {}

    # fit S0, Fivim, D*
    ivim_model = Model(ivim_1pool_model)
    # set params initial values and bounds
    ivim_model.set_param_hint('S0', value=S[0], min=0, max=10 * S[0])
    ivim_model.set_param_hint('Fivim', value=.02, min=0, max=.99)
    ivim_model.set_param_hint('Dstar', value=1e-3, min=0, max=1e-1)
    ivim_model.set_param_hint('D', value=1e-4, min=0, max=1e-3)
    fitting_params = ivim_model.make_params()
    # run fit algo
    fit_res = ivim_model.fit(S, x=bvals, params=fitting_params, method='cg')

    # save fit params
    ivim_params["S0"] = fit_res.params["S0"].value
    ivim_params["Fivim"] = fit_res.params["Fivim"].value
    ivim_params["Dstar"] = fit_res.params["Dstar"].value
    ivim_params["D"] = fit_res.params["D"].value

    # plot and save fit
    fit_res.approach = '1pool_1shot'
    ax = plot_fit(bvals, S, fit_res)
    plt.savefig(oplot_fname)
    plt.close()

    # display progress by counting number of plots in directory
    plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
    n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
    print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')

    return ivim_params

def fit_lemke2010(S, bvals, oplot_fname, n_vox_to_fit=1):
    """
    Fit IVIM 1-pool model: S0*( Fivim*exp(-x*Dstar) + 1-Fivim )*exp(-x*D)
    :param vox_coord: coordinates of the voxel to be fit
    :param dwi_data: diffusion-weighted 4D data (x, y, z, b-values)
    :param bvals: b-values acquired
    :param mask: binary mask defining the voxels to fit
    :param ofolder_plot: path of the output folder for plots
    :return:
    """

    ivim_params = {}

    # fit S0, Fivim, D*
    ivim_model = Model(ivim_lemke2010_model)
    # set params initial values and bounds
    ivim_model.set_param_hint('S0', value=S[0])  # , min=0, max=10*S[0])
    ivim_model.set_param_hint('Fivim', value=.05)  # , min=0, max=.99)
    ivim_model.set_param_hint('Dstar', value=1e-3)  # , min=0, max=1e-1)
    ivim_model.set_param_hint('D', value=3e-4)  # , min=0, max=1e-3)
    # set fixed values
    # ivim_model.set_param_hint('TR', value=3300, vary=False)  # fix TR
    ivim_model.set_param_hint('TE', value=51.6, vary=False)  # fix TE
    # ivim_model.set_param_hint('T1tiss', value=np.mean([1313, 1182]), vary=False)  # T1 in spinal cord from Massire et al., NeuroImage 2016 (mean T1 across GM and WM at 7T)
    ivim_model.set_param_hint('T2tiss', value=50.,
                              vary=False)  # T2 in spinal cord from Massire et al., Proceedings ISMRM 2016, abstract #1130
    # ivim_model.set_param_hint('T1bl', value=2100, vary=False)  # T1 in blood at 7T (ms) from Zhang et al., MRM 2013
    ivim_model.set_param_hint('T2bl', value=235.,
                              vary=False)  # T2 in blood (value linearly interpolated from 1.5 et 3T values)
    fitting_params = ivim_model.make_params()
    # run fit algo
    fit_res = ivim_model.fit(S, x=bvals, params=fitting_params, method='bfgs', nan_policy='propagate')

    # save fit params
    ivim_params["S0"] = fit_res.params["S0"].value
    ivim_params["Fivim"] = fit_res.params["Fivim"].value
    ivim_params["D"] = fit_res.params["D"].value
    ivim_params["Dstar"] = fit_res.params["Dstar"].value

    # plot and save fit
    fit_res.approach = 'lemke2010'
    ax = plot_fit(bvals, S, fit_res)
    plt.savefig(oplot_fname)
    plt.close()

    # display progress by counting number of plots in directory
    plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
    n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
    print(str(100 * n_voxel_done / n_vox_to_fit) + '% of voxels done.')

    return ivim_params


def fit_2shots(S, bvals, oplot_fname, n_vox_to_fit=1, true_params={}, verbose=1, bval_thr=500):
    """
    v2: adapted bounds for D and D* to adapt to real data and D fixed to the initial value resulting from the fit of
    b-values >= bval_thr only.
    """

    ivim_params = {"exception": 0}

    try:
        # 1) fit high b-values to get a first estimate of D
        p_highb = np.poly1d(np.polyfit(bvals[bval_thr <= bvals], np.log(S[bval_thr <= bvals]), 1))
        # ivim_params["Dinit"] = -p_highb.c[0]
        # ivim_params["S0init"] = np.exp(p_highb.c[1])
        D_representation = Model(D_log_representation)
        D_representation.set_param_hint('D', value=-p_highb.c[0], min=0.15e-3, max=4.5e-3)
        D_representation.set_param_hint('lnS0', value=p_highb.c[1], min=0.5 * p_highb.c[1],
                                        max=1.5 * p_highb.c[1])
        D_fit_contraints = D_representation.make_params()
        D_fit = D_representation.fit(np.log(S[bval_thr <= bvals]), x=bvals[bval_thr <= bvals],
                                     params=D_fit_contraints, method='cg')
        ivim_params["Dinit"] = D_fit.params["D"].value
        ivim_params["S0(1-f)"] = np.exp(D_fit.params["lnS0"].value)

        # 2) fit S0, Fivim, D* and D with biexponential model using the previously determined D as initial value
        ivim_model = Model(ivim_1pool_model)
        ivim_model.set_param_hint('S0', value=1.2 * ivim_params["S0(1-f)"], min=0.7 * ivim_params["S0(1-f)"],
                                  max=1.7 * ivim_params["S0(1-f)"])
        ivim_model.set_param_hint('Fivim', value=0.1, min=0, max=0.35)
        ivim_model.set_param_hint('Dstar', value=(ivim_params["Dinit"] + 35.5e-3) / 2.,
                                  min=ivim_params["Dinit"],
                                  max=35.5e-3)
        # ivim_model.set_param_hint('D', value=ivim_params["Dinit"],  min=0.2e-3, max=ivim_params["Dinit"]*1.02)
        ivim_model.set_param_hint('D', value=ivim_params["Dinit"], vary=False)  # fix D to Dinit
        ivim_model_constraints = ivim_model.make_params()
        # run fit algo
        fit_res_DE = ivim_model.fit(S, x=bvals, params=ivim_model_constraints, method='differential_evolution')
        # fit_res = ivim_model.fit(S[bvals >= 25], x=bvals[bvals >= 25], params=fitting_params, method='cg')  # try to fit only b-values >=25

        # 3) fit S0, Fivim, D* and D with biexponential model using the previously determined D as initial value
        ivim_model = Model(ivim_1pool_model)
        ivim_model.set_param_hint('S0', value=fit_res_DE.params["S0"].value,
                                  min=0.95 * fit_res_DE.params["S0"].value,
                                  max=1.05 * fit_res_DE.params["S0"].value)
        if fit_res_DE.params["Fivim"].value <= 1e-12:
            ivim_model.set_param_hint('Fivim', value=fit_res_DE.params["Fivim"].value, min=0, max=0.05)
        else:
            ivim_model.set_param_hint('Fivim', value=fit_res_DE.params["Fivim"].value,
                                      min=0.95 * fit_res_DE.params["Fivim"].value,
                                      max=1.05 * fit_res_DE.params["Fivim"].value)
        ivim_model.set_param_hint('Dstar', value=fit_res_DE.params["Dstar"].value,
                                  min=0.95 * fit_res_DE.params["Dstar"].value,
                                  max=1.05 * fit_res_DE.params["Dstar"].value)
        # ivim_model.set_param_hint('D', value=fit_res_DE.params["D"].value, min=0.95*fit_res_DE.params["D"].value, max=1.05*fit_res_DE.params["D"].value)
        ivim_model.set_param_hint('D', value=ivim_params["Dinit"], vary=False)  # fix D to Dinit
        ivim_model_constraints = ivim_model.make_params()
        # run fit algo
        fit_res = ivim_model.fit(S, x=bvals, params=ivim_model_constraints, method='differential_evolution')
        # fit_res = ivim_model.fit(S[bvals >= 25], x=bvals[bvals >= 25], params=fitting_params, method='cg')  # try to fit only b-values >=25

        # save fit params
        ivim_params["S0"] = fit_res.params["S0"].value
        ivim_params["Fivim"] = fit_res.params["Fivim"].value
        ivim_params["D"] = fit_res.params["D"].value
        ivim_params["Dstar"] = fit_res.params["Dstar"].value
        ivim_params["AIC"] = fit_res.aic
        ivim_params["R2"] = get_r2(fit_res)

        fit_res.approach = '1shot_initD_v2'
        plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
        if plot_dir[:2] != '//':  # BECAREFUL: THIS WON'T WORK IF THE SELECTED OUTPUT FOLDER IS '/'
            # plot and save fit
            ax = plot_fit(bvals, S, fit_res)
            xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
            ax.plot(xwide, -ivim_params["Dinit"] * xwide + np.log(ivim_params["S0(1-f)"]), color='orange',
                    linestyle='-', linewidth=1, label='D and S0 initialization')
            ax.plot(xwide, -0.2e-3 * ivim_params["Dinit"] * xwide + 1.7 * np.log(ivim_params["S0(1-f)"]),
                    color='orange', linestyle='--', linewidth=1, label='Lower bound for D and S0')
            ax.plot(xwide, -ivim_params["Dinit"] * 1.2 * xwide + 0.7 * np.log(ivim_params["S0(1-f)"]),
                    color='orange', linestyle='--', linewidth=1, label='Upper bound for D and S0')
            ax.legend(loc=1, prop={'size': 15})
            if true_params != {}:
                true_params_value_to_display = "TRUE VALUES\nS$_0$=%.1f\nf$_{IVIM}$=%.3f\nD$^*$=%.3e mm$^2$/s\nD=%.3e mm$^2$/s" % (
                    true_params['S0'], true_params['Fivim'], true_params['Dstar'], true_params['D'])
                bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
                ax.text(.92, .01,
                        true_params_value_to_display,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        size=15,
                        bbox=bbox_props)
            # ivim_params["fig"] = plt.gcf()

            # save figure and display progress by counting number of plots in directory
            plt.savefig(oplot_fname)
            plt.close()
            n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])

        else:
            plot_filename = oplot_fname.split('/')[-1].split('.')[0]
            z_idx, y_idx, x_idx = int(plot_filename.split('z')[1].split('_')[0]), int(
                plot_filename.split('y')[1].split('_')[0]), int(plot_filename.split('x')[1])
            dim_cube = pow(n_vox_to_fit, 1. / 3)
            n_voxel_done = z_idx * dim_cube * dim_cube + y_idx * dim_cube + x_idx

        if verbose:
            print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')


    except ValueError, err_detail:
        print('/!\\/!\\/!\\ VALUE ERROR /!\\/!\\/!\\: ' + str(err_detail))
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0(1-f)"] = 0
        ivim_params["Dinit"] = 0
        ivim_params["S0"] = 0
        ivim_params["Fivim"] = 0
        ivim_params["D"] = 0
        ivim_params["Dstar"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    return ivim_params

def fit_1shot_initD(S, bvals, oplot_fname, n_vox_to_fit=1, true_params={}, bval_thr=500):
    """
    Fit IVIM 1-pool model: S0*( Fivim*exp(-x*Dstar) + 1-Fivim )*exp(-x*D)
    """

    ivim_params = {"exception": 0}

    try:
        # 1) fit high b-values to get a first estimate of D
        p_highb = np.poly1d(np.polyfit(bvals[bval_thr <= bvals], np.log(S[bval_thr <= bvals]), 1))
        # ivim_params["Dinit"] = -p_highb.c[0]
        # ivim_params["S0init"] = np.exp(p_highb.c[1])
        D_representation = Model(D_log_representation)
        D_representation.set_param_hint('D', value=-p_highb.c[0], min=0.2e-3, max=2.95e-3)
        D_representation.set_param_hint('lnS0', value=p_highb.c[1], min=0.5 * p_highb.c[1],
                                        max=1.5 * p_highb.c[1])
        D_fit_contraints = D_representation.make_params()
        D_fit = D_representation.fit(np.log(S[bval_thr <= bvals]), x=bvals[bval_thr <= bvals],
                                     params=D_fit_contraints, method='cg')
        ivim_params["Dinit"] = D_fit.params["D"].value
        ivim_params["S0(1-f)"] = np.exp(D_fit.params["lnS0"].value)

        # 2) fit S0, Fivim, D* and D with biexponential model using the previously determined D as initial value
        ivim_model = Model(ivim_1pool_model)
        ivim_model.set_param_hint('S0', value=1.2 * ivim_params["S0(1-f)"], min=0.7 * ivim_params["S0(1-f)"],
                                  max=1.5 * ivim_params["S0(1-f)"])
        ivim_model.set_param_hint('Fivim', value=0.06, min=0, max=0.3)
        ivim_model.set_param_hint('Dstar', value=5e-3, min=3e-3, max=35.1e-3)
        ivim_model.set_param_hint('D', value=ivim_params["Dinit"], min=0.2e-3, max=2.95e-3)
        # ivim_model.set_param_hint('D', value=ivim_params["Dinit"], vary=False)  # fix D to Dinit
        ivim_model_constraints = ivim_model.make_params()
        # run fit algo
        fit_res = ivim_model.fit(S, x=bvals, params=ivim_model_constraints, method='differential_evolution')
        # fit_res = ivim_model.fit(S[bvals >= 25], x=bvals[bvals >= 25], params=fitting_params, method='cg')  # try to fit only b-values >=25

        # save fit params
        ivim_params["S0"] = fit_res.params["S0"].value
        ivim_params["Fivim"] = fit_res.params["Fivim"].value
        ivim_params["D"] = fit_res.params["D"].value
        ivim_params["Dstar"] = fit_res.params["Dstar"].value
        ivim_params["AIC"] = fit_res.aic
        ivim_params["R2"] = get_r2(fit_res)

        plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
        # plot and save fit if ofolder selected
        fit_res.approach = '1shot_initD'
        if len(plot_dir.split('/')) > 2:  # BECAREFUL: THIS WON'T WORK IF THE SELECTED OUTPUT FOLDER IS '/'
            ax = plot_fit(bvals, S, fit_res)
            xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
            ax.plot(xwide, -ivim_params["Dinit"] * xwide + np.log(ivim_params["S0(1-f)"]), color='orange',
                    linestyle='-', linewidth=1, label='D and S0 initialization')
            ax.plot(xwide, -0.2e-3 * ivim_params["Dinit"] * xwide + 1.5 * np.log(ivim_params["S0(1-f)"]),
                    color='orange', linestyle='--', linewidth=1, label='Lower bound for D and S0')
            ax.plot(xwide, -2.9e-3 * xwide + 0.7 * np.log(ivim_params["S0(1-f)"]), color='orange',
                    linestyle='--',
                    linewidth=1, label='Upper bound for D and S0')
            ax.legend(loc=1, prop={'size': 15})
            if true_params != {}:
                true_params_value_to_display = "TRUE VALUES\nS$_0$=%.1f\nf$_{IVIM}$=%.3f\nD$^*$=%.3e mm$^2$/s\nD=%.3e mm$^2$/s" % (
                    true_params['S0'], true_params['Fivim'], true_params['Dstar'], true_params['D'])
                bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
                ax.text(.92, .01,
                        true_params_value_to_display,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        size=15,
                        bbox=bbox_props)

            # save figure and display progress by counting number of plots in directory
            plt.savefig(oplot_fname)
            n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
            plt.close()

        else:
            plot_filename = oplot_fname.split('/')[-1].split('.')[0]
            z_idx, y_idx, x_idx = int(plot_filename.split('z')[1].split('_')[0]), int(
                plot_filename.split('y')[1].split('_')[0]), int(plot_filename.split('x')[1])
            dim_cube = pow(n_vox_to_fit, 1. / 3)
            n_voxel_done = z_idx * dim_cube * dim_cube + y_idx * dim_cube + x_idx

        print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')


    except ValueError, err_detail:
        print('/!\\/!\\/!\\ VALUE ERROR /!\\/!\\/!\\: ' + str(err_detail))
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0(1-f)"] = 0
        ivim_params["Dinit"] = 0
        ivim_params["S0"] = 0
        ivim_params["Fivim"] = 0
        ivim_params["D"] = 0
        ivim_params["Dstar"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    return ivim_params


def fit_1shot_initD_v2(S, bvals, oplot_fname, n_vox_to_fit=1, true_params={}, verbose=1, bval_thr=500):
    """
    v3: adapted bounds for D and D* to adapt to real data and D NOT FIXED to the initial value resulting from the fit
    of b-values >= bval_thr only.
    """

    ivim_params = {"exception": 0}

    try:
        # 1) get first estimate of D
        if sum(bval_thr <= bvals) > 1:
            # by fiting high b-values
            p_highb = np.poly1d(np.polyfit(bvals[bval_thr <= bvals], np.log(S[bval_thr <= bvals]), 1))
            D_representation = Model(D_log_representation)
            D_representation.set_param_hint('D', value=-p_highb.c[0], min=0.15e-3, max=4.5e-3)
            D_representation.set_param_hint('lnS0', value=p_highb.c[1], min=0.5 * p_highb.c[1],
                                            max=1.5 * p_highb.c[1])
            D_fit_contraints = D_representation.make_params()
            D_fit = D_representation.fit(np.log(S[bval_thr <= bvals]), x=bvals[bval_thr <= bvals],
                                         params=D_fit_contraints, method='cg')
            ivim_params["Dinit"] = D_fit.params["D"].value
            ivim_params["S0(1-f)"] = np.exp(D_fit.params["lnS0"].value)
        else:
            # if no high b-value available
            ivim_params["Dinit"] = np.mean([0.15e-3, 4.5e-3])  # mean of min and max bounds
            ivim_params["S0(1-f)"] = max(S)  # max signal value

        # 2) fit S0, Fivim, D* and D with biexponential model using the previously determined D as initial value
        ivim_model = Model(ivim_1pool_model)
        ivim_model.set_param_hint('S0', value=1.2 * ivim_params["S0(1-f)"], min=0.7 * ivim_params["S0(1-f)"],
                                  max=1.7 * ivim_params["S0(1-f)"])
        ivim_model.set_param_hint('Fivim', value=0.1, min=0, max=0.35)
        ivim_model.set_param_hint('Dstar', value=(ivim_params["Dinit"] * 0.8 + 35.5e-3) / 2.,
                                  min=ivim_params["Dinit"] * 0.8, max=35.5e-3)
        ivim_model.set_param_hint('D', value=ivim_params["Dinit"], min=0.15e-3, max=ivim_params["Dinit"] * 1.2)
        # ivim_model.set_param_hint('D', value=ivim_params["Dinit"], vary=False)  # fix D to Dinit
        ivim_model_constraints = ivim_model.make_params()
        # run fit algo
        fit_res_DE = ivim_model.fit(S, x=bvals, params=ivim_model_constraints, method='differential_evolution')

        # 3) refine fit
        ivim_model = Model(ivim_1pool_model)
        ivim_model.set_param_hint('S0', value=fit_res_DE.params["S0"].value,
                                  min=0.95 * fit_res_DE.params["S0"].value,
                                  max=1.05 * fit_res_DE.params["S0"].value)
        if fit_res_DE.params["Fivim"].value <= 1e-12:
            ivim_model.set_param_hint('Fivim', value=fit_res_DE.params["Fivim"].value, min=0, max=0.05)
        else:
            ivim_model.set_param_hint('Fivim', value=fit_res_DE.params["Fivim"].value,
                                      min=0.95 * fit_res_DE.params["Fivim"].value,
                                      max=1.05 * fit_res_DE.params["Fivim"].value)
        ivim_model.set_param_hint('Dstar', value=fit_res_DE.params["Dstar"].value,
                                  min=0.95 * fit_res_DE.params["Dstar"].value,
                                  max=1.05 * fit_res_DE.params["Dstar"].value)
        ivim_model.set_param_hint('D', value=fit_res_DE.params["D"].value,
                                  min=0.95 * fit_res_DE.params["D"].value,
                                  max=1.05 * fit_res_DE.params["D"].value)
        ivim_model_constraints = ivim_model.make_params()
        # run fit algo
        fit_res = ivim_model.fit(S, x=bvals, params=ivim_model_constraints, method='differential_evolution')

        # save fit params
        ivim_params["S0"] = fit_res.params["S0"].value
        ivim_params["Fivim"] = fit_res.params["Fivim"].value
        ivim_params["D"] = fit_res.params["D"].value
        ivim_params["Dstar"] = fit_res.params["Dstar"].value
        ivim_params["AIC"] = fit_res.aic
        ivim_params["R2"] = get_r2(fit_res)

        fit_res.approach = '1shot_initD_v3'
        plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
        if plot_dir[:2] != '//':  # BECAREFUL: THIS WON'T WORK IF THE SELECTED OUTPUT FOLDER IS '/'
            # plot and save fit
            ax = plot_fit(bvals, S, fit_res)
            xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
            ax.plot(xwide, -ivim_params["Dinit"] * xwide + np.log(ivim_params["S0(1-f)"]), color='orange',
                    linestyle='-', linewidth=1, label='D and S0 initialization')
            ax.plot(xwide, -0.2e-3 * ivim_params["Dinit"] * xwide + 1.7 * np.log(ivim_params["S0(1-f)"]),
                    color='orange', linestyle='--', linewidth=1, label='Lower bound for D and S0')
            ax.plot(xwide, -ivim_params["Dinit"] * 1.2 * xwide + 0.7 * np.log(ivim_params["S0(1-f)"]),
                    color='orange', linestyle='--', linewidth=1, label='Upper bound for D and S0')
            ax.legend(loc=1, prop={'size': 15})
            if true_params != {}:
                true_params_value_to_display = "TRUE VALUES\nS$_0$=%.1f\nf$_{IVIM}$=%.3f\nD$^*$=%.3e mm$^2$/s\nD=%.3e mm$^2$/s" % (
                    true_params['S0'], true_params['Fivim'], true_params['Dstar'], true_params['D'])
                bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
                ax.text(.92, .01,
                        true_params_value_to_display,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        size=15,
                        bbox=bbox_props)

            # save figure and display progress by counting number of plots in directory
            plt.savefig(oplot_fname)
            plt.close()
            n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])

        else:
            plot_filename = oplot_fname.split('/')[-1].split('.')[0]
            z_idx, y_idx, x_idx = int(plot_filename.split('z')[1].split('_')[0]), int(
                plot_filename.split('y')[1].split('_')[0]), int(plot_filename.split('x')[1])
            dim_cube = pow(n_vox_to_fit, 1. / 3)
            n_voxel_done = z_idx * dim_cube * dim_cube + y_idx * dim_cube + x_idx

        if verbose:
            print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')


    except ValueError, err_detail:
        print('/!\\/!\\/!\\ VALUE ERROR /!\\/!\\/!\\: ' + str(err_detail))
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0(1-f)"] = 0
        ivim_params["Dinit"] = 0
        ivim_params["S0"] = 0
        ivim_params["Fivim"] = 0
        ivim_params["D"] = 0
        ivim_params["Dstar"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    except IndexError, err_detail:
        print('/!\\/!\\/!\\ INDEX ERROR /!\\/!\\/!\\: ' + str(
            err_detail) + '\nMIGHT BE DUE TO numpy.polyfit UNDER UBUNTU 16 WHICH RETURNS ONLY ONE COEFFICIENT INSTEAD OF TWO UNDER MACOSX...')
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0(1-f)"] = 0
        ivim_params["Dinit"] = 0
        ivim_params["S0"] = 0
        ivim_params["Fivim"] = 0
        ivim_params["D"] = 0
        ivim_params["Dstar"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    return ivim_params

def fit_1shot_initD_noise(S, bvals, oplot_fname, n_vox_to_fit=1, bval_thr=500):
    """
    Fit IVIM 1-pool model: S0*( Fivim*exp(-x*Dstar) + 1-Fivim )*exp(-x*D)
    :param vox_coord: coordinates of the voxel to be fit
    :param dwi_data: diffusion-weighted 4D data (x, y, z, b-values)
    :param bvals: b-values acquired
    :param mask: binary mask defining the voxels to fit
    :param ofolder_plot: path of the output folder for plots
    :return:
    """

    ivim_params = {"exception": 0}

    try:
        # 1) fit high b-values to get a first estimate of D
        p_highb = np.poly1d(np.polyfit(bvals[bvals >= bval_thr], np.log(S[bvals >= bval_thr]), 1))
        ivim_params["D"] = -p_highb.c[0]

        # 2) fit S0, Fivim, D* and D with biexponential model using the previously determined D as initial value
        ivim_model = Model(ivim_1pool_model_with_noise)
        # set params initial values and bounds
        ivim_model.set_param_hint('S0', value=S[0], min=S[0], max=10 * S[0])
        ivim_model.set_param_hint('Fivim', value=.02, min=0, max=.99)
        ivim_model.set_param_hint('Dstar', value=1e-3, min=0, max=1e-1)
        ivim_model.set_param_hint('D', value=ivim_params["D"], min=0.5 * ivim_params["D"],
                                  max=1.5 * ivim_params["D"])
        ivim_model.set_param_hint('noise', value=0, min=-np.mean(S), max=np.mean(S))
        fitting_params = ivim_model.make_params()
        # run fit algo
        fit_res = ivim_model.fit(S, x=bvals, params=fitting_params, method='cg')

        # save fit params
        ivim_params["S0"] = fit_res.params["S0"].value
        ivim_params["Fivim"] = fit_res.params["Fivim"].value
        ivim_params["D"] = fit_res.params["D"].value
        ivim_params["Dstar"] = fit_res.params["Dstar"].value
        ivim_params["noise"] = fit_res.params["noise"].value
        ivim_params["AIC"] = fit_res.aic
        ivim_params["R2"] = get_r2(fit_res)

        # plot and save fit
        fit_res.approach = '1shot_initD_noise'
        ax = plot_fit(bvals, S, fit_res)
        xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
        ax.plot(xwide, p_highb.c[0] * xwide + p_highb.c[1], color='orange', linestyle='-', linewidth=1,
                label='fit b$\leq$' + str(bval_thr))
        ax.plot(xwide, 0.5 * p_highb.c[0] * xwide + p_highb.c[1], color='orange', linestyle='--', linewidth=1,
                label='Lower bound for D')
        ax.plot(xwide, 1.5 * p_highb.c[0] * xwide + p_highb.c[1], color='orange', linestyle='--', linewidth=1,
                label='Upper bound for D')
        ax.legend(loc=1, prop={'size': 15})
        plt.savefig(oplot_fname)
        plt.close()

        # display progress by counting number of plots in directory
        plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
        n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
        print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')

    except ValueError, err_detail:
        print('/!\\/!\\/!\\ VALUE ERROR /!\\/!\\/!\\: ' + str(err_detail))
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0"] = 0
        ivim_params["Fivim"] = 0
        ivim_params["D"] = 0
        ivim_params["Dstar"] = 0
        ivim_params["noise"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    return ivim_params

def fit_combine_2shots_1shot_lemke(S, bvals, oplot_fname, n_vox_to_fit=1, bval_thr=500):
    """
    Fit IVIM 1-pool model: S0*( Fivim*exp(-x*Dstar) + 1-Fivim )*exp(-x*D)
    :param vox_coord: coordinates of the voxel to be fit
    :param dwi_data: diffusion-weighted 4D data (x, y, z, b-values)
    :param bvals: b-values acquired
    :param mask: binary mask defining the voxels to fit
    :param ofolder_plot: path of the output folder for plots
    :return:
    """

    ivim_params = {"exception": 0}

    try:
        # 1) fit high b-values to get a first estimate of D
        p_highb = np.poly1d(np.polyfit(bvals[bval_thr <= bvals], np.log(S[bval_thr <= bvals]), 1))
        # p_highb = np.poly1d(np.polyfit(bvals[(bvals <= 5) | (bval_thr <= bvals)], np.log(S[(bvals <= 5) | (bval_thr <= bvals)]), 1))
        # p_highb = np.poly1d(np.polyfit(bvals[(bvals == bvals[0]) | (bval_thr <= bvals)], np.log(S[(bvals == bvals[0]) | (bval_thr <= bvals)]), 1))
        # ivim_params["Dinit"] = -p_highb.c[0]
        # ivim_params["S0init"] = np.exp(p_highb.c[1])
        D_representation = Model(D_log_representation)
        D_representation.set_param_hint('D', value=0.6e-3, min=0.2e-3, max=2.9e-3)
        D_representation.set_param_hint('lnS0', value=p_highb.c[1], min=0.5 * p_highb.c[1],
                                        max=2 * p_highb.c[1])
        D_fit_contraints = D_representation.make_params()
        D_fit = D_representation.fit(np.log(S[bval_thr <= bvals]), x=bvals[bval_thr <= bvals],
                                     params=D_fit_contraints, method='brute')
        ivim_params["Dinit"] = D_fit.params["D"].value
        ivim_params["S0(1-f)"] = np.exp(D_fit.params["lnS0"].value)

        # 2) fit S0, Fivim, D* and D with biexponential model using the previously determined D as initial value
        ivim_model = Model(ivim_1pool_model)
        ivim_model.set_param_hint('S0', value=1.2 * ivim_params["S0(1-f)"], min=0.5 * ivim_params["S0(1-f)"],
                                  max=2. * ivim_params["S0(1-f)"])
        ivim_model.set_param_hint('Fivim', value=0.06, min=0, max=0.3)
        ivim_model.set_param_hint('Dstar', value=5e-3, min=3e-3, max=35e-3)
        ivim_model.set_param_hint('D', value=ivim_params["Dinit"], min=0.2e-3, max=2.9e-3)
        # ivim_model.set_param_hint('D', value=ivim_params["Dinit"], vary=False)  # fix D to Dinit
        ivim_model_constraints = ivim_model.make_params()
        # run fit algo
        fit_res = ivim_model.fit(S, x=bvals, params=ivim_model_constraints, method='brute')
        # fit_res = ivim_model.fit(S[bvals >= 25], x=bvals[bvals >= 25], params=fitting_params, method='cg')  # try to fit only b-values >=25

        # save fit params
        ivim_params["S0"] = fit_res.params["S0"].value
        ivim_params["Fivim"] = fit_res.params["Fivim"].value
        ivim_params["Dstar"] = fit_res.params["Dstar"].value
        ivim_params["D"] = fit_res.params["D"].value

        # 3) fit Lemke 2010 model (T1, T2 corrected)
        ivim_lemke_model = Model(ivim_lemke2010_model)
        # set params initial values and bounds
        ivim_lemke_model.set_param_hint('S0', value=fit_res.params["S0"].value, min=0.7 * fit_res.params["S0"],
                                        max=1.7 * fit_res.params["S0"])
        ivim_lemke_model.set_param_hint('Fivim', value=fit_res.params["Fivim"].value, min=0, max=.3)
        ivim_lemke_model.set_param_hint('Dstar', value=fit_res.params["Dstar"].value, min=3e-3, max=35e-3)
        ivim_lemke_model.set_param_hint('D', value=fit_res.params["D"].value, min=0.2e-3, max=2.9e-3)
        # ivim_lemke_model.set_param_hint('D', value=ivim_params["Dinit"], vary=False)  # fix D to Dinit
        # set fixed values
        # ivim_model.set_param_hint('TR', value=3300, vary=False)  # fix TR
        ivim_lemke_model.set_param_hint('TE', value=51.6, vary=False)  # fix TE
        # ivim_model.set_param_hint('T1tiss', value=np.mean([1313, 1182]), vary=False)  # T1 in spinal cord from Massire et al., NeuroImage 2016 (mean T1 across GM and WM at 7T)
        ivim_lemke_model.set_param_hint('T2tiss', value=50.,
                                        vary=False)  # T2 in spinal cord from Massire et al., Proceedings ISMRM 2016, abstract #1130
        # ivim_model.set_param_hint('T1bl', value=2100, vary=False)  # T1 in blood at 7T (ms) from Zhang et al., MRM 2013
        ivim_lemke_model.set_param_hint('T2bl', value=235.,
                                        vary=False)  # T2 in blood (value linearly interpolated from 1.5 et 3T values)
        ivim_lemke_model_constraints = ivim_lemke_model.make_params()
        # run fit algo
        fit_lemke_res = ivim_lemke_model.fit(S, x=bvals, params=ivim_lemke_model_constraints, method='brute',
                                             nan_policy='raise')
        # fit_lemke_res = ivim_lemke_model.fit(S[bvals >= 25], x=bvals[bvals >= 25], params=fitting_params, method='cg', nan_policy='raise')  # try to fit only b-values >=25

        # save fit params
        ivim_params["S0"] = fit_lemke_res.params["S0"].value
        ivim_params["Fivim"] = fit_lemke_res.params["Fivim"].value
        ivim_params["D"] = fit_lemke_res.params["D"].value
        ivim_params["Dstar"] = fit_lemke_res.params["Dstar"].value
        ivim_params["AIC"] = fit_lemke_res.aic
        ivim_params["R2"] = get_r2(fit_lemke_res)

        # plot and save fit
        fit_lemke_res.approach = 'combine'
        ax = plot_fit(bvals, S, fit_lemke_res)
        xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
        ax.plot(xwide, -ivim_params["Dinit"] * xwide + np.log(ivim_params["S0(1-f)"]), color='orange',
                linestyle='-', linewidth=1, label='D and S0 initialization')
        ax.plot(xwide, -0.7 * ivim_params["Dinit"] * xwide + 1.2 * np.log(ivim_params["S0(1-f)"]),
                color='orange',
                linestyle='--', linewidth=1, label='Lower bound for D and S0')
        ax.plot(xwide, -1.3 * ivim_params["Dinit"] * xwide + 0.8 * np.log(ivim_params["S0(1-f)"]),
                color='orange',
                linestyle='--', linewidth=1, label='Upper bound for D and S0')
        ax.plot(xwide, np.log(fit_res.eval(x=xwide)), color='r', linestyle=':', linewidth=1,
                label='biexponential fit initializing TE/T2 correction')
        ax.legend(loc=1, prop={'size': 15})
        ivim_params["fig"] = plt.gcf()

        # save figure and display progress by counting number of plots in directory
        plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
        if len(plot_dir.split('/')) > 2:  # BECAREFUL: THIS WON'T WORK IF THE SELECTED OUTPUT FOLDER IS '/'
            plt.savefig(oplot_fname)
            n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
        else:
            plot_filename = oplot_fname.split('/')[-1].split('.')[0]
            z_idx, y_idx, x_idx = int(plot_filename.split('z')[1][0]), int(plot_filename.split('y')[1][0]), int(
                plot_filename.split('x')[1][0])
            dim_cube = n_vox_to_fit ** (1 / 3)
            n_voxel_done = z_idx * dim_cube * dim_cube + y_idx * dim_cube + x_idx

        plt.close()
        print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')

    except ValueError, err_detail:
        print('/!\\/!\\/!\\ VALUE ERROR /!\\/!\\/!\\: ' + str(err_detail))
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0(1-f)"] = 0
        ivim_params["Dinit"] = 0
        ivim_params["S0"] = 0
        ivim_params["Fivim"] = 0
        ivim_params["D"] = 0
        ivim_params["Dstar"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    return ivim_params

def fit_FivimXDstar(S, bvals, oplot_fname, n_vox_to_fit=1, bval_thr=500):
    """
    Fit FivimXDstar directly
    :param vox_coord: coordinates of the voxel to be fit
    :param dwi_data: diffusion-weighted 4D data (x, y, z, b-values)
    :param bvals: b-values acquired
    :param mask: binary mask defining the voxels to fit
    :param ofolder_plot: path of the output folder for plots
    :return:
    """

    ivim_params = {"exception": 0}

    try:
        # 1) fit high b-values to estimate pure D
        p_highb = np.poly1d(
            np.polyfit(bvals[(bvals <= 5) | (bval_thr <= bvals)], np.log(S[(bvals <= 5) | (bval_thr <= bvals)]),
                       1))
        ivim_params["D"] = -p_highb.c[0]

        # 2) fit Taylor expansion (until Kurtosis term) on low b-values
        kurtosis_model = Model(kurtosis_representation)
        # set params initial values and bounds
        kurtosis_model.set_param_hint('S0', value=np.exp(p_highb.c[1]), min=0.9 * np.exp(p_highb.c[1]),
                                      max=1.1 * np.exp(p_highb.c[1]))
        # kurtosis_model.set_param_hint('S0', value=np.exp(p_highb.c[1]), vary=False)
        kurtosis_model.set_param_hint('ADC', value=1.0e-3, min=0.0,
                                      max=1.0e-4)  # Apparent diffusion coefficient and fractional anisotropy in spinal cord: Age and cervical spondylosis–related changes Hatsuho Mamata MD, PhD Ferenc A. Jolesz MD Stephan E. Maier MD, PhD
        kurtosis_model.set_param_hint('K', value=0.6, min=0, max=10)
        fitting_params = kurtosis_model.make_params()
        # run fit algo
        fit_kurt_res = kurtosis_model.fit(S[bvals <= bval_thr], x=bvals[bvals <= bval_thr],
                                          params=fitting_params,
                                          method='cg')

        # save fit params
        ivim_params["S0"] = fit_kurt_res.params["S0"].value
        ivim_params["ADC"] = fit_kurt_res.params["ADC"].value
        ivim_params["K"] = fit_kurt_res.params["K"].value
        ivim_params["AIC"] = fit_kurt_res.aic
        ivim_params["R2"] = get_r2(fit_kurt_res)

        # 3) compute Fivim*Dstar
        ivim_params["FivimXDstar"] = ivim_params["ADC"] - ivim_params["D"]

        # plot and save fit
        fit_kurt_res.approach = 'FivimXDstar'
        ax = plot_fit(bvals, S, fit_kurt_res)
        xwide = np.linspace(0, np.max(bvals), np.max(bvals) * 2)
        ax.plot(xwide, p_highb.c[0] * xwide + p_highb.c[1], color='orange', linestyle='-', linewidth=1,
                label='fit b = ' + ', '.join(map(str, bvals[(bvals <= 5) | (bval_thr <= bvals)])))
        ax.plot(xwide, p_highb.c[0] * xwide + 0.9 * p_highb.c[1], color='orange', linestyle='--', linewidth=1,
                label='Lower bound for S0')
        ax.plot(xwide, p_highb.c[0] * xwide + 1.1 * p_highb.c[1], color='orange', linestyle='--', linewidth=1,
                label='Upper bound for S0')
        # ax.plot(xwide, np.log(fit_kurt_res.eval(x=xwide)), color='r', linestyle=':', linewidth=1, label='Kurtosis representation fit on b$\leq$'+str(bval_thr))
        ax.legend(loc=1, prop={'size': 15})
        plt.savefig(oplot_fname)
        plt.close()

        # display progress by counting number of plots in directory
        plot_dir = os.path.dirname(os.path.realpath(oplot_fname))
        n_voxel_done = len([plot for plot in os.walk(plot_dir).next()[2] if plot[-4:] == ".png"])
        print(str(100. * n_voxel_done / n_vox_to_fit) + '% of voxels done.')

    except ValueError, err_detail:
        print('/!\\/!\\/!\\ VALUE ERROR /!\\/!\\/!\\: ' + str(err_detail))
        print('--> ignoring voxel (' + oplot_fname.split('/')[-1].split(',')[0] + ')')
        ivim_params["S0"] = 0
        ivim_params["FivimXDstar"] = 0
        ivim_params["D"] = 0
        ivim_params["ADC"] = 0
        ivim_params["K"] = 0
        ivim_params["AIC"] = 0
        ivim_params["R2"] = 0
        ivim_params["exception"] = 1

    return ivim_params


def fit_warpper(args_fit_func):
    """
    Helps to unpack arguments for parallel processing.
    :param args_fit_func:
    :return:
    """
    if approach == '1pool_separate':
        return fit_1pool_separate(*args_fit_func)
    elif approach == '1pool_1shot':
        return fit_1pool_1shot(*args_fit_func)
    elif approach == 'lemke2010':
        return fit_lemke2010(*args_fit_func)
    elif approach == 'combine':
        return fit_combine_2shots_1shot_lemke(*args_fit_func)
    elif approach == '1shot_initD':
        return fit_1shot_initD(*args_fit_func)
    elif approach == 'two-step':
        return fit_2shots(*args_fit_func)
    elif approach == 'one-step':
        return fit_1shot_initD_v2(*args_fit_func)
    elif approach == '1shot_initD_noise':
        return fit_1shot_initD_noise(*args_fit_func)
    elif approach == 'FivimXDstar':
        return fit_FivimXDstar(*args_fit_func)
    else:
        print('ERROR: Unknown model.')

# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program fits IVIM model voxel-wise and produces IVIM parameters maps.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', '--dwi', dest='dwi_fname', help='Path to 4D nifti diffusion-weighted images.', type=str, required=True)
    requiredArgs.add_argument('-b', '--bval', dest='bval_fname', help="Path to corresponding bval file.", type=str, required=True)
    requiredArgs.add_argument('-ma', '--mask', dest='mask_fname', help='Path to mask nifti file defining voxels to be considered.', type=str, required=True)

    optionalArgs.add_argument('-mo', '--model', dest='model', help='Fit approach: one-step or two-step.', type=str, required=False, default='one-step')
    optionalArgs.add_argument('-o', '--ofolder', dest='ofolder', help='Output folder name.', type=str, required=False, default='params_map')
    optionalArgs.add_argument('-mt', '--multithreading', dest='multithreading', help='Parallelize fit on multiple threads.', type=str, required=False, default='1')
    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # print citation
    print '\n\n'
    print '\n****************************** <3 Thank you for using our toolbox! <3 *********************************' \
          '\n********************************* PLEASE CITE THE FOLLOWING PAPER *************************************' \
          '\nLévy S., Rapacchi S., Massire A., Troalen T., Feiweier T., Guye M., Callot V., Intra-Voxel Incoherent ' \
          '\nMotion at 7 Tesla to quantify human spinal cord microperfusion: limitations and promises, Magnetic ' \
          '\nResonance in Medicine, 1902:334-357, 2019.' \
          '\n*******************************************************************************************************'
    print '\n\n'

    # run main
    main(dwi_fname=args.dwi_fname, bval_fname=args.bval_fname, mask_fname=args.mask_fname, model=args.model, ofolder=args.ofolder, multithreading=bool(int(args.multithreading)))

