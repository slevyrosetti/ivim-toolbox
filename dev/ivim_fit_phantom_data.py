#!/usr/bin/env python3.6

import numpy as np
import ivim_fitting
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import os

def main(maskFnames, dwiFnames, bvalFnames, oPlotNames, average):
    """Main."""

    # initialize variables
    nCases = len(oPlotNames)
    data = np.empty((nCases), dtype=dict)

    # in case the same DWI data, mask or b-values can be used for everyone
    if len(dwiFnames) == 1:
        dwi = nib.load(dwiFnames[0]).get_data()
    if len(maskFnames) == 1:
        mask = nib.load(maskFnames[0]).get_data()
    if len(bvalFnames) == 1:
        bvals = np.loadtxt(bvalFnames[0], delimiter=None)

    # loop over cases
    for i_case in range(nCases):

        # load data
        if len(dwiFnames) != 1 and i_case > 0:
            dwi = nib.load(dwiFnames[i_case]).get_data()
        if len(maskFnames) != 1 and i_case > 0:
            mask = nib.load(maskFnames[i_case]).get_data()
        if len(bvalFnames) != 1 and i_case > 0:
            bvals = np.loadtxt(bvalFnames[i_case], delimiter=None)
        data[i_case] = {"b-values": bvals, "plot file name": os.path.abspath(oPlotNames[i_case])}

        # extract signal for each volume
        data[i_case]["Sroi"] = np.zeros((dwi.shape[3]))
        for i_t in range(dwi.shape[3]):
            dwi_t_i = dwi[:, :, :, i_t]
            data[i_case]["Sroi"][i_t] = np.mean(dwi_t_i[mask > 0])

        # average across repetitions if asked by user, and normalize by b=0
        if average:
            bvals_unique = np.unique(bvals)
            data[i_case]["Sroi_averaged"] = np.zeros(len(bvals_unique))
            for i_b in range(len(bvals_unique)):
                data[i_case]["Sroi_averaged"][i_b] = np.mean(data[i_case]["Sroi"][bvals == bvals_unique[i_b]])
            data[i_case]["Sroi_norm"] = np.divide(data[i_case]["Sroi_averaged"], np.broadcast_to(data[i_case]["Sroi_averaged"][0], data[i_case]["Sroi_averaged"].shape))

        # fit bi-exponential model with one-step method
        ivim_fit = ivim_fitting.IVIMfit(bvals=bvals_unique,
                                        voxels_values=np.array([data[i_case]["Sroi_norm"]]),
                                        voxels_idx=tuple([np.array([0]), np.array([0]), np.array([0])]),
                                        ofit_dir=data[i_case]["plot file name"],
                                        model='one-step',
                                        multithreading=0)
        ivim_fit.run_fit()

        # linear fit to get D
        fit_x = bvals_unique
        fit_y = data[i_case]["Sroi_norm"]
        p_highb, r2, sum_squared_error = fit_D_only(fit_x[(fit_x >= 500) & (fit_x <= 1000)], np.log(fit_y[(fit_x >= 500) & (fit_x <= 1000)]))
        data[i_case]["polyfit high D"] = p_highb

    # plot the data along with the fit on high b-values
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
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap(np.repeat(np.linspace(0, 1, nCases), 2)))
    # colors = plt.cm.jet(np.linspace(0, 1, len(data_folders)))
    # colors = ['r', 'b', 'g', 'k', 'y', 'c', 'orange']
    xp = np.linspace(bvals[0], bvals[-1], 1000)

    for i_case in range(nCases):

        # color = '%02x%02x%02x' % tuple(colors[i_folder][0:-1])
        # plot = ax.plot(bvals_unique, np.log(data[i_case]["Sroi_norm"]), '.-', label=oPlotNames[i_case], markersize=8.)
        plot = ax.plot(bvals_unique, data[i_case]["Sroi_averaged"], '.-', label=oPlotNames[i_case], markersize=8.)
        # ax.plot(xp, data[i_case]["polyfit high D"](xp), '-', linewidth=0.5, color=plot[0].get_color())

    ax.grid()
    # ax.set(xlabel='b-values (s/mm$^2$)', ylabel='ln(S)', title='DW signal for different pump speeds')
    ax.set(xlabel='b-values (s/mm$^2$)', ylabel='S', title='Acquisition on phantom (2020-09-03)')
    ax.legend()
    fig.savefig("ivim_fit_phantom_data_noLogNoNorm.png")

    print('**** Done ****')


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


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program extracts signal values within provided ROIs, fit the IVIM models and generates plots.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-mask', dest='maskFnames', help='List (separate items by commas) of paths to mask Nifti data.', type=str, required=True)
    requiredArgs.add_argument('-dwi', dest='dwiFnames', help="List (separate items by commas) of paths to DWI Nifti data.", type=str, required=True)
    requiredArgs.add_argument('-bval', dest='bvalFnames', help="List (separate items by commas) of paths to b_values files.", type=str, required=True)
    requiredArgs.add_argument('-o', dest='oPlotNames', help="List (separate items by commas) of corresponding names for the output plots.", type=str, required=True)

    optionalArgs.add_argument('-average', dest='bool', help='Optional argument to average or not volumes across repetitions before fitting and plotting (ONLY THIS OPTION IMPLEMENTED SO FAR).', type=str, required=False, default=True)

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
    main(maskFnames=args.maskFnames.split(','), dwiFnames=args.dwiFnames.split(','), bvalFnames=args.bvalFnames.split(','), oPlotNames=args.oPlotNames.split(','), average=args.bool)




