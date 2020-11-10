#!/usr/bin/env python3.6

import numpy as np
import ivim_fitting
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import os

def main(maskFnames, dwiFnames, bvalFnames, oPlotNames, average, title):
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
        if len(dwiFnames) != 1:
            dwi = nib.load(dwiFnames[i_case]).get_data()
        if len(maskFnames) != 1:
            mask = nib.load(maskFnames[i_case]).get_data()
        if len(bvalFnames) != 1:
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
            data[i_case]["Sroi_averaged"] = np.zeros((len(bvals_unique), 2))  # Nbvals X (mean, std across reps)
            for i_b in range(len(bvals_unique)):
                data[i_case]["Sroi_averaged"][i_b, 0] = np.mean(data[i_case]["Sroi"][bvals == bvals_unique[i_b]])
                data[i_case]["Sroi_averaged"][i_b, 1] = np.std(data[i_case]["Sroi"][bvals == bvals_unique[i_b]])
            data[i_case]["Sroi_norm"] = np.divide(data[i_case]["Sroi_averaged"], np.broadcast_to(data[i_case]["Sroi_averaged"][0, 0], data[i_case]["Sroi_averaged"].shape))
            # compute the SD after normalization and log
            Sroi_nonAv_norm = np.divide(data[i_case]["Sroi"], np.broadcast_to(data[i_case]["Sroi_averaged"][0, 0], data[i_case]["Sroi"].shape))
            data[i_case]["Sroi_norm_log"] = np.zeros((len(bvals_unique), 2))
            for i_b in range(len(bvals_unique)):
                data[i_case]["Sroi_norm_log"][i_b, 0] = np.mean(np.log(Sroi_nonAv_norm[bvals == bvals_unique[i_b]]))
                data[i_case]["Sroi_norm_log"][i_b, 1] = np.std(np.log(Sroi_nonAv_norm[bvals == bvals_unique[i_b]]))

        # fit bi-exponential model with one-step method
        ivim_fit = ivim_fitting.IVIMfit(bvals=bvals_unique,
                                        voxels_values=np.array([data[i_case]["Sroi_norm"][:, 0]]),
                                        voxels_idx=tuple([np.array([0]), np.array([0]), np.array([0])]),
                                        ofit_dir=data[i_case]["plot file name"],
                                        model='one-step',
                                        multithreading=0)
        ivim_fit.run_fit()

        # linear fit to get D
        fit_x = bvals_unique
        fit_y = data[i_case]["Sroi_norm_log"][:, 0]
        p_highb, r2, sum_squared_error = fit_D_only(fit_x[(fit_x >= 500) & (fit_x <= 1000)], fit_y[(fit_x >= 500) & (fit_x <= 1000)])
        data[i_case]["polyfit high D"] = p_highb

    # print results
    print("SD across repetitions for each b-value and case\n"
          "===============================================\n"
          +str(bvals_unique)+"\n")
    for i_case in range(nCases):
        print(oPlotNames[i_case]+"\n------------\n"+str(data[i_case]["Sroi_norm_log"][:, 1]))

    # plot results
    # ------------
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), num="Fit D")
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.9, hspace=0.3, bottom=0.1, top=0.85)
    cmap = plt.cm.jet
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap(np.repeat(np.linspace(0, 1, nCases), 2)))
    xp = np.linspace(bvals[0], bvals[-1], 1000)

    for i_case in range(nCases):

        # mean signal
        plot = axes[0, 0].errorbar(bvals_unique, data[i_case]["Sroi_averaged"][:, 0],
                                yerr=data[i_case]["Sroi_averaged"][:, 1],
                                elinewidth=0.3,
                                marker='.',
                                markersize=8.,
                                lw=0,
                                label=oPlotNames[i_case])
        axes[0, 0].plot(xp, data[i_case]["Sroi_averaged"][0, 0]*np.exp(data[i_case]["polyfit high D"](xp)), '-', linewidth=0.5, color=plot[0].get_color())  # fit
        plotLog = axes[0, 1].errorbar(bvals_unique, data[i_case]["Sroi_norm_log"][:, 0],
                                   yerr=data[i_case]["Sroi_norm_log"][:, 1],
                                   elinewidth=0.3,
                                   marker='.',
                                   markersize=8.,
                                   lw=0,
                                   label=oPlotNames[i_case])
        axes[0, 1].plot(xp, data[i_case]["polyfit high D"](xp), '-', linewidth=0.5, color=plotLog[0].get_color())  # fit

        # SD across repetitions
        axes[1, 0].plot(bvals_unique, data[i_case]["Sroi_averaged"][:, 1],
                                marker='.',
                                markersize=8.,
                                lw=0.5,
                                label=oPlotNames[i_case])
        axes[1, 1].plot(bvals_unique, data[i_case]["Sroi_norm_log"][:, 1],
                                marker='.',
                                markersize=8.,
                                lw=0.5,
                                label=oPlotNames[i_case])

    axes[0, 0].grid()
    axes[0, 1].grid()
    axes[0, 0].set(xlabel='b-values (s/mm$^2$)', ylabel='S', title='Mean raw signal across repetitions')
    axes[0, 1].set(xlabel='b-values (s/mm$^2$)', ylabel='ln(S)', title='Log of the signal normalized')
    axes[0, 0].legend()
    axes[1, 0].set(xlabel='b-values (s/mm$^2$)', ylabel='SD across repetitions')
    axes[1, 1].set(xlabel='b-values (s/mm$^2$)', ylabel='SD across repetitions')
    fig.suptitle(title, fontsize=20)
    fig.savefig(title+".png")

    plt.show()

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
    requiredArgs.add_argument('-cases', dest='oPlotNames', help="List (separate items by commas) of corresponding names for the output plots.", type=str, required=True)

    optionalArgs.add_argument('-average', dest='bool', help='Optional argument to average or not volumes across repetitions before fitting and plotting (ONLY THIS OPTION IMPLEMENTED SO FAR).', type=str, required=False, default=True)
    optionalArgs.add_argument('-o', dest='title', help='Title of the main generated plot and file.', type=str, required=False, default='')

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
    main(maskFnames=args.maskFnames.split(','), dwiFnames=args.dwiFnames.split(','), bvalFnames=args.bvalFnames.split(','), oPlotNames=args.oPlotNames.split(','), average=args.bool, title=args.title)




