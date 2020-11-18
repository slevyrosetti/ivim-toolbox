#!/usr/bin/env python3.6

import numpy as np
import ivim_fitting
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import os
import warnings
from cycler import cycler

def main(maskFnames, dwiFnames, bvalFnames, oPlotNames, analysis, title):
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


        # loop over voxels (to have a voxel-wise estimation)
        idx_vox = np.where(mask > 0)
        bvals_unique = np.unique(bvals)
        data[i_case]["S across reps by vox"], data[i_case]["log(S/S0) across reps by vox"], data[i_case]["D across reps by vox"], data[i_case]["S0 across reps by vox"] = np.zeros((len(idx_vox[0]), bvals_unique.shape[0], 2)), np.zeros((len(idx_vox[0]), bvals_unique.shape[0], 2)), np.zeros((len(idx_vox[0]), bvals_unique.shape[0], 2)), np.zeros((len(idx_vox[0]), bvals_unique.shape[0], 2))
        data[i_case]["D based average by vox"], data[i_case]["S0 based average by vox"] = np.zeros((len(idx_vox[0]))), np.zeros((len(idx_vox[0])))
        for i_vox in range(len(idx_vox[0])):
            print("Processing voxel x={}, y={}, z={}\n".format(idx_vox[0][i_vox], idx_vox[1][i_vox], idx_vox[2][i_vox]))

            # extract signal for each volume
            Svox = dwi[idx_vox[0][i_vox], idx_vox[1][i_vox], idx_vox[2][i_vox], :]

            # Calculate mean and SD signal across reps by b-value
            # data[i_case]["Sroi_averaged"] = np.zeros((len(bvals_unique), 2))  # Nbvals X (mean, std across reps)
            for i_b in range(len(bvals_unique)):
                data[i_case]["S across reps by vox"][i_vox, i_b, :] = [np.mean(Svox[bvals == bvals_unique[i_b]]), np.std(Svox[bvals == bvals_unique[i_b]])]

            # Calculate mean and SD log(S/S0) across reps by b-value
            logS_S0_vox = np.log(Svox/data[i_case]["S across reps by vox"][i_vox, bvals_unique == 0, 0])
            for i_b in range(len(bvals_unique)):
                data[i_case]["log(S/S0) across reps by vox"][i_vox, i_b, :] = [np.mean(logS_S0_vox[bvals == bvals_unique[i_b]]), np.std(logS_S0_vox[bvals == bvals_unique[i_b]])]

            # linear fit to get D based on the average signal across reps for this voxel
            p_highb, r2, sum_squared_error = fit_D_only(bvals_unique, data[i_case]["S across reps by vox"][i_vox, :, 0]/data[i_case]["S across reps by vox"][i_vox, bvals_unique == 0, 0])
            data[i_case]["D based average by vox"][i_vox], data[i_case]["S0 based average by vox"][i_vox] = p_highb.coef[0], p_highb.coef[1]

            # # fit bi-exponential model with one-step method
            # ivim_fit = ivim_fitting.IVIMfit(bvals=bvals_unique,
            #                                 voxels_values=np.array([data[i_case]["S across reps by vox"][i_vox, :, 0]/data[i_case]["S across reps by vox"][i_vox, bvals_unique == 0, 0]]),
            #                                 voxels_idx=tuple(idx_vox),
            #                                 ofit_dir='',
            #                                 model='one-step',
            #                                 multithreading=0,
            #                                 save_plots=False)
            # ivim_fit.run_fit()
            # data[i_case]["D based average by vox"][i_vox], data[i_case]["S0 based average by vox"][i_vox] = ivim_fit.ivim_metrics_all_voxels[0]["D"], ivim_fit.ivim_metrics_all_voxels[0]["S0"]
            # del ivim_fit

            # if asked by user, estimate the SD across repetitions on the estimation of D based on the b-value and the 2 previous ones
            if analysis == "rep" and len(bvals) % len(bvals_unique) == 0:

                Nrep = len(bvals)//len(bvals_unique)
                print("\nAnalysis by repetition\nData include {} repetitions\n".format(Nrep))

                for i_b in range(2, len(bvals_unique)):
                    print("Processing b-value {}mm²/s\n".format(bvals_unique[i_b]))

                    D = np.zeros(Nrep)
                    S0 = np.zeros(Nrep)
                    for i_rep in range(Nrep):
                        # linear fit
                        fitPoly, r2, sum_squared_error = fit_D_only(bvals[i_rep*len(bvals_unique)+i_b-2:i_rep*len(bvals_unique)+i_b+1],
                                                                    Svox[i_rep*len(bvals_unique)+i_b-2:i_rep*len(bvals_unique)+i_b+1]/Svox[i_rep*len(bvals_unique)])
                        D[i_rep], S0[i_rep] = fitPoly.coef[0], fitPoly.coef[1]

                        # # bi-exponential fit
                        # if i_b <= len(bvals)-6:
                        #     first_bval = i_b
                        # else:
                        #     first_bval = len(bvals)-6
                        # ivim_fit = ivim_fitting.IVIMfit(bvals=bvals[i_rep*len(bvals_unique)+first_bval:i_rep*len(bvals_unique)+first_bval+6],
                        #                                 voxels_values=np.array([Svox[i_rep*len(bvals_unique)+first_bval:i_rep*len(bvals_unique)+first_bval+1]/Svox[i_rep*len(bvals_unique)]]),
                        #                                 voxels_idx=tuple(idx_vox),
                        #                                 ofit_dir='',
                        #                                 model='one-step',
                        #                                 multithreading=0,
                        #                                 save_plots=False)
                        # ivim_fit.run_fit()
                        # D[i_rep], S0[i_rep] = ivim_fit.ivim_metrics_all_voxels[0]["D"], ivim_fit.ivim_metrics_all_voxels[0]["S0"]
                        # del ivim_fit

                    data[i_case]["D across reps by vox"][i_vox, i_b, :] = [np.mean(D), np.std(D)]
                    data[i_case]["S0 across reps by vox"][i_vox, i_b, :] = [np.mean(S0), np.std(S0)]

            elif len(bvals) % len(bvals_unique) != 0:
                 warnings.warn("Not the same number of b-values is available for each repetition ==> the repetition analysis cannot be performed.")

        # Get the stats (mean and SD) across all voxels in the ROI
        data[i_case]["D based average across all voxels"] = np.array([np.mean(data[i_case]["D based average by vox"]), np.std(data[i_case]["D based average by vox"])])
        data[i_case]["S0 based average across all voxels"] = np.array([np.mean(data[i_case]["S0 based average by vox"]), np.std(data[i_case]["S0 based average by vox"])])
        data[i_case]["S across reps across all voxels"] = np.mean(data[i_case]["S across reps by vox"], axis=0)
        data[i_case]["log(S/S0) across reps across all voxels"] = np.mean(data[i_case]["log(S/S0) across reps by vox"], axis=0)
        data[i_case]["D across reps across all voxels"] = np.mean(data[i_case]["D across reps by vox"], axis=0)
        data[i_case]["S0 across reps across all voxels"] = np.mean(data[i_case]["S0 across reps by vox"], axis=0)


    # -------------
    # print results
    # -------------
    print("Mean+/-SD across voxels of D based on the average signal across repetitions for each case\n"
          "==================================================================================\n")
    for i_case in range(nCases):
        print("{}\n------------\n{:.3e} +/- {:.3e}".format(oPlotNames[i_case], data[i_case]["D based average across all voxels"][0], data[i_case]["D based average across all voxels"][1])+"\n\n")
    if analysis == "rep":
        print("Mean across voxels of Mean +/- SD across repetitions of D for each b-value and case\n"
              "=========================================================\n"
              +str(bvals_unique)+"\n")
        for i_case in range(nCases):
            print("{}\n------------\n{}\n+/-{}\n\n".format(oPlotNames[i_case], data[i_case]["D across reps across all voxels"][:, 0], data[i_case]["D across reps across all voxels"][:, 1]))

    # plot results
    # ------------
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), num="Fit based on the average across repetitions")
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.9, hspace=0.3, bottom=0.1, top=0.85)
    jet_cycler = cycler(color=plt.cm.jet(np.linspace(0, 1, nCases)))
    axes[0, 0].set_prop_cycle(jet_cycler)
    axes[0, 1].set_prop_cycle(jet_cycler)
    axes[1, 0].set_prop_cycle(jet_cycler)
    axes[1, 1].set_prop_cycle(jet_cycler)
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.jet(np.repeat(np.linspace(0, 1, nCases), 2)))
    xp = np.linspace(bvals_unique[0], bvals_unique[-1], 1000)
    for i_case in range(nCases):

        # mean signal
        plot = axes[0, 0].errorbar(bvals_unique, data[i_case]["S across reps across all voxels"][:, 0],
                                yerr=data[i_case]["S across reps across all voxels"][:, 1],
                                elinewidth=0.3,
                                marker='.',
                                markersize=8.,
                                lw=0,
                                label=oPlotNames[i_case])
        axes[0, 0].plot(xp, data[i_case]["S0 based average across all voxels"][0]*np.exp(data[i_case]["D based average across all voxels"][0]*xp),
                        '-',
                        linewidth=0.5,
                        color=plot[0].get_color())  # fit
        axes[0, 0].set(xlabel='b-values (s/mm$^2$)', ylabel='S', title='Mean raw signal across repetitions')

        # log
        plotLog = axes[0, 1].errorbar(bvals_unique, data[i_case]["log(S/S0) across reps across all voxels"][:, 0],
                                   yerr=data[i_case]["log(S/S0) across reps across all voxels"][:, 1],
                                   elinewidth=0.3,
                                   marker='.',
                                   markersize=8.,
                                   lw=0,
                                   label=oPlotNames[i_case])
        axes[0, 1].plot(xp, data[i_case]["D based average across all voxels"][0]*xp+data[i_case]["S0 based average across all voxels"][0],
                        '-',
                        linewidth=0.5,
                        color=plotLog[0].get_color())  # fit
        axes[0, 1].set(xlabel='b-values (s/mm$^2$)', ylabel='ln(S\S0)', title='Mean voxel-wise ln(S/S0) and SD across repetitions')

        # SD across repetitions
        axes[1, 0].plot(bvals_unique, 100*data[i_case]["S across reps across all voxels"][:, 1]/data[i_case]["S across reps across all voxels"][:, 0],
                                marker='.',
                                markersize=8.,
                                lw=0.5,
                                label=oPlotNames[i_case])
        axes[1, 0].set(xlabel='b-values (s/mm$^2$)', ylabel='SD across repetitions (% of mean)')
        axes[1, 1].plot(bvals_unique, 100*data[i_case]["log(S/S0) across reps across all voxels"][:, 1],
                                marker='.',
                                markersize=8.,
                                lw=0.5,
                                label=oPlotNames[i_case])
        axes[1, 1].set(xlabel='b-values (s/mm$^2$)', ylabel='SD across repetitions')

        axes[0, 0].grid()
        axes[0, 1].grid()
        axes[0, 0].legend()
        fig.suptitle(title, fontsize=20)
        fig.savefig(oPlotNames[i_case]+".png")
        # plt.close()

    if analysis == "rep":
        for i_case in range(nCases):

            figRep, axesRep = plt.subplots(1, 2, figsize=(17, 9.5), num="Analysis of D estimation by repetition")
            plt.subplots_adjust(wspace=0.3, left=0.1, right=0.9, hspace=0.3, bottom=0.1, top=0.85)
            jet_cycler = cycler(color=plt.cm.jet(np.linspace(0, 1, len(bvals_unique)-2), 2))
            axesRep[0].set_prop_cycle(jet_cycler)
            axesRep[1].set_prop_cycle(jet_cycler)
            # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.jet(np.repeat(np.linspace(0, 1, len(bvals_unique)-2), 2)))

            for i_b in range(2, len(bvals_unique)):
                # mean signal
                plotRep = axesRep[0].plot(bvals_unique[i_b], data[i_case]["log(S/S0) across reps across all voxels"][i_b, 0],
                                marker='.',
                                markersize=10.,
                                lw=0,
                                label=oPlotNames[i_case] if i_b==5 else "_nolegend_")
                # linear fit
                axesRep[0].plot(xp, data[i_case]["D across reps across all voxels"][i_b, 0]*xp+data[i_case]["S0 across reps across all voxels"][i_b, 0], '-', linewidth=0.5, color=plotRep[0].get_color())
            axesRep[0].set(xlabel='b-values (s/mm$^2$)', ylabel='ln(S/S0)', title='Mean estimated D by b-value')
            axesRep[0].legend()

            # plot SD on D
            axesRep[1].bar(bvals_unique, -100*data[i_case]["D across reps across all voxels"][:, 1]/np.mean(data[i_case]["D across reps across all voxels"][2:, 0]), width=45)
            axesRep[1].set(xlabel='b-values (s/mm$^2$)', ylabel='SD (% of the mean)', title='SD on D across repetitions')
            figRep.suptitle(title, fontsize=20)
            figRep.savefig(oPlotNames[i_case]+"_repAnalysis.png")
            # plt.close()

    plt.show()

    print('**** Done ****')


def fit_D_only(x, y):
    polyfit = np.poly1d(np.polyfit(x, np.log(y), 1))

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

    optionalArgs.add_argument('-analysis', dest='analysis', help='Optional argument to perform a specific analysis: \"rep\" will estimate the SD on D across repetitions.', type=str, required=False, default="")
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
    main(maskFnames=args.maskFnames.split(','), dwiFnames=args.dwiFnames.split(','), bvalFnames=args.bvalFnames.split(','), oPlotNames=args.oPlotNames.split(','), analysis=args.analysis, title=args.title)




