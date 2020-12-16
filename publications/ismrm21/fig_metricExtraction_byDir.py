#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plot IVIM params values within regions of interest (Fig. 7).


Created on Tue Jul  4 17:45:43 2017

@author: slevy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker
from pathlib import Path
home = str(Path.home())

hc_id_nb = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
dcm_id_nb = [1, 2]
hcDir = home+'/job/data/zurich/3T/mean_hc/extract_metric'
dcmDir = home+'/job/data/zurich/3T/mean_dcm/extract_metric'
fig_dir = '.'
legend_labels = ['Spinal cord', 'White Matter (WM)', 'Gray Matter (GM)']
xticklabels = ['HC group', 'DCM1', 'DCM2']
diffDirs = ['phase', 'read', 'slice']
diffDirsLabels = ['A-P', 'R-L', 'I-S']

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$\\mathdefault{%s}$' % self.format


def main():
    """Main."""


    def plot_v3():
        """"""

        n_roi = len(legend_labels)

        fig, axes = plt.subplots(2, 2, figsize=(21, 10.5))
        plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.11, left=0.04, right=0.97)
        fig.suptitle('IVIM parameters value in healthy controls (N='+str(len(hc_id_nb))+') and two DCM patients', y=0.05, x=0.50, fontsize=25)
        cs = 3
        mew = 2
        ms_roi = 0
        roiColors = ['tab:green', 'tab:blue', 'tab:red']
        dirMarkers = ['.', '^', 'x']
        dirMarkersSize = [15, 11, 15]

        for i_metric in range(len(ivimParams_hc)):
            for i_dir in range(len(diffDirs)):

                # average across subjects HC
                # ---------------------------
                metric_mean, metric_std = np.mean(ivimParams_hc[i_metric][i_dir], axis=0), np.std(ivimParams_hc[i_metric][i_dir], axis=0)
                for i_roi in range(len(legend_labels)):

                    # plot HC
                    axes[np.unravel_index(i_metric, axes.shape)].errorbar([i_roi+0.3*i_dir-.05], metric_mean[i_roi], yerr=metric_std[i_roi], color=roiColors[i_roi], marker=dirMarkers[i_dir], markersize=dirMarkersSize[i_dir], capsize=cs, linewidth=2, fmt='.')

                    # plot DCM patients
                    for i_patient in range(len(xticklabels)-1):
                        axes[np.unravel_index(i_metric, axes.shape)].errorbar([(i_patient+1)*4 + i_roi + 0.3*i_dir -.05], ivimParams_dcm[i_metric][i_dir][i_patient, i_roi], color=roiColors[i_roi], marker=dirMarkers[i_dir], markersize=dirMarkersSize[i_dir], capsize=cs, linewidth=2, fmt='.')

            # set axis layout
            # ----------------
            axes[np.unravel_index(i_metric, axes.shape)].set_xticks([1.3, 5.3, 9.3])
            axes[np.unravel_index(i_metric, axes.shape)].set_xticklabels(xticklabels)
            axes[np.unravel_index(i_metric, axes.shape)].set_xlim((-1+0.6, 10.65))
            # ax1.set_xlabel('D$^*$ (mm$^2$/s)')
            # ax1.set_ylabel('f$_{IVIM}$ (fraction)')
            axes[np.unravel_index(i_metric, axes.shape)].yaxis.grid(which='both')
            axes[np.unravel_index(i_metric, axes.shape)].spines['right'].set_visible(False)
            axes[np.unravel_index(i_metric, axes.shape)].spines['top'].set_visible(False)
            axes[np.unravel_index(i_metric, axes.shape)].yaxis.set_ticks_position('left')
            axes[np.unravel_index(i_metric, axes.shape)].xaxis.set_ticks_position('bottom')
            axes[np.unravel_index(i_metric, axes.shape)].set_title(paramLabels[i_metric])
            if paramLabels[i_metric] != 'f$_{IVIM}$ (%)':
                axes[np.unravel_index(i_metric, axes.shape)].ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
                axes[np.unravel_index(i_metric, axes.shape)].yaxis.major.formatter._useMathText = True
            if paramLabels[i_metric] == 'D (mm$^2$/s)':
                axes[np.unravel_index(i_metric, axes.shape)].set_ylim(bottom=0.2e-3)

        # fake plots for the legend
        # ROIs
        line_rois = []
        for i_roi in range(len(roiColors)):
            line = axes[0, 0].bar([-10], [0], color=roiColors[i_roi])
            line_rois.append(line)
        fig.legend(line_rois, legend_labels, loc='lower left', ncol=3, labelspacing=1, numpoints=1, fancybox=True, shadow=True, bbox_to_anchor=(0.1, 0.9), handletextpad=0.1, prop={'size': 21})
        # diffusion-encoding directions
        line_dirs = []
        for i_dir in range(len(diffDirs)):
            line = axes[-1, -1].errorbar([-10], [0], yerr=[0], color='k', marker=dirMarkers[i_dir], markersize=dirMarkersSize[i_dir], fmt=dirMarkers[i_dir], label=diffDirsLabels[i_dir])
            line_dirs.append(line)
        axes[-1, -1].legend(line_dirs, diffDirsLabels, ncol=3, loc='upper right',  labelspacing=0.8, numpoints=1, fancybox=True, shadow=True, handletextpad=0.1, bbox_to_anchor=(0.7, 2.93), prop={'size':21})

        # plt.show()
        fig.savefig(fig_dir+'/fig_metricExtraction_allDirs.jpeg')
        fig.savefig(fig_dir+'/fig_metricExtraction_allDirs.pdf')
        plt.close(fig)

    # ==================================================================================================================
    # MAIN
    # ==================================================================================================================

    # Extract metric values in healthy controls
    hc_ids = ['hc'+str(nb) for nb in hc_id_nb]
    ivimParams_hc, paramLabels, roiLabels = extract_results_by_subjects(hc_ids, hcDir)

    # Extract metric values in patients
    dcm_ids = ['dcm'+str(nb) for nb in dcm_id_nb]
    ivimParams_dcm, _, _ = extract_results_by_subjects(dcm_ids, dcmDir)

    # plot results
    plt.rcParams.update({'font.size': 19})
    plot_v3()

def extract_results_by_subjects(subj_ids, results_folder):

    n_subj = len(subj_ids)

    Fivim = np.zeros((len(diffDirs)), dtype=object)
    Dstar = np.zeros((len(diffDirs)), dtype=object)
    FivimXDstar = np.zeros((len(diffDirs)), dtype=object)
    D = np.zeros((len(diffDirs)), dtype=object)
    labels = np.zeros((n_subj), dtype=object)

    for i_dir in range(len(diffDirs)):

       Fivim_dir_i_bySubj = np.zeros((n_subj), dtype=object)
       Dstar_dir_i_bySubj = np.zeros((n_subj), dtype=object)
       FivimXDstar_dir_i_bySubj = np.zeros((n_subj), dtype=object)
       D_dir_i_bySubj = np.zeros((n_subj), dtype=object)

       for i_sbj in range(n_subj):

            extract_metric_output = pd.read_csv(results_folder + '/' + diffDirs[i_dir] + '/' + subj_ids[i_sbj] + '_Fivim.pickle')
            Fivim_dir_i_bySubj[i_sbj] = np.array(100.*extract_metric_output.iloc[:, -2].array)  # set Fivim in %
            if i_dir == 0:
               labels[i_sbj] = extract_metric_output['Label']

            extract_metric_output = pd.read_csv(results_folder + '/' + diffDirs[i_dir] + '/' + subj_ids[i_sbj] + '_Dstar.pickle')
            Dstar_dir_i_bySubj[i_sbj] = extract_metric_output.iloc[:, -2].array

            extract_metric_output = pd.read_csv(results_folder + '/' + diffDirs[i_dir] + '/' + subj_ids[i_sbj] + '_FivimXDstar.pickle')
            FivimXDstar_dir_i_bySubj[i_sbj] = extract_metric_output.iloc[:, -2].array

            extract_metric_output = pd.read_csv(results_folder + '/' + diffDirs[i_dir] + '/' + subj_ids[i_sbj] + '_D.pickle')
            D_dir_i_bySubj[i_sbj] = extract_metric_output.iloc[:, -2].array

       Fivim[i_dir] = np.stack(Fivim_dir_i_bySubj, axis=0)
       Dstar[i_dir] = np.stack(Dstar_dir_i_bySubj, axis=0)
       FivimXDstar[i_dir] = np.stack(FivimXDstar_dir_i_bySubj, axis=0)
       D[i_dir] = np.stack(D_dir_i_bySubj, axis=0)

    return [Fivim, Dstar, FivimXDstar, D], ['f$_{IVIM}$ (%)', 'D$^*$ (mm$^2$/s)', 'f$_{IVIM}$D$^*$ (mm$^2$/s)', 'D (mm$^2$/s)'], labels





# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    main()

