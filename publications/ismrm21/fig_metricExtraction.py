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

hc_id_nb = [1, 3, 4, 5]
dcm_id_nb = [1, 2]
hcDir = home+'/job/data/zurich/3T/mean_hc/extract_metric'
dcmDir = home+'/job/data/zurich/3T/mean_dcm/extract_metric'
fig_dir = '.'
legend_labels = ['Spinal cord', 'White Matter (WM)', 'Gray Matter (GM)']
xticklabels = ['HC group', 'DCM1', 'DCM2']

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
        """With in-ROI standard deviations"""

        n_roi = len(legend_labels)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(21, 10.5))
        plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.11, left=0.04, right=0.97)
        fig.suptitle('IVIM parameters value in healthy controls (N='+str(len(hc_id_nb))+') and DCM patients (N='+str(len(dcm_id_nb))+')', y=0.05, x=0.50, fontsize=25)
        cs = 3
        mew = 2
        ms_roi = 0

        # Fivim
        # --------------------------------------
        # HC
        Fivim_mean, Fivim_std = np.mean(Fivim_hc, axis=0), np.std(Fivim_hc, axis=0)
        l1 = ax1.errorbar([0-.05], Fivim_mean[0], yerr=Fivim_std[0], color='tab:green', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.', label=labels[0, 0])
        l2 = ax1.errorbar([0.5-.05], Fivim_mean[1], yerr=Fivim_std[1], color='tab:blue', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.', label=labels[0, 1])
        l3 = ax1.errorbar([1-.05], Fivim_mean[2], yerr=Fivim_std[2], color='tab:red', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.', label=labels[0, 2])
        # DCM 1
        ax1.plot([2-.05], Fivim_dcm[0, 0], color='tab:green', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        ax1.plot([2.5-.05], Fivim_dcm[0, 1], color='tab:blue', marker='o', markersize=15, label='_nolegend_')
        ax1.plot([3-.05], Fivim_dcm[0, 2], color='tab:red', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        # DCM 2
        ax1.plot([4-.05], Fivim_dcm[1, 0], color='tab:green', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        ax1.plot([4.5-.05], Fivim_dcm[1, 1], color='tab:blue', marker='o', markersize=15, label='_nolegend_')
        ax1.plot([5-.05], Fivim_dcm[1, 2], color='tab:red', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        # lnone = ax1.errorbar([], [], marker=None, markersize=0, linewidth=0)  # to set legend in two columns
        # # In-ROI std
        # ax1.errorbar([0.05], Fivim_mean[1], yerr=Fivim_roi[0, 0], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[1, 0])
        # ax1.errorbar([0.55], Fivim_mean[2], yerr=Fivim_roi[0, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[2, 0])
        # ax1.errorbar([1.05], Fivim_mean[3], yerr=Fivim_roi[0, 2], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[3, 0])

        ax1.set_xticks([0.5, 2.5, 4.5])
        ax1.set_xticklabels(xticklabels)
        ax1.set_xlim(-1, 5.95)
        # ax1.set_xlabel('D$^*$ (mm$^2$/s)')
        # ax1.set_ylabel('f$_{IVIM}$ (fraction)')
        ax1.yaxis.grid(which='both')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.set_title('f$_{IVIM}$ (%)')

        # Dstar
        # --------------------------------------
        # HC
        Dstar_mean, Dstar_std = np.mean(Dstar_hc, axis=0), np.std(Dstar_hc, axis=0)
        ax2.errorbar([0-.05], Dstar_mean[0], yerr=Dstar_std[0], color='tab:green', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.')
        ax2.errorbar([0.5-.05], Dstar_mean[1], yerr=Dstar_std[1], color='tab:blue', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.')
        ax2.errorbar([1-.05], Dstar_mean[2], yerr=Dstar_std[2], color='tab:red', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.')
        # DCM 1
        ax2.plot([2-.05], Dstar_dcm[0, 0], color='tab:green', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        ax2.plot([2.5-.05], Dstar_dcm[0, 1], color='tab:blue', marker='o', markersize=15, label='_nolegend_')
        ax2.plot([3-.05], Dstar_dcm[0, 2], color='tab:red', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        # DCM 2
        ax2.plot([4-.05], Dstar_dcm[1, 0], color='tab:green', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        ax2.plot([4.5-.05], Dstar_dcm[1, 1], color='tab:blue', marker='o', markersize=15, label='_nolegend_')
        ax2.plot([5-.05], Dstar_dcm[1, 2], color='tab:red', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        # # In-ROI std
        # ax2.errorbar([0.05], Dstar_mean[1], yerr=Dstar_std[0], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[1, 0])
        # ax2.errorbar([0.55], Dstar_mean[2], yerr=Dstar_std[1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[2, 0])
        # ax2.errorbar([1.05], Dstar_mean[3], yerr=Dstar_std[2], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[3, 0])

        ax2.set_xticks([0.5, 2.5, 4.5])
        ax2.set_xticklabels(xticklabels)
        ax2.set_xlim(-1, 5.95)
        # ax2.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # ax2.yaxis.get_major_formatter().set_powerlimits((-3, -3))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax2.yaxis.major.formatter._useMathText = True
        # ax2.set_xlabel('D$^*$ (mm$^2$/s)')
        # ax2.set_ylabel('f$_{IVIM}$ (fraction)')
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.grid(which='both')
        ax2.set_title('D$^*$ (mm$^2$/s)')

        # FivimXDstar
        # --------------------------------------
        # HC
        FivimXDstar_mean, FivimXDstar_std = np.mean(FivimXDstar_hc, axis=0), np.std(FivimXDstar_hc, axis=0)
        ax3.errorbar([0-.05], FivimXDstar_mean[0], yerr=FivimXDstar_std[0], color='tab:green', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.')
        ax3.errorbar([0.5-.05], FivimXDstar_mean[1], yerr=FivimXDstar_std[1], color='tab:blue', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.')
        ax3.errorbar([1-.05], FivimXDstar_mean[2], yerr=FivimXDstar_std[2], color='tab:red', marker='o', markersize=15, capsize=cs, linewidth=2, fmt='.')
        # DCM 1
        ax3.plot([2-.05], FivimXDstar_dcm[0, 0], color='tab:green', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        ax3.plot([2.5-.05], FivimXDstar_dcm[0, 1], color='tab:blue', marker='o', markersize=15, label='_nolegend_')
        ax3.plot([3-.05], FivimXDstar_dcm[0, 2], color='tab:red', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        # DCM 2
        ax3.plot([4-.05], FivimXDstar_dcm[1, 0], color='tab:green', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        ax3.plot([4.5-.05], FivimXDstar_dcm[1, 1], color='tab:blue', marker='o', markersize=15, label='_nolegend_')
        ax3.plot([5-.05], FivimXDstar_dcm[1, 2], color='tab:red', marker='o', markersize=15, linewidth=2, label='_nolegend_')
        # # In-ROI std
        # ax3.errorbar([0.05], FivimXDstar_mean[1], yerr=FivimXDstar_std[0], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[1, 0])
        # ax3.errorbar([0.55], FivimXDstar_mean[2], yerr=FivimXDstar_std[1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[2, 0])
        # ax3.errorbar([1.05], FivimXDstar_mean[3], yerr=FivimXDstar_std[2], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.', label=labels[3, 0])

        ax3.set_xticks([0.5, 2.5, 4.5])
        ax3.set_xticklabels(xticklabels)
        ax3.set_xlim(-1, 5.95)
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax3.yaxis.major.formatter._useMathText = True
        # ax3.set_xlabel('D$^*$ (mm$^2$/s)')
        # ax3.set_ylabel('f$_{IVIM}$ (fraction)')
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.grid(which='both')
        ax3.set_title('f$_{IVIM}$D$^*$ (mm$^2$/s)')

        # Dphase, Dread, Dslice
        # ------------------------------------------------------------------------
        # HC
        # ------------------------------------------------------------------------
        Dphase_mean, Dphase_std = np.mean(Dphase_hc, axis=0), np.std(Dphase_hc, axis=0)
        Dread_mean, Dread_std = np.mean(Dread_hc, axis=0), np.std(Dread_hc, axis=0)
        Dslice_mean, Dslice_std = np.mean(Dslice_hc, axis=0), np.std(Dslice_hc, axis=0)
        # ax4.set_aspect('equal', 'box')
        # in SC
        ax4.errorbar([0-.05], Dphase_mean[0], yerr=Dphase_std[0], color='tab:green', marker='s', markersize=6, capsize=cs, linewidth=2)
        ax4.errorbar([0.3-.05], Dread_mean[0], yerr=Dread_std[0], color='tab:green', marker='^', markersize=10, capsize=cs, linewidth=2)
        ax4.errorbar([0.6-.05], Dslice_mean[0], yerr=Dslice_std[0], color='tab:green', marker='x', markersize=13, capsize=cs, linewidth=2)
        # in WM
        ax4.errorbar([1-.05], Dphase_mean[1], yerr=Dphase_std[1], color='tab:blue', marker='s', markersize=6, capsize=cs, linewidth=2)
        ax4.errorbar([1.3-.05], Dread_mean[1], yerr=Dread_std[1], color='tab:blue', marker='^', markersize=10, capsize=cs, linewidth=2)
        ax4.errorbar([1.6-.05], Dslice_mean[1], yerr=Dslice_std[1], color='tab:blue', marker='x', markersize=13, capsize=cs, linewidth=2)
        # in GM
        ax4.errorbar([2-.05], Dphase_mean[2], yerr=Dphase_std[2], color='tab:red', marker='s', markersize=6, capsize=cs, linewidth=2)
        ax4.errorbar([2.3-.05], Dread_mean[2], yerr=Dread_std[2], color='tab:red', marker='^', markersize=10, capsize=cs, linewidth=2)
        ax4.errorbar([2.6-.05], Dslice_mean[2], yerr=Dslice_std[2], color='tab:red', marker='x', markersize=13, capsize=cs, linewidth=2)

        # DCM 1
        # ------------------------------------------------------------------------
        # in SC
        ax4.plot([4-.05], Dphase_dcm[0, 0], color='tab:green', marker='s', markersize=6, linewidth=2)
        ax4.plot([4.3-.05], Dread_dcm[0, 0], color='tab:green', marker='^', markersize=10, linewidth=2)
        ax4.plot([4.6-.05], Dslice_dcm[0, 0], color='tab:green', marker='x', markersize=13, linewidth=2)
        # in WM
        ax4.plot([5-.05], Dphase_dcm[0, 1], color='tab:blue', marker='s', markersize=6, linewidth=2)
        ax4.plot([5.3-.05], Dread_dcm[0, 1], color='tab:blue', marker='^', markersize=10, linewidth=2)
        ax4.plot([5.6-.05], Dslice_dcm[0, 1], color='tab:blue', marker='x', markersize=13, linewidth=2)
        # in GM
        ax4.plot([6-.05], Dphase_dcm[0, 2], color='tab:red', marker='s', markersize=6, linewidth=2)
        ax4.plot([6.3-.05], Dread_dcm[0, 2], color='tab:red', marker='^', markersize=10, linewidth=2)
        ax4.plot([6.6-.05], Dslice_dcm[0, 2], color='tab:red', marker='x', markersize=13, linewidth=2)

        # DCM 2
        # ------------------------------------------------------------------------
        # in SC
        ax4.plot([8-.05], Dphase_dcm[1, 0], color='tab:green', marker='s', markersize=6, linewidth=2)
        ax4.plot([8.3-.05], Dread_dcm[1, 0], color='tab:green', marker='^', markersize=10, linewidth=2)
        ax4.plot([8.6-.05], Dslice_dcm[1, 0], color='tab:green', marker='x', markersize=13, linewidth=2)
        # in WM
        ax4.plot([9-.05], Dphase_dcm[1, 1], color='tab:blue', marker='s', markersize=6, linewidth=2)
        ax4.plot([9.3-.05], Dread_dcm[1, 1], color='tab:blue', marker='^', markersize=10, linewidth=2)
        ax4.plot([9.6-.05], Dslice_dcm[1, 1], color='tab:blue', marker='x', markersize=13, linewidth=2)
        # in GM
        ax4.plot([10-.05], Dphase_dcm[1, 2], color='tab:red', marker='s', markersize=6, linewidth=2)
        ax4.plot([10.3-.05], Dread_dcm[1, 2], color='tab:red', marker='^', markersize=10, linewidth=2)
        ax4.plot([10.6-.05], Dslice_dcm[1, 2], color='tab:red', marker='x', markersize=13, linewidth=2)

        # # In-ROI std
        # ax4.errorbar([0.05], Dphase_mean[1], yerr=Dphase_roi[1, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([1.05], Dphase_mean[2], yerr=Dphase_roi[2, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([2.05], Dphase_mean[3], yerr=Dphase_roi[3, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([3.05], Dphase_mean[4], yerr=Dphase_roi[4, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([4.05], Dphase_mean[5], yerr=Dphase_roi[5, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([5.05], Dphase_mean[6], yerr=Dphase_roi[6, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([6.05], Dphase_mean[7], yerr=Dphase_roi[7, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        #
        # ax4.errorbar([0.35], Dread_mean[1], yerr=Dread_roi[1, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([1.35], Dread_mean[2], yerr=Dread_roi[2, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([2.35], Dread_mean[3], yerr=Dread_roi[3, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([3.35], Dread_mean[4], yerr=Dread_roi[4, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([4.35], Dread_mean[5], yerr=Dread_roi[5, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([5.35], Dread_mean[6], yerr=Dread_roi[6, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([6.35], Dread_mean[7], yerr=Dread_roi[7, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        #
        # ax4.errorbar([0.65], Dslice_mean[1], yerr=Dslice_roi[1, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([1.65], Dslice_mean[2], yerr=Dslice_roi[2, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([2.65], Dslice_mean[3], yerr=Dslice_roi[3, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([3.65], Dslice_mean[4], yerr=Dslice_roi[4, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([4.65], Dslice_mean[5], yerr=Dslice_roi[5, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([5.65], Dslice_mean[6], yerr=Dslice_roi[6, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        # ax4.errorbar([6.65], Dslice_mean[7], yerr=Dslice_roi[7, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')

        ax4.set_xticks([1.3, 5.3, 9.3])
        ax4.set_xticklabels(xticklabels)
        ax4.set_xlim(-1+0.6, 10.65)
        ax4.set_ylim(bottom=0.2e-3)
        ax4.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax4.yaxis.major.formatter._useMathText = True
        # ax4.set_xlabel('D$^*$ (mm$^2$/s)')
        # ax4.set_ylabel('f$_{IVIM}$ (fraction)')
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.yaxis.set_ticks_position('left')
        ax4.xaxis.set_ticks_position('bottom')
        ax4.yaxis.grid(which='both')
        ax4.set_title('D (mm$^2$/s)', x=0.4, y=1.10)
        l_Dphase = ax4.errorbar([-10], [0], yerr=[0], color='k', marker='s', markersize=7, fmt='s', label='D$_{A-P}$')
        l_Dread = ax4.errorbar([-10], [0], yerr=[0], color='k', marker='^', markersize=10, fmt='^', label='D$_{R-L}$')
        l_Dslice = ax4.errorbar([-10], [0], yerr=[0], color='k', marker='x', markersize=10, fmt='x', label='D$_{I-S}$')
        ax4.legend((l_Dphase, l_Dread, l_Dslice), ['D$_{A-P}$', 'D$_{R-L}$', 'D$_{I-S}$'], ncol=3, loc='upper right',  labelspacing=0.8, numpoints=1, fancybox=True, shadow=True, handletextpad=0.1, bbox_to_anchor=(1.05, 1.27), prop={'size':18})

        fig.legend((l1, l2, l3), legend_labels, loc='upper center', ncol=3, labelspacing=1, numpoints=1, fancybox=True, shadow=True, handletextpad=0.1, prop={'size': 21})

        # plt.show()
        fig.savefig(fig_dir+'/fig_metricExtraction.png')
        fig.savefig(fig_dir+'/fig_metricExtraction.pdf')
        plt.close(fig)

        # plot fig for legend
        fig_leg, ax_leg = plt.subplots(1, 1)
        ax_leg.errorbar([0 - .05], FivimXDstar_mean[0], yerr=FivimXDstar_std[0], marker='.', markersize=20, capsize=0, linewidth=2, color='k',)
        # ax_leg.errorbar([0.05], FivimXDstar_mean[0], yerr=FivimXDstar_roi[0, 1], color='gray', marker='.', markersize=ms_roi, capsize=cs, markeredgewidth=mew, fmt='.')
        ax_leg.set_xlim(-1, n_roi-5)
        fig_leg.savefig(fig_dir+'/fig_metricExtraction_legend.png')


    # Extract metric values in healthy controls
    hc_ids = ['hc'+str(nb) for nb in hc_id_nb]
    Fivim_hc, Dstar_hc, FivimXDstar_hc, Dphase_hc, Dread_hc, Dslice_hc, labels = extract_results_by_subjects(hc_ids, hcDir)

    # Extract metric values in patients
    dcm_ids = ['dcm'+str(nb) for nb in dcm_id_nb]
    Fivim_dcm, Dstar_dcm, FivimXDstar_dcm, Dphase_dcm, Dread_dcm, Dslice_dcm, _ = extract_results_by_subjects(dcm_ids, dcmDir)

    # plot results
    plt.rcParams.update({'font.size': 19})
    plot_v3()

def extract_results_by_subjects(subj_ids, results_folder):

    n_subj = len(subj_ids)

    Fivim = np.zeros((n_subj), dtype=object)
    Dstar = np.zeros((n_subj), dtype=object)
    FivimXDstar = np.zeros((n_subj), dtype=object)
    Dphase = np.zeros((n_subj), dtype=object)
    Dread = np.zeros((n_subj), dtype=object)
    Dslice = np.zeros((n_subj), dtype=object)
    labels = np.zeros((n_subj), dtype=object)
    for i_sbj in range(n_subj):

        extract_metric_output = pd.read_csv(results_folder + '/' + subj_ids[i_sbj] + '_Fivim.pickle')
        Fivim[i_sbj] = np.array(100.*extract_metric_output.iloc[:, -2].array)  # set Fivim in %
        labels[i_sbj] = extract_metric_output['Label']

        extract_metric_output = pd.read_csv(results_folder + '/' + subj_ids[i_sbj] + '_Dstar.pickle')
        Dstar[i_sbj] = extract_metric_output.iloc[:, -2].array

        extract_metric_output = pd.read_csv(results_folder + '/' + subj_ids[i_sbj] + '_FivimXDstar.pickle')
        FivimXDstar[i_sbj] = extract_metric_output.iloc[:, -2].array

        extract_metric_output = pd.read_csv(results_folder + '/' + subj_ids[i_sbj] + '_Dphase.pickle')
        Dphase[i_sbj] = extract_metric_output.iloc[:, -2].array

        extract_metric_output = pd.read_csv(results_folder + '/' + subj_ids[i_sbj] + '_Dread.pickle')
        Dread[i_sbj] = extract_metric_output.iloc[:, -2].array

        extract_metric_output = pd.read_csv(results_folder + '/' + subj_ids[i_sbj] + '_Dslice.pickle')
        Dslice[i_sbj] = extract_metric_output.iloc[:, -2].array

    # # get metric results (and ROI std) on mean maps
    # Fivim_roi = np.empty((n_roi, 2))
    # f = open('Fivim_subjects_averaged_map.pickle', 'r')
    # extract_metric_output = pickle.load(f)
    # Fivim_roi[:, 0] = 100.*extract_metric_output['Metric value']  # set Fivim in %
    # Fivim_roi[:, 1] = 100.*extract_metric_output['Metric STDEV within label']  # set Fivim in %
    #
    # Dstar_roi = np.empty((n_roi, 2))
    # f = open('Dstar_subjects_averaged_map.pickle', 'r')
    # extract_metric_output = pickle.load(f)
    # Dstar_roi[:, 0] = extract_metric_output['Metric value']
    # Dstar_roi[:, 1] = extract_metric_output['Metric STDEV within label']
    #
    # FivimXDstar_roi = np.empty((n_roi, 2))
    # f = open('FivimXDstar_subjects_averaged_map.pickle', 'r')
    # extract_metric_output = pickle.load(f)
    # FivimXDstar_roi[:, 0] = extract_metric_output['Metric value']
    # FivimXDstar_roi[:, 1] = extract_metric_output['Metric STDEV within label']
    #
    # Dphase_roi = np.empty((n_roi, 2))
    # f = open('D_phase_subjects_averaged_map.pickle', 'r')
    # extract_metric_output = pickle.load(f)
    # Dphase_roi[:, 0] = extract_metric_output['Metric value']
    # Dphase_roi[:, 1] = extract_metric_output['Metric STDEV within label']
    #
    # Dread_roi = np.empty((n_roi, 2))
    # f = open('D_read_subjects_averaged_map.pickle', 'r')
    # extract_metric_output = pickle.load(f)
    # Dread_roi[:, 0] = extract_metric_output['Metric value']
    # Dread_roi[:, 1] = extract_metric_output['Metric STDEV within label']
    #
    # Dslice_roi = np.empty((n_roi, 2))
    # f = open('D_slice_subjects_averaged_map.pickle', 'r')
    # extract_metric_output = pickle.load(f)
    # Dslice_roi[:, 0] = extract_metric_output['Metric value']
    # Dslice_roi[:, 1] = extract_metric_output['Metric STDEV within label']

    return np.stack(Fivim, axis=0), np.stack(Dstar, axis=0), np.stack(FivimXDstar, axis=0), np.stack(Dphase, axis=0), np.stack(Dread, axis=0), np.stack(Dslice, axis=0), np.stack(labels, axis=0) #, Fivim_roi, Dstar_roi, FivimXDstar_roi, Dphase_roi, Dread_roi, Dslice_roi





# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    main()

