import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from skimage.measure import regionprops
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
import subprocess
import pandas as pd
# import sct_extract_metric
import _pickle as pickle
from ivim_fit_phantom_data import fit_D_only
import ivim_fitting

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================

# parameters to draw signal profiles
# ----------------------------------
subjDataBaseFolder = str(Path.home())+"/job/data/zurich/3T"
subj_ids = ['hc1', 'hc3', 'hc4', 'hc5', 'hc6', 'hc7', 'hc8', 'hc9', 'hc10', 'hc11']
diffDirs = ['phase', 'read', 'slice']
diffDirsLabels = ['A-P', 'R-L', 'I-S']
roiLabels = ['Spinal Cord', 'White Matter', 'Gray Matter']
roiColors = ['tab:green', 'tab:blue', 'tab:red']
foregroundColor="white"
extractAgain = False

# ======================================================================================================================

# --------------------------------------------------------------
# Extract signal values across all subjects
# --------------------------------------------------------------

if extractAgain:
    # array to store results
    signal_bySubj_byDir_byBval = np.zeros((len(subj_ids), len(diffDirs)), dtype=object)

    # make temporary directory to work in
    tmpDirFname = "%s/tpmDir%s" % (os.path.dirname(os.path.abspath(__file__)), datetime.now().strftime("%y%m%d%H%M%S%f")[0:13])
    os.makedirs(tmpDirFname)

    for i_subj in range(len(subj_ids)):

        for i_dir in range(len(diffDirs)):

            bvals = np.loadtxt(subjDataBaseFolder+'/'+subj_ids[i_subj]+'/'+diffDirs[i_dir]+'/fwd_mean.bval', delimiter=None)
            signal_byBval = np.zeros((len(bvals), len(roiLabels)))

            # split b-value volumes
            subprocess.call('fslsplit '+subjDataBaseFolder+'/'+subj_ids[i_subj]+'/'+diffDirs[i_dir]+'/topup_mean.nii.gz '+tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b -t', shell=True)

            for i_b in range(len(bvals)):
                # extract signal value in spinal cord based on the eroded atlas
                ib_4d = "%04d" % i_b
                cmd='sct_extract_metric -method wa' \
                    ' -i '+tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.nii.gz' \
                    ' -f '+subjDataBaseFolder+'/'+subj_ids[i_subj]+'/'+'phase/label/atlas_eroded/' \
                    ' -o '+tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.csv' \
                    ' -l 50,51,52' \
                    ' -z 0:8'
                subprocess.call(cmd, shell=True, env=os.environ)
                # sct_extract_metric.main(fname_data=tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.nii.gz',
                #                         path_label=subjDataBaseFolder+'/'+subj_ids[i_subj]+'/'+'phase/label/atlas_eroded/',
                #                         fname_output=tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.csv',
                #                         labels_user='50',
                #                         slices=[0, 1, 2, 3, 4, 5, 6, 7, 8])

                # store value
                extract_metric_output = pd.read_csv(tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.csv')
                signal_byBval[i_b, :] = extract_metric_output.iloc[:, -2]

                # remove useless files
                os.remove(tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.nii.gz')
                os.remove(tmpDirFname+'/'+subj_ids[i_subj]+'_'+diffDirs[i_dir]+'_b'+ib_4d+'.csv')

            # store values for all b-values
            signal_bySubj_byDir_byBval[i_subj, i_dir] = signal_byBval/signal_byBval[0]

    # remove temporary directory
    os.rmdir(tmpDirFname)

    # save results in pickle file
    pickle.dump({"signal": signal_bySubj_byDir_byBval, "bvals": bvals, "roiLabels": roiLabels}, open(os.path.dirname(os.path.abspath(__file__)) + '/fig_signalProfiles_data.pickle', 'wb'))

# load results in pickle file
signal_extraction_results = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + '/fig_signalProfiles_data.pickle', 'rb'))
bvals = signal_extraction_results["bvals"]

# calculate mean across subjects
signal_bySubj_byDir_byBval = np.zeros((len(subj_ids), len(diffDirs), len(bvals), len(roiLabels)))
for i_subj in range(len(subj_ids)):
    for i_dir in range(len(diffDirs)):
        signal_bySubj_byDir_byBval[i_subj, i_dir, :, :] = signal_extraction_results["signal"][i_subj, i_dir]
signal_byDir_byBval = np.mean(signal_bySubj_byDir_byBval, axis=0)

# --------------------------------------------------------------
# make figure
# --------------------------------------------------------------

# with WM and GM
# ---------------
fig, axes = plt.subplots(1, 1, figsize=(20, 8.7))

dirMarkers = ['.', '^', 'x']
dirMarkersSize = [15, 11, 15]
for i_dir in range(len(diffDirs)):
    for i_roi in range(1, len(roiLabels)):
        axes.plot(bvals, np.log(signal_byDir_byBval[i_dir, :, i_roi]), lw=0.8, ls='--', marker=dirMarkers[i_dir], ms=dirMarkersSize[i_dir], color=roiColors[i_roi])

# fake plots for the legend
# ROIs
line_rois = []
for i_roi in range(1, len(roiColors)):
    line = axes.bar([-10], [0], color=roiColors[i_roi])
    line_rois.append(line)
fig.legend(line_rois, roiLabels[1:], loc='lower left', ncol=3, labelspacing=1, numpoints=1, fancybox=True, shadow=True, bbox_to_anchor=(0.1, 0.9), handletextpad=0.1, prop={'size': 21})
# diffusion-encoding directions
line_dirs = []
for i_dir in range(len(diffDirs)):
    line = axes.errorbar([-10], [0], yerr=[0], color='k', marker=dirMarkers[i_dir], markersize=dirMarkersSize[i_dir], fmt=dirMarkers[i_dir], label=diffDirsLabels[i_dir])
    line_dirs.append(line)
axes.legend(line_dirs, diffDirsLabels, ncol=3, loc='upper right',  labelspacing=0.8, numpoints=1, fancybox=True, shadow=True, handletextpad=0.1, bbox_to_anchor=(0.7, 0.8), prop={'size':21})

# axes settings
axes.set_xlim(left=0)
axes.set_xlabel('b-values (s/mm$^2$)')
axes.set_ylabel(r'$\frac{Signal}{Signal_{b=0}}$', rotation=0)

fig.savefig(os.path.dirname(os.path.abspath(__file__)) + '/fig_hc_signalProfiles_GM_WM.jpeg') #, transparent=True)
plt.close(fig)



# with SC
# ---------------
i_roi = 0
fig, axes = plt.subplots(1, 1, figsize=(20, 6.5))
plt.subplots_adjust(wspace=0.1, left=0.2, right=0.95, hspace=0, bottom=.2, top=0.95)

fig.patch.set_facecolor('black')

dirMarkers = ['.', '^', 'x']
dirColors = ['magenta', 'cyan', 'yellow']
dirMarkersSize = [15, 11, 15]
for i_dir in range(len(diffDirs)):
    # points
    axes.plot(bvals, np.log(signal_byDir_byBval[i_dir, :, i_roi]), lw=0, ls='--', marker='.', ms=15, color=dirColors[i_dir])
    # D fit
    p_highb, r2, sum_squared_error = fit_D_only(bvals[bvals >= 400], signal_byDir_byBval[i_dir, bvals >= 400, i_roi])
    xp = np.linspace(min(bvals), max(bvals), 100)
    axes.plot(xp, p_highb(xp), lw=2.5, ls=':', color=dirColors[i_dir])
    # complete bi-exponential fit
    ivim_fit = ivim_fitting.IVIMfit(bvals=bvals,
                                    voxels_values=np.array([signal_byDir_byBval[i_dir, :, i_roi]]),
                                    voxels_idx=(np.array([0]), np.array([0]), np.array([0])),
                                    ofit_dir='',
                                    model='one-step',
                                    multithreading=0,
                                    save_plots=False)
    ivim_fit.run_fit()
    axes.plot(xp,
              np.log(ivim_fitting.ivim_1pool_model(xp, ivim_fit.ivim_metrics_all_voxels[0]["S0"], ivim_fit.ivim_metrics_all_voxels[0]["D"], ivim_fit.ivim_metrics_all_voxels[0]["Fivim"], ivim_fit.ivim_metrics_all_voxels[0]["Dstar"])),
              lw=1.5, color=dirColors[i_dir])
    del ivim_fit

# fake plots for the legend
# diffusion-encoding directions
line_dirs = []
for i_dir in range(len(diffDirs)):
    line = axes.bar([-10], [0], color=dirColors[i_dir])
    line_dirs.append(line)
axes.legend(line_dirs, diffDirsLabels, ncol=3, loc='lower left',  labelspacing=0.8, numpoints=1, fancybox=True, shadow=True, handletextpad=0.1, bbox_to_anchor=(0.0, 0.405), prop={'size':21}, facecolor="black", labelcolor=foregroundColor)
# fit types
l_points = axes.plot([-10], [0], lw=0, marker='.', ms=10, color=foregroundColor, label="Mean data across subjects")
l_linear_fit = axes.plot([-10], [0], lw=2.7, ls=':', color=foregroundColor, label="Linear fit on b-values$\geq$400s/mm$^2$")
l_biexp_fit = axes.plot([-10], [0], lw=1.5, color=foregroundColor, label="Final biexponential fit")
fig.legend((l_points[0], l_linear_fit[0], l_biexp_fit[0]), ["Mean signal across subjects", "Linear fit on b-values$\geq$400s/mm$^2$\n(Diffusion process)", "Final biexponential fit"],
           loc='lower left', ncol=1, labelspacing=1, numpoints=1, fancybox=True, shadow=True, bbox_to_anchor=(0.2, 0.185), handletextpad=0.3, prop={'size': 21}, facecolor="black", labelcolor=foregroundColor)

# axes settings
axes.set_xlim(left=-3)
axes.set_ylim(top=0.1)
# axes.set_title('Mean signal profile in Spinal Cord across subjects', fontsize=30, color=foregroundColor, y=0.99)
axes.set_xlabel('b-values (s/mm$^2$)', fontsize=27, color=foregroundColor)
axes.set_ylabel(r'$ln(\frac{Signal}{Signal_{b=0}})$', rotation=0, fontsize=33, color=foregroundColor, labelpad=100)
axes.set_facecolor('black')
axes.spines['bottom'].set_color(foregroundColor)
axes.spines['left'].set_color(foregroundColor)
axes.tick_params(colors=foregroundColor, labelsize=20)


fig.savefig(os.path.dirname(os.path.abspath(__file__)) + '/fig_hc_signalProfiles_SC.jpeg') #, transparent=True)
