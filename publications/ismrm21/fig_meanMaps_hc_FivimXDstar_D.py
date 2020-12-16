import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from skimage.measure import regionprops
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================

dataFolder = str(Path.home())+"/job/data/zurich/3T/mean_hc"

levels = [1, 2, 3]
levelsZ = [[938, 954], [915, 931], [885, 901]]
# metrics = ['mge',    'Fivim', 'Dstar',     'FivimXDstar', 'D_phase',   'D_read',    'D_slice']
# clims =   [(200, 700), (0, .3), (0, 20e-3),  (0, 1e-3),     (0, 0.5e-3), (0, 0.5e-3), (0, 2.0e-3)]
# metricsLabels = ['MEDIC', 'f$_{IVIM}$ (%)', 'D$^*$ (mm$^2$/s)', 'f$_{IVIM}$D$^*$ (mm$^2$/s)', 'D$_{A-P}$ (mm$^2$/s)', 'D$_{R-L}$ (mm$^2$/s)', 'D$_{I-S}$ (mm$^2$/s)']
metrics = ['mge',   'FivimXDstar_phase', 'FivimXDstar_read',   'FivimXDstar_slice', 'FivimXDstar',  'D_phase', 'D_read', 'D_slice', 'D']
clims =   [(200, 600), (0, 1.5e-3), (0, 1.5e-3), (0, 1.5e-3), (0, 1.5e-3), (0.3e-3, 0.45e-3), (0.3e-3, 0.45e-3), (1.4e-3, 1.9e-3), (0.75e-3, 0.85e-3)]
metricsLabels = ['MEDIC', 'A-P', 'R-L', 'I-S', 'Average', 'A-P', 'R-L', 'I-S', 'Average']
cbarLabels = [r'f$_{IVIM}\times$D$^*$ (mm$^2$/s)', 'D (mm$^2$/s)']
# ======================================================================================================================

fig, axes = plt.subplots(len(levels), len(metrics), figsize=(20, 8.7))
plt.subplots_adjust(wspace=0.1, left=0.07, right=0.99, hspace=-0.89, bottom=-0.25, top=1.00)
fig.patch.set_facecolor('black')

# load mask
SCmask_nii = nib.load(os.environ['SCT_DIR']+'/data/PAM50/template/PAM50_cord.nii.gz')
SCmask = SCmask_nii.get_data()
# define crop size
resolution = SCmask_nii.header['pixdim'][1:3]
crop_xsize = 8 / resolution[0]
crop_ysize = 8 / resolution[1]

for i_metric in range(len(metrics)):

    print('\n>>> Metric: {}'.format(metrics[i_metric]))

    # load metric
    MetricMap = nib.load(dataFolder+'/'+metrics[i_metric]+'_mean_all_subj.nii.gz').get_data()

    for i_lev in range(len(levels)):

        print('   >>> Level: C{}'.format(levels[i_lev]))

        # average across the selected slices for this level
        mask_lev_i = np.mean(SCmask[:, :, levelsZ[i_lev][0]:levelsZ[i_lev][1]], axis=2) > 0.5
        MetricMap_lev_i = np.mean(MetricMap[:, :, levelsZ[i_lev][0]:levelsZ[i_lev][1]], axis=2)

        # mask the metric map
        MetricMap_masked = np.ma.masked_where(mask_lev_i == 0.0, MetricMap_lev_i)

        # get center of mass of the slice
        properties = regionprops((mask_lev_i > 0).astype(int), MetricMap_masked)
        center_of_mass = properties[0].centroid
        weighted_center_of_mass = properties[0].weighted_centroid
        crop_xmin = int(round(center_of_mass[0]-crop_xsize, 0))
        crop_xmax = int(round(center_of_mass[0]+crop_xsize, 0))
        crop_ymin = int(round(center_of_mass[1]-crop_ysize, 0))
        crop_ymax = int(round(center_of_mass[1]+crop_ysize, 0))

        # plot image
        # -----------
        c = axes[i_lev, i_metric].imshow(np.rot90(MetricMap_masked[crop_xmin:crop_xmax+1, crop_ymin:crop_ymax+1]), cmap='jet' if metricsLabels[i_metric] != 'MEDIC' else 'gray', clim=clims[i_metric])
        # axes[i_subj, i_metric].axis("off")
        axes[i_lev, i_metric].set_frame_on(False)
        axes[i_lev, i_metric].xaxis.set_visible(False)
        axes[i_lev, i_metric].yaxis.set_ticks([])

        # add vertical line
        if i_lev == 2 and i_metric == 4:
            axline = inset_axes(axes[i_lev, i_metric],
                                width="5%",  # width = 5% of parent_bbox width
                                height="400%",  # height : 50%
                                loc='lower left',
                                bbox_to_anchor=(1.07, 0, 1, 1),
                                bbox_transform=axes[i_lev, i_metric].transAxes,
                                borderpad=0)
            axline.axvline(x=0, color="white")
            axline.set_frame_on(False)
            axline.xaxis.set_visible(False)
            axline.yaxis.set_ticks([])

        # Titles
        # --------
        if i_lev == 0: axes[i_lev, i_metric].set_title(metricsLabels[i_metric], fontsize=20, color='white', y=0.9)
        if i_metric == 0: axes[i_lev, i_metric].set_ylabel('C'+str(levels[i_lev]), fontsize=35, rotation=0, labelpad=50, color='white', y=0.4)

        # Colorbars
        # -----------
        if i_lev == 0 and (i_metric-1) == 0:
            axins = inset_axes(axes[i_lev, i_metric],
                   width="390%",  # width = 5% of parent_bbox width
                   height="15%",  # height : 50%
                   loc='upper center',
                   bbox_to_anchor=(1.7, 0.4, 1, 1),
                   bbox_transform=axes[i_lev, i_metric].transAxes,
                   borderpad=0)
            cbar = fig.colorbar(c, cax=axins, orientation='horizontal') #, ticks=clims[i_metric]) #, fraction=0.046, pad=0.04, location='top')
            cbar.ax.tick_params(labelsize=18, color='black', labelcolor='white', labeltop=True, labelbottom=False, direction='in', length=axins.get_window_extent().height)
            cbar.set_label(cbarLabels[int((i_metric-1)/4)], color='white', fontsize=35, labelpad=-110)

            # for scientific notation
            if clims[i_metric][-1] < 0.1:
                cbar.formatter.set_powerlimits((-3, -3))
                cbar.ax.xaxis.get_offset_text().set(size=13, color='white')
                cbar.ax.xaxis.major.formatter._useMathText = True
            else:  # trick to show the fIVIM values as percentage
                cbar.formatter.set_powerlimits((-2, -2))
                cbar.ax.xaxis.get_offset_text().set(size=0, color='black')
        # for D
        # transverse
        if i_lev == 0 and i_metric == 5:
            axinsTrans = inset_axes(axes[i_lev, i_metric],
                   width="150%",  # width = 5% of parent_bbox width
                   height="15%",  # height : 50%
                   loc='upper center',
                   bbox_to_anchor=(0.49, 0.4, 1, 1),
                   bbox_transform=axes[i_lev, i_metric].transAxes,
                   borderpad=0)
            cbar = fig.colorbar(c, cax=axinsTrans, orientation='horizontal') #, ticks=clims[i_metric]) #, fraction=0.046, pad=0.04, location='top')
            cbar.ax.tick_params(labelsize=18, color='black', labelcolor='white', labeltop=True, labelbottom=False, direction='in', length=axinsTrans.get_window_extent().height)
            # cbar.set_label(cbarLabels[1], color='white', fontsize=35, labelpad=-110)

            # for scientific notation
            cbar.formatter.set_powerlimits((-3, -3))
            cbar.ax.xaxis.get_offset_text().set(size=13, color='white')
            cbar.ax.xaxis.major.formatter._useMathText = True

        #
        if i_lev == 0 and i_metric == 7:
            axinsIS = inset_axes(axes[i_lev, i_metric],
                   width="80%",  # width = 5% of parent_bbox width
                   height="15%",  # height : 50%
                   loc='upper center',
                   bbox_to_anchor=(0.0, 0.4, 1, 1),
                   bbox_transform=axes[i_lev, i_metric].transAxes,
                   borderpad=0)
            cbar = fig.colorbar(c, cax=axinsIS, orientation='horizontal') #, ticks=clims[i_metric]) #, fraction=0.046, pad=0.04, location='top')
            cbar.ax.tick_params(labelsize=18, color='black', labelcolor='white', labeltop=True, labelbottom=False, direction='in', length=axinsIS.get_window_extent().height)
            cbar.set_label(cbarLabels[1], color='white', fontsize=35, labelpad=-110, x=0.0)

            # for scientific notation
            cbar.formatter.set_powerlimits((-3, -3))
            cbar.ax.xaxis.get_offset_text().set(size=13, color='white')
            cbar.ax.xaxis.major.formatter._useMathText = True

        if i_lev == 0 and i_metric == 8:
            axinsIS = inset_axes(axes[i_lev, i_metric],
                   width="80%",  # width = 5% of parent_bbox width
                   height="15%",  # height : 50%
                   loc='upper center',
                   bbox_to_anchor=(0.0, 0.4, 1, 1),
                   bbox_transform=axes[i_lev, i_metric].transAxes,
                   borderpad=0)
            cbar = fig.colorbar(c, cax=axinsIS, orientation='horizontal') #, ticks=clims[i_metric]) #, fraction=0.046, pad=0.04, location='top')
            cbar.ax.tick_params(labelsize=18, color='black', labelcolor='white', labeltop=True, labelbottom=False, direction='in', length=axinsIS.get_window_extent().height)

            # for scientific notation
            cbar.formatter.set_powerlimits((-3, -3))
            cbar.ax.xaxis.get_offset_text().set(size=13, color='white')
            cbar.ax.xaxis.major.formatter._useMathText = True



# plt.show()
fig.savefig(os.path.dirname(os.path.abspath(__file__)) + '/fig_meanMaps_hc_perfDirs_FivimXDstar_D.jpeg') #, transparent=True)
# fig.savefig(os.path.dirname(os.path.abspath(__file__)) + '/fig_meanMaps_hc_perfDirs.pdf') #, transparent=True)


