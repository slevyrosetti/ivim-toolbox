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

dataFolder = str(Path.home())+"/job/data/zurich/3T/dcm2"
outputFname = './Fivim_radial_dcm2.jpeg'
SCmask_fname = "phase/seg.nii.gz"
mapFname = "phase/ivim_maps_mean_dirs/Fivim_radial_map.nii.gz"
colormap = "jet"
clims = (0, 0.2)  # (0.0002, 0.0004) (0.0009, 0.002)
# ======================================================================================================================

# load mask
SCmask_nii = nib.load(dataFolder+'/'+SCmask_fname)
SCmask = SCmask_nii.get_data()
# define crop size
resolution = SCmask_nii.header['pixdim'][1:3]
crop_xsize = 8 / resolution[0]
crop_ysize = 8 / resolution[1]

# load map
metricMap = nib.load(dataFolder+'/'+mapFname).get_data()

# add a fake dimension if the map is 2D
if len(SCmask.shape) == 2:
    SCmask = SCmask[..., np.newaxis]
    metricMap = metricMap[..., np.newaxis]

# Make figure
fig, axes = plt.subplots(SCmask.shape[2], 1, figsize=(1.5, 8.5))
plt.subplots_adjust(wspace=0.1, left=0, right=1, hspace=-0.77, bottom=-0.16, top=1.16)
fig.patch.set_facecolor('black')

if SCmask.shape[2] == 1:
    axes = [axes]

for i_z in range(SCmask.shape[2]):
    i_ax = SCmask.shape[2] - i_z - 1

    print('\n>>> Slice: z={}'.format(i_z))

    # mask the metric map
    metricMap_z_i_masked = np.ma.masked_where(SCmask[:, :, i_z] == 0.0, metricMap[:, :, i_z])

    # get center of mass of the slice
    properties = regionprops((SCmask[:, :, i_z] > 0).astype(int), metricMap_z_i_masked)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    crop_xmin = int(round(center_of_mass[0]-crop_xsize, 0))
    crop_xmax = int(round(center_of_mass[0]+crop_xsize, 0))
    crop_ymin = int(round(center_of_mass[1]-crop_ysize, 0))
    crop_ymax = int(round(center_of_mass[1]+crop_ysize, 0))

    # plot image
    # -----------
    c = axes[i_ax].imshow(np.rot90(metricMap_z_i_masked[crop_xmin:crop_xmax+1, crop_ymin:crop_ymax+1]), cmap=colormap, clim=clims)
    axes[i_ax].set_frame_on(False)
    axes[i_ax].xaxis.set_visible(False)
    axes[i_ax].yaxis.set_ticks([])

# plt.show()
fig.savefig(outputFname, transparent=False)
