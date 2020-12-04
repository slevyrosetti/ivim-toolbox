#!/bin/bash

dcm_path=../dcm

# -------------------------------- start processing --------------------------------

# convert to NII
# -------------------------------------------------------
# convert data with FWD phase encoding to NII
mkdir tmp_dcm2nii
dcm2niix -z y -m y -o tmp_dcm2nii ${dcm_path}/*_DWI_ZOOMit_IVIM_3dir_1b_phase_P_A
mv tmp_dcm2nii/*.bval fwd.bval
mv tmp_dcm2nii/*.bvec fwd.bvec
mv tmp_dcm2nii/*.nii.gz fwd.nii.gz
mv tmp_dcm2nii/*.json fwd.json
rm -rf tmp_dcm2nii
# convert data with RVS phase encoding to NII
mkdir tmp_dcm2nii
dcm2niix -z y -m y -o tmp_dcm2nii ${dcm_path}/*_DWI_ZOOMit_IVIM_3dir_1b_phase_A_P
mv tmp_dcm2nii/*.bval rvs.bval
mv tmp_dcm2nii/*.bvec rvs.bvec
mv tmp_dcm2nii/*.nii.gz rvs.nii.gz
mv tmp_dcm2nii/*.json rvs.json
rm -rf tmp_dcm2nii


# denoise
# -------------------------------------------------------
ivim_denoise_lpca.py -dwi fwd.nii.gz -bval fwd.bval -bvec fwd.bvec -o fwd_denoised
ivim_denoise_lpca.py -dwi rvs.nii.gz -bval rvs.bval -bvec rvs.bvec -o rvs_denoised

# ungibbs
# -------------------------------------------------------
ivim_ungibbs.py -i fwd_denoised.nii.gz -o fwd_ungibbs
ivim_ungibbs.py -i rvs_denoised.nii.gz -o rvs_ungibbs


# motion correction
# -------------------------------------------------------
moco_based_1st_volume.sh -i fwd_ungibbs.nii.gz -outfname fwd_ungibbs_moco -metric MI -ref mean #-mask seg_dilated.nii.gz (better without mask so far)
rm -rf fwd_moco_1stvol
mv moco_1stvol fwd_moco_1stvol

moco_based_1st_volume.sh -i rvs_ungibbs.nii.gz -outfname rvs_ungibbs_moco -metric MI -ref mean #-mask seg_dilated.nii.gz (better without mask so far)
rm -rf rvs_moco_1stvol
mv moco_1stvol rvs_moco_1stvol


# distortion correction
# -------------------------------------------------------
ivim_create_topup_params_file.py -fwd fwd.json -rvs rvs.json -o topup_acqparams
tls_topup.sh -fwd fwd_moco_1stvol/fwd_ungibbs_moco -rvs rvs_moco_1stvol/rvs_ungibbs_moco -bvec_fwd fwd.bvec -bvec_rvs rvs.bvec -bval_fwd fwd.bval -bval_rvs rvs.bval -outfname topup -acqparams topup_acqparams -acqparams_apply topup_acqparams -cfg b02b0_1.cnf

# mean across repetitions for each b-value
# -------------------------------------------------------
ivim_average_bvalue_wise.py -i topup.nii.gz -bval fwd.bval -bvec fwd.bvec -suffix mean -operation mean

# cord segmentation
# -------------------------------------------------------
sct_dmri_separate_b0_and_dwi -i topup_mean.nii.gz -bvec fwd_mean.bvec -a 1 -bval fwd_mean.bval -bvalmin 401
sct_propseg -i topup_mean_dwi_mean.nii.gz -c t1
mv topup_mean_dwi_mean_seg.nii.gz seg.nii.gz
rm topup_mean_dwi_mean_rescaled_CSF_seg.nii.gz topup_mean_dwi_mean_rescaled_CSF_mesh.vtk topup_mean_dwi_mean_rescaled_cross_sectional_areas_CSF.txt segmentation_CSF_mesh_low_resolution.vtk topup_mean_dwi_mean_rescaled_mesh.vtk topup_mean_dwi_mean_rescaled_cross_sectional_areas.txt topup_mean_dwi_mean_rescaled_centerline.txt InitialTubeCSF2.vtk InitialTubeCSF1.vtk segmentation_mesh_low_resolution.vtk InitialTube2.vtk InitialTube1.vtk

# ------------------------ IVIM PARAMETERS MAPS CALCULATION -------------
fslmaths seg.nii.gz -kernel sphere 5 -dilM seg_dilated.nii.gz
ivim_fitting.py -i ivim_fitting.py -i topup_mean.nii.gz -b fwd_mean.bval -ma seg_dilated.nii.gz -mo one-step -o ivim_maps -mt 1

# ivim_fits_viewer.py -i ivim_maps/Fivim_map.nii.gz -mode multiaxial -plotdir ivim_maps/19*_plots/ -param cmap=jet:clim=0\;0.3
