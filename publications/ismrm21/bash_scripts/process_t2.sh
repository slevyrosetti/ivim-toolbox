#!/bin/bash

dcm_path=../dcm/

# -------------------------------- convert DCM to NII --------------------------------

# T2 sagittal
mkdir tmp_dcm2nii
dcm2niix -z y -m y -o tmp_dcm2nii ${dcm_path}*_t2_tse_sag*
mv tmp_dcm2nii/*.nii.gz t2.nii.gz
mv tmp_dcm2nii/*.json t2.json
rm -rf tmp_dcm2nii

# cord segmentation
sct_propseg -i t2.nii.gz -c t2

# label vertebrae
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2
rm warp_curve2straight.nii.gz warp_straight2curve.nii.gz straight_ref.nii.gz straightening.cache