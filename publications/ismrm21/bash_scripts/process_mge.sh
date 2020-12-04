#!/bin/bash

dcm_path=../dcm/

# -------------------------------- convert DCM to NII --------------------------------

# convert to NII
mkdir tmp_dcm2nii
dcm2niix -z y -m y -o tmp_dcm2nii ${dcm_path}*_GRE-ME*
mv tmp_dcm2nii/*.nii.gz mge.nii.gz
mv tmp_dcm2nii/*.json mge.json
rm -rf tmp_dcm2nii

# cord segmentation
sct_propseg -i mge.nii.gz -c t2s -d 1 -radius 5

# # manual segmentation of cord
# sct_label_utils -i mge_3te_sqr_mean.nii.gz -o mge_seg_ok.nii.gz -create ${cord_seg}