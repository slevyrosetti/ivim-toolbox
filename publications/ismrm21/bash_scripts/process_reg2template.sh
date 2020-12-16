#!/bin/bash

# DIFF DIRS --> PHASE
# average b-values higher than 500
sct_dmri_separate_b0_and_dwi -i ../read/topup_mean.nii.gz -bvec ../read/fwd_mean.bvec -a 1 -bval ../read/fwd_mean.bval -bvalmin 201 -ofolder read_getHighBvals
mv read_getHighBvals/topup_mean_dwi_mean.nii.gz read_highb_mean.nii.gz
rm -rf read_getHighBvals

sct_dmri_separate_b0_and_dwi -i ../phase/topup_mean.nii.gz -bvec ../phase/fwd_mean.bvec -a 1 -bval ../phase/fwd_mean.bval -bvalmin 201 -ofolder phase_getHighBvals
mv phase_getHighBvals/topup_mean_dwi_mean.nii.gz phase_highb_mean.nii.gz
rm -rf phase_getHighBvals

sct_dmri_separate_b0_and_dwi -i ../slice/topup_mean.nii.gz -bvec ../slice/fwd_mean.bvec -a 1 -bval ../slice/fwd_mean.bval -bvalmin 201 -ofolder slice_getHighBvals
mv slice_getHighBvals/topup_mean_dwi_mean.nii.gz slice_highb_mean.nii.gz
rm -rf slice_getHighBvals

# # segment cord
# sct_propseg -i phase_highb_mean.nii.gz -c t1
# sct_propseg -i read_highb_mean.nii.gz -c t1
# sct_propseg -i slice_highb_mean.nii.gz -c t1

# estimate warping fields
sct_register_multimodal -i read_highb_mean.nii.gz -d phase_highb_mean.nii.gz -iseg ../read/seg.nii.gz -dseg ../phase/seg.nii.gz -o read2phase.nii.gz -owarp warp_read2phase.nii.gz -param step=1,type=seg,algo=rigid,slicewise=1,metric=MeanSquares -x spline
mv warp_phase_highb_mean2read_highb_mean.nii.gz warp_phase2read.nii.gz
mv read2phase_inv.nii.gz phase2read.nii.gz
rm read_highb_mean_warp2d_*GenericAffine.mat

sct_register_multimodal -i slice_highb_mean.nii.gz -d phase_highb_mean.nii.gz -iseg ../slice/seg.nii.gz -dseg ../phase/seg.nii.gz -o slice2phase.nii.gz -owarp warp_slice2phase.nii.gz -param step=1,type=seg,algo=rigid,slicewise=1,metric=MeanSquares -x spline
mv warp_phase_highb_mean2slice_highb_mean.nii.gz warp_phase2slice.nii.gz
mv slice2phase_inv.nii.gz phase2slice.nii.gz
rm slice_highb_mean_warp2d_*GenericAffine.mat

# fslview phase_highb_mean.nii.gz read2phase.nii.gz slice2phase.nii.gz &

# PHASE --> MGE
sct_register_multimodal -i phase_highb_mean.nii.gz -d ../mge/mge.nii.gz -iseg ../phase/seg.nii.gz -dseg ../mge/mge_seg.nii.gz -o phase2mge.nii.gz -owarp warp_phase2mge.nii.gz -param step=1,type=seg,algo=rigid,slicewise=1,metric=MeanSquares -x spline
mv warp_mge2phase_highb_mean.nii.gz warp_mge2phase.nii.gz
mv phase2mge_inv.nii.gz mge2phase.nii.gz
rm phase_highb_mean_warp2d_*GenericAffine.mat

# # reslice MGE to T2 sag to get vertebral labelling
# fsleyes ../t2/t2.nii.gz ../mge/mge.nii.gz &
# fslview ../mge/mge.nii.gz &
# read -p "Press enter when file mge_labels.nii.gz has been created..."
sct_register_multimodal -i ../t2/t2_seg_labeled.nii.gz -d ../mge/mge.nii.gz -o mge_labeled.nii.gz -owarp warp_reslice_t22mge.nii.gz -identity 1 -x nn
rm warp_reslice_t22mge.nii.gz warp_mge2t2_seg_labeled.nii.gz mge_labeled_inv.nii.gz
sct_label_utils -i mge_labeled.nii.gz -vert-body 1,3 -o mge_labels.nii.gz

# TEMPLATE --> MGE
# # get good MGE seg
# # sct_propseg -i phase2mge.nii.gz -c t1
# # fslview ../mge/mge_3te_sqr_mean.nii.gz phase2mge.nii.gz phase2mge_seg.nii.gz &
# read -p "Press enter when file mge_seg_ok.nii.gz has been created..."

# run registration to template
# sct_register_to_template -i ../mge/mge_4te_sqr_mean.nii.gz -s ../mge/mge_seg_ok.nii.gz -l mge_labels.nii.gz -c t2s -ref subject -param step=1,type=seg,algo=centermassrot,smooth=0:step=2,type=seg,algo=columnwise,smooth=0,smoothWarpXY=2
sct_register_to_template -i ../mge/mge.nii.gz -s ../mge/mge_seg.nii.gz -l mge_labels.nii.gz -c t2s -ref subject -param step=1,type=seg,algo=centermassrot,smooth=0
mv anat2template.nii.gz mge2template_rough.nii.gz
mv template2anat.nii.gz template2mge_rough.nii.gz
mv warp_template2anat.nii.gz warp_template2mge_rough.nii.gz
mv warp_anat2template.nii.gz warp_mge2template_rough.nii.gz

# WM TEMPLATE --> MGE 
# segment gray and white matter on MGE
sct_deepseg_gm -i ../mge/mge.nii.gz -o mge_gm.nii.gz
fslmaths ../mge/mge_seg.nii.gz -sub mge_gm.nii.gz mge_wm.nii.gz
# roughly warp template WM map to MGE
sct_warp_template -d ../mge/mge.nii.gz -w warp_template2mge_rough.nii.gz -ofolder label_rough -a 0
# sct_register_multimodal -i reg2template_exampleparam/label/template/PAM50_t2s.nii.gz -d ../mge/mge_3te_sqr_mean.nii.gz -iseg reg2template_exampleparam/label/template/PAM50_wm.nii.gz -dseg ../mge/mge_3te_sqr_mean_wm.nii.gz -param step=1,type=seg,algo=syn,slicewise=1,metric=MeanSquares:step=2,type=im,algo=syn,slicewise=1,metric=MeanSquares -x spline -o template2anat_gm.nii.gz
sct_register_multimodal -i label_rough/template/PAM50_wm.nii.gz -d mge_wm.nii.gz -param step=1,type=im,algo=syn,slicewise=1,metric=MeanSquares -x spline -o template2mge_gm.nii.gz -owarp warp_template2mge_gm.nii.gz
mv template2mge_gm_inv.nii.gz mge2template_gm.nii.gz
mv warp_mge_wm2PAM50_wm.nii.gz warp_mge2template_gm.nii.gz

# CONCATENATE WARPING FIELDS
# TEMPLATE->MGE + WM TEMPLATE2MGE->WM MGE + MGE->PHASE
sct_concat_transfo -d phase_highb_mean.nii.gz -w warp_template2mge_rough.nii.gz warp_template2mge_gm.nii.gz warp_mge2phase.nii.gz -o warp_template2phase.nii.gz
# PHASE->MGE + WM MGE-> WM TEMPLATE2MGE + MGE->TEMPLATE
sct_concat_transfo -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_phase2mge.nii.gz warp_mge2template_gm.nii.gz warp_mge2template_rough.nii.gz -o warp_phase2template.nii.gz

# template to read
sct_concat_transfo -d read_highb_mean.nii.gz -w warp_template2mge_rough.nii.gz warp_template2mge_gm.nii.gz warp_mge2phase.nii.gz warp_phase2read.nii.gz -o warp_template2read.nii.gz
# and read to template
sct_concat_transfo -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_read2phase.nii.gz warp_phase2mge.nii.gz warp_mge2template_gm.nii.gz warp_mge2template_rough.nii.gz -o warp_read2template.nii.gz

# template to slice
sct_concat_transfo -d slice_highb_mean.nii.gz -w warp_template2mge_rough.nii.gz warp_template2mge_gm.nii.gz warp_mge2phase.nii.gz warp_phase2slice.nii.gz -o warp_template2slice.nii.gz
# and slice to template
sct_concat_transfo -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_slice2phase.nii.gz warp_phase2mge.nii.gz warp_mge2template_gm.nii.gz warp_mge2template_rough.nii.gz -o warp_slice2template.nii.gz

# template to MGE
sct_concat_transfo -d ../mge/mge.nii.gz -w warp_template2mge_rough.nii.gz warp_template2mge_gm.nii.gz -o warp_template2mge.nii.gz
# and MGE to template
sct_concat_transfo -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_mge2template_gm.nii.gz warp_mge2template_rough.nii.gz -o warp_mge2template.nii.gz

# WARP TEMPLATE AND ATLAS TO EACH DIFFUSION DIRECTION (AND TO MGE)
sct_warp_template -d phase_highb_mean.nii.gz -w warp_template2phase.nii.gz -a 1 -ofolder ../phase/label
sct_warp_template -d read_highb_mean.nii.gz -w warp_template2read.nii.gz -a 1 -ofolder ../read/label
sct_warp_template -d slice_highb_mean.nii.gz -w warp_template2slice.nii.gz -a 1 -ofolder ../slice/label
sct_warp_template -d ../mge/mge.nii.gz -w warp_template2mge.nii.gz -a 1 -ofolder ../mge/label

# WARP IVIM MAPS FROM EACH DIRECTION TO TEMPLATE
IVIM_PARAMS="Fivim_map Dstar_map D_map FivimXDstar_map"
mkdir ../template
# phase
mkdir ../template/phase
for param in ${IVIM_PARAMS}; do
	sct_apply_transfo -i ../phase/ivim_maps/${param}.nii.gz -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_phase2template.nii.gz -o ../template/phase/${param}.nii.gz
done
# read
mkdir ../template/read
for param in ${IVIM_PARAMS}; do
	sct_apply_transfo -i ../read/ivim_maps/${param}.nii.gz -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_read2template.nii.gz -o ../template/read/${param}.nii.gz
done
# slice
mkdir ../template/slice
for param in ${IVIM_PARAMS}; do
	sct_apply_transfo -i ../slice/ivim_maps/${param}.nii.gz -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_slice2template.nii.gz -o ../template/slice/${param}.nii.gz
done
# mge
sct_apply_transfo -i ../mge/mge.nii.gz -d ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz -w warp_mge2template.nii.gz -o ../template/mge2template.nii.gz

# ============================================================================================================
# PROCESS MAPS IN TEMPLATE SPACE (averaging across directions, register across slices, average across slices)
# ============================================================================================================
# 
# COMPUTE MEAN ACROSS DIRECTIONS AND CROP ALONG Z (and across slices without registration)
# ------------------------------------------------------------------------------------------------------------
zmin=873
zsize=$((972-$zmin))

# MGE
fslroi ../template/mge2template.nii.gz ../template/mge2template_zcrop_${zmin}_${zsize}.nii.gz 0 -1 0 -1 $zmin $zsize
# fslmaths ../template/mge2template_zcrop_${zmin}_${zsize}.nii.gz -Zmean ../template/mge2template_mean_z.nii.gz
# IVIM params
for param in ${IVIM_PARAMS}; do
	fslmerge -t ../template/${param}_3dirs.nii.gz ../template/phase/${param}.nii.gz ../template/read/${param}.nii.gz ../template/slice/${param}.nii.gz
	# mean across diff dirs in template space
	fslmaths ../template/${param}_3dirs.nii.gz -Tmean ../template/${param}_mean_dirs.nii.gz 
	fslroi ../template/${param}_mean_dirs.nii.gz ../template/${param}_mean_dirs_zcrop_${zmin}_${zsize}.nii.gz 0 -1 0 -1 $zmin $zsize
	# fslmaths ../template/${param}_mean_dirs_zcrop_${zmin}_${zsize}.nii.gz -Zmean ../template/${param}_mean_dirs_mean_z.nii.gz
done

# CROP EACH DIRECTION ALONG Z (and average each direction across slices independently without registration)
# ------------------------------------------------------------------------------------------------------------
DIRECTIONS="phase read slice"
PARAMS="Fivim Dstar FivimXDstar D"
for param in ${PARAMS}; do
	for dir in ${DIRECTIONS}; do
		fslroi ../template/${dir}/${param}_map.nii.gz ../template/${dir}/${param}_map_zcrop_${zmin}_${zsize}.nii.gz 0 -1 0 -1 $zmin $zsize
		# fslmaths ../template/${dir}/${param}_map_zcrop_${zmin}_${zsize}.nii.gz -Zmean ../template/${dir}/${param}_map_mean_z.nii.gz
	done
done

fslview ${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz ../template/mge2template.nii.gz ../template/FivimXDstar_map_mean_dirs.nii.gz -b 0,0.003 -l $jet ../template/Fivim_map_mean_dirs.nii.gz -b 0,0.3 -l $jet ../template/Dstar_map_mean_dirs.nii.gz -b 0,0.020 -l $jet ../template/D_map_mean_dirs.nii.gz -b 0,0.0015 -l $jet &


# REGISTER MAPS IN TEMPLATE SPACE ACROSS SLICES AND THEN AVERAGE ACROSS SLICES
# ------------------------------------------------------------------------------------------------------------
zRef_template=909
# get cord segmentation cropped
fslroi ${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz ../template/PAM50_cord_zcrop_${zmin}_${zsize}.nii.gz 0 -1 0 -1 $zmin $zsize
# remove background from MGE
fslmaths ../template/mge2template_zcrop_${zmin}_${zsize}.nii.gz -mul ../template/PAM50_cord_zcrop_${zmin}_${zsize}.nii.gz ../template/mge2template_zcrop_${zmin}_${zsize}_edit.nii.gz

DIRECTIONS="phase read slice"
for dir in ${DIRECTIONS}; do
	mv ../template/${dir}/D_map_zcrop_${zmin}_${zsize}.nii.gz ../template/D${dir}_map_zcrop_${zmin}_${zsize}.nii.gz
done

# register IVIM maps across slices based on MGE
reg_across_slices.py -ref ../template/mge2template_zcrop_${zmin}_${zsize}_edit.nii.gz -sliceref $[$zRef_template-$zmin+1] -maps ../template/Fivim_map_mean_dirs_zcrop_${zmin}_${zsize}.nii.gz,../template/Dstar_map_mean_dirs_zcrop_${zmin}_${zsize}.nii.gz,../template/FivimXDstar_map_mean_dirs_zcrop_${zmin}_${zsize}.nii.gz,../template/Dphase_map_zcrop_${zmin}_${zsize}.nii.gz,../template/Dread_map_zcrop_${zmin}_${zsize}.nii.gz,../template/Dslice_map_zcrop_${zmin}_${zsize}.nii.gz,../template/D_map_mean_dirs_zcrop_${zmin}_${zsize}.nii.gz,../template/mge2template_zcrop_${zmin}_${zsize}.nii.gz -owarp ../template/warps_mean_z -omap ../template/maps_mean_z

# rename output maps for easier handle
IVIM_PARAMS="Fivim Dstar FivimXDstar"
for param in ${IVIM_PARAMS}; do
	mv ../template/maps_mean_z/${param}_map_mean_dirs_zcrop_${zmin}_${zsize}_reg_z_mean_z.nii.gz ../template/maps_mean_z/${param}_map_mean_dirs_reg_z_mean_z.nii.gz
done
DIRECTIONS="phase read slice"
for dir in ${DIRECTIONS}; do
	mv ../template/maps_mean_z/D${dir}_map_zcrop_${zmin}_${zsize}_reg_z_mean_z.nii.gz ../template/maps_mean_z/D${dir}_map_reg_z_mean_z.nii.gz
done
mv ../template/maps_mean_z/mge2template_zcrop_${zmin}_${zsize}_reg_z_mean_z.nii.gz ../template/maps_mean_z/mge2template_reg_z_mean_z.nii.gz

fslview -m single ../template/maps_mean_z/mge2template_reg_z_mean_z.nii.gz ../template/maps_mean_z/FivimXDstar_map_mean_dirs_reg_z_mean_z.nii.gz  -b 0,0.003 -l $jet ../template/maps_mean_z/Fivim_map_mean_dirs_reg_z_mean_z.nii.gz  -b 0,0.3 -l $jet ../template/maps_mean_z/Dstar_map_mean_dirs_reg_z_mean_z.nii.gz  -b 0,0.020 -l $jet ../template/maps_mean_z/Dphase_map_reg_z_mean_z.nii.gz  -b 0,0.0009 -l $jet ../template/maps_mean_z/Dread_map_reg_z_mean_z.nii.gz  -b 0,0.0009 -l $jet ../template/maps_mean_z/Dslice_map_reg_z_mean_z.nii.gz  -b 0,0.0020 -l $jet &

# ============================================================================================================
# PROCESS MAPS IN NATIVE SPACE (averaging across directions, register across slices, average across slices)
# ============================================================================================================

# Register IVIM maps from Read and Slice diffusion-encoding directions to Phase (and average them)
# ------------------------------------------------------------------------------------------------------------
mkdir ../phase/ivim_maps_read2phase
mkdir ../phase/ivim_maps_slice2phase
mkdir ../phase/ivim_maps_mean_dirs
IVIM_PARAMS="Fivim Dstar FivimXDstar D"
for param in ${IVIM_PARAMS}; do
	# warp to phase space
	sct_apply_transfo -i ../read/ivim_maps/${param}_map.nii.gz -d phase_highb_mean.nii.gz -w warp_read2phase.nii.gz -o ../phase/ivim_maps_read2phase/${param}_map.nii.gz
	sct_apply_transfo -i ../slice/ivim_maps/${param}_map.nii.gz -d phase_highb_mean.nii.gz -w warp_slice2phase.nii.gz -o ../phase/ivim_maps_slice2phase/${param}_map.nii.gz
	# average across diffusion-encoding directions
	fslmaths ../phase/ivim_maps/${param}_map.nii.gz -add ../phase/ivim_maps_read2phase/${param}_map.nii.gz -add ../phase/ivim_maps_slice2phase/${param}_map.nii.gz -div 3 ../phase/ivim_maps_mean_dirs/${param}_map.nii.gz
	# # average across slices
	# fslmaths ../phase/ivim_maps_mean_dirs/${param}.nii.gz -Zmean ../phase/ivim_maps_mean_dirs/${param}_mean_z.nii.gz
done

fslview -m lightbox mge2phase.nii.gz ../phase/ivim_maps_mean_dirs/FivimXDstar_map.nii.gz  -b 0,0.003 -l $jet ../phase/ivim_maps_mean_dirs/Fivim_map.nii.gz  -b 0,0.3 -l $jet ../phase/ivim_maps_mean_dirs/Dstar_map.nii.gz  -b 0,0.020 -l $jet ../phase/ivim_maps_mean_dirs/D_map.nii.gz -b 0,0.0015 -l $jet &

# REGISTER ACROSS SLICES BASED ON THE ANATOMIC MGE AND AVERAGE ACROSS SLICES
# ------------------------------------------------------------------------------------------------------------
zRef=4
reg_across_slices.py -ref mge2phase.nii.gz -sliceref $zRef -mask ../phase/seg_dilated.nii.gz -maps ../phase/ivim_maps_mean_dirs/Fivim_map.nii.gz,../phase/ivim_maps_mean_dirs/Dstar_map.nii.gz,../phase/ivim_maps_mean_dirs/FivimXDstar_map.nii.gz,../phase/ivim_maps_mean_dirs/D_map.nii.gz -owarp ../phase/ivim_maps_mean_dirs/warps_mean_z -omap ../phase/ivim_maps_mean_dirs/maps_mean_z

fslview -m single ../phase/ivim_maps_mean_dirs/maps_mean_z/ref_reg_z_mean_z.nii.gz  -b 0,900 ../phase/ivim_maps_mean_dirs/maps_mean_z/FivimXDstar_map_reg_z_mean_z.nii.gz  -b 0,0.003 -l $jet ../phase/ivim_maps_mean_dirs/maps_mean_z/Fivim_map_reg_z_mean_z.nii.gz  -b 0,0.3 -l $jet ../phase/ivim_maps_mean_dirs/maps_mean_z/Dstar_map_reg_z_mean_z.nii.gz  -b 0,0.020 -l $jet ../phase/ivim_maps_mean_dirs/maps_mean_z/D_map_reg_z_mean_z.nii.gz  -b 0,0.0015 -l $jet &

