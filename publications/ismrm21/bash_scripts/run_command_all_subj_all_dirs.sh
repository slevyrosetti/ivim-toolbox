#!/bin/bash

# ==============================================
# PARAMETERS
# ==============================================
dataFolder="${HOME}/job/data/zurich/3T"
subjIDs="hc1 hc3 hc4 hc5 hc6 hc7 hc8 dcm1 dcm2"
diffDirs="phase read slice"
command="ivim_fitting.py -i ivim_fitting.py -i topup_mean.nii.gz -b fwd_mean.bval -ma seg_dilated.nii.gz -mo one-step -o ivim_maps -mt 1"
# ==============================================
bashScriptFolderPath=$(pwd)

# go to data folder
cd ${dataFolder}

# loop over subjects and diffusion-encoding directions to run the command
for id in $subjIDs; do
	echo "Processing subject: ${id}"
	cd $id
	for dir in $diffDirs; do
		echo ">>> ${dir}"
		cd $dir
		eval $command
		cd ..
	done
	cd ..
done

# come back to original folder
cd ${bashScriptFolderPath}
echo "***** All done! *****"





