#!/bin/bash

bashScriptFolderPath=$(pwd)

subjectFolderPath=`expr $1`

if [ ! -d ${subjectFolderPath} ]; then
    echo "Directory ${subjectFolderPath} DOES NOT exists." 
    exit 9999 # die with error code 9999
fi

# go to subject's folder and get subject's name
cd ${subjectFolderPath}
subjID="$(basename $PWD)"

# process t2
mkdir t2
cp ${bashScriptFolderPath}/process_t2.sh ${subjectFolderPath}/process_${subjID}_t2.sh 
cd t2
./process_${subjID}_t2.sh
cd ..

# process mge
mkdir mge
cp ${bashScriptFolderPath}/process_mge.sh ${subjectFolderPath}/process_${subjID}_mge.sh 
cd mge
./process_${subjID}_mge.sh
cd ..

# process IVIM with diffusion-encoding = phase
mkdir phase
cp ${bashScriptFolderPath}/process_phase.sh ${subjectFolderPath}/process_${subjID}_phase.sh 
cd phase
./process_${subjID}_phase.sh
cd ..

# process IVIM with diffusion-encoding = read
mkdir read
cp ${bashScriptFolderPath}/process_read.sh ${subjectFolderPath}/process_${subjID}_read.sh 
cd read
./process_${subjID}_read.sh
cd ..

# process IVIM with diffusion-encoding = slice
mkdir slice
cp ${bashScriptFolderPath}/process_slice.sh ${subjectFolderPath}/process_${subjID}_slice.sh 
cd slice
./process_${subjID}_slice.sh
cd ..

# register to template
mkdir reg2template
cp ${bashScriptFolderPath}/process_reg2template.sh ${subjectFolderPath}/process_${subjID}_reg2template.sh 
cd reg2template
./process_${subjID}_reg2template.sh
cd ..

cd ${bashScriptFolderPath}
echo "***** All done! *****"





