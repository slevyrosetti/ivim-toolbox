#!/bin/bash

# ==============================================
# PARAMETERS
# ==============================================
dataFolder="${HOME}/job/data/zurich/3T"
subjIDs="hc1 hc3 hc4 hc5 hc6 hc7 hc8 hc9 dcm1 dcm2"
# ==============================================
bashScriptFolderPath=$(pwd)

# go to data folder
cd ${dataFolder}

# loop over subjects to run the command
for id in $subjIDs; do
	echo "Processing subject: ${id}"
	cd ${id}/reg2template
	./process_${id}_reg2template.sh
	cd ../..
done

# come back to original folder
cd ${bashScriptFolderPath}
echo "***** All done! *****"





