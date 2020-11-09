#!/usr/bin/env python3.6

import argparse
import os
from datetime import datetime
import numpy as np
import sys
import warnings


def main(bvalsRequested, nbRep, unitVectDir, oFname, order):
    """Main."""

    # display command
    cmd = ' '.join([os.path.basename(sys.argv[0])]+sys.argv[1:])
    print("Running:\n"+cmd+"\n")

    # check if the order requested is compatible with the
    if order == "rep" and not np.all(nbRep == nbRep[0]):
        warnings.warn("The number of repetitions per b-value is different across b-values --> the order of the distribution is changed to \"bval\".")
        order = "bval"

    # if only one number of repetitions requested, apply the same to each b-values
    if len(nbRep) == 1:
        nbRep = np.tile(nbRep, (len(bvalsRequested)))

    # create file
    fileObj = open(oFname+'.txt', 'w')

    # write heading
    username = os.getlogin()
    fileObj.write('#-------------------------------------------------------------------------------\n'
                '# DWI Siemens orientation file for IVIM\n'
                '# Author: '+username+'\n'
                '# Date: '+datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")+'\n'
                '# Generated with command:\n'
                '# '+cmd+'\n'              '# \n'
                '# If run with b = '+str(max(bvalsRequested))+', obtained b-values will be (s/mm2):\n'
                '# '+str(bvalsRequested)+'\n'
                '# Number of b-values: '+str(len(bvalsRequested))+'\n'
                '# Number of repetitions: '+str(nbRep)+'\n'
                '# Order of distribution: '+order+'\n'                         
                '# Direction: '+str(unitVectDir)+'\n'
                '# Do not alternate positive and negative diffusion encoding directions\n'
                '#-------------------------------------------------------------------------------\n\n')

    # write characteristics of the diffusion file that will be read by the scanner
    fileObj.write('[directions='+str(np.sum(nbRep))+']\nCoordinateSystem = prs\nNormalisation = none\n')
    # NOT SUPPORTED ON VB17: fileID.write('Comment = To be run with b='+str(max(bvalsRequested))+'s/mm2\n')

    # compute vectors norms
    vectNorm = np.sqrt(bvalsRequested/max(bvalsRequested))

    # normalize unity vector to be sure (and get it as row vector)
    unitVectDir = unitVectDir/np.sqrt(np.sum(unitVectDir**2))

    # compute vectors to be written in the diffusion vector (requested norm from b-values * the direction)
    diff_vectors = np.tile(vectNorm, (3, 1)).T * np.tile(unitVectDir, (len(bvalsRequested), 1))

    # write vectors in file
    bvalsWritten = []  # matrix to record written b-values (for debugging)
    if order == "bval":
        for i_vect in range(len(bvalsRequested)):
            for i_rep in range(nbRep[i_vect]):

                sign = 1
                volume_nb = np.sum(nbRep[0:i_vect])+i_rep
                fileObj.write('Vector[{:d}] = ( {:1.8f}, {:1.8f}, {:1.8f} )\n'.format(volume_nb, sign*diff_vectors[i_vect, 0], sign*diff_vectors[i_vect, 1], sign*diff_vectors[i_vect, 2]))
                bvalsWritten.append(sign*bvalsRequested[i_vect])  # record written b-value for debugging

    elif order == "rep":
        for i_rep in range(nbRep[0]):
            for i_vect in range(len(bvalsRequested)):

                sign = 1
                volume_nb = i_rep*diff_vectors.shape[0] + i_vect
                fileObj.write('Vector[{:d}] = ( {:1.8f}, {:1.8f}, {:1.8f} )\n'.format(volume_nb, sign*diff_vectors[i_vect, 0], sign*diff_vectors[i_vect, 1], sign*diff_vectors[i_vect, 2]))
                bvalsWritten.append(sign*bvalsRequested[i_vect])  # record written b-value for debugging

    else:
        sys.exit("ERROR: the requested order (argument \"-order\" is not recognized.")


    print(bvalsWritten)

    # all done
    fileObj.close()
    print('\n>>> Diffusion vector file saved to: {}.txt'.format(oFname))


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program produces a diffusion vector file with the requested '
                                                 'diffusion-encoding directions and b-values and number of repetitions '
                                                 'per b-value to be used on Siemens MRI scanners in "free" diffusion mode.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-b', dest='bvalsRequested', help='List (within brackets and separate items by commas) of b-values desired.', type=str, required=True)
    requiredArgs.add_argument('-r', dest='nbRep', help="List (within squared brackets and separate items by commas) of the number of repetitions desired for each b-value (if only one value, the same number of repetitions will be applied to each b-value).", type=str, required=True)
    requiredArgs.add_argument('-v', dest='unitVectDir', help="List (within squared brackets and separate items by commas) of coordinates of a vector of norm 1 giving the desired diffusion gradient direction.", type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help="Output file name.", type=str, required=True)

    optionalArgs.add_argument('-order', dest='order', help='Enter \"bval\" for an order by b-value or \"rep\" for an order by repetition. Note that the order repetition-wise can only be performed if the same number of repetitions per b-value is requested.', type=str, required=False, default="bval")

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # print citation
    print('\n\n'
          '\n****************************** <3 Thank you for using our toolbox! <3 ***********************************'
          '\n********************************* PLEASE CITE THE FOLLOWING PAPER ***************************************'
          '\nLévy S, Rapacchi S, Massire A, et al. Intravoxel Incoherent Motion at 7 Tesla to quantify human spinal '
          '\ncord perfusion: limitations and promises. Magn Reson Med. 2020;00:1-20. https://doi.org/10.1002/mrm.28195'
          '\n*********************************************************************************************************'
          '\n\n')

    # run main
    main(bvalsRequested=np.array(args.bvalsRequested.strip("[]").split(','), dtype=float), nbRep=np.array(args.nbRep.strip("[]").split(','), dtype=int), unitVectDir=np.array(args.unitVectDir.strip("[]").split(','), dtype=float), oFname=args.oFname, order=args.order)




