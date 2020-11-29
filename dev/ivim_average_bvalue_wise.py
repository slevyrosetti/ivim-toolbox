#!/usr/bin/env python3.6
"""
Average data per b-value.


Created on Tue Jul  4 17:45:43 2017

@author: slevy
"""

import argparse
import numpy as np
import nibabel as nib
import os
from dipy.io.image import load_nifti


def main(iFname, bvalFname, bvecFname, suffix, operation):
    """Main."""

    # load data
    data, affine = load_nifti(iFname)
    bvals = np.loadtxt(bvalFname, delimiter=None)
    bvals_uniques, idx_unique_bvals = np.unique(bvals, return_index=True)
    bvecs = np.loadtxt(bvecFname, delimiter=None)
    # bvecs_uniques_sort, indexes_uniques = np.unique(bvecs, axis=1, return_index=True)
    bvecs_uniques = bvecs[:, idx_unique_bvals]
    
    # average images per b-value
    data_average_bvalwise = np.zeros([data.shape[0], data.shape[1], data.shape[2], len(bvals_uniques)])
#    bvecs_average_bvalwise = np.zeros([3, len(bvals_uniques)])
    for i_b in range(len(bvals_uniques)):
        if operation == 'mean':
            data_average_bvalwise[:,:,:,i_b]=np.mean(data[:,:,:,bvals == bvals_uniques[i_b]], axis=3)
#            bvecs_average_bvalwise[:, i_b] = np.mean(bvecs[:, bvals == bvals_uniques[i_b]], axis=1)            
        elif operation == 'median':
            data_average_bvalwise[:,:,:,i_b]=np.median(data[:,:,:,bvals == bvals_uniques[i_b]], axis=3)
#            bvecs_average_bvalwise[:, i_b] = np.median(bvecs[:, bvals == bvals_uniques[i_b]], axis=1)
        elif operation == 'max':
            data_average_bvalwise[:,:,:,i_b]=np.amax(data[:,:,:,bvals == bvals_uniques[i_b]], axis=3)
    
    # save outputs
    base = os.path.basename(iFname)
    input_filename, input_ext = base.split('.nii')
    input_ext = '.nii'+input_ext
    input_path = os.path.dirname(iFname)
    output_fname = input_filename + '_' + suffix + input_ext
    nib.save(nib.Nifti1Image(data_average_bvalwise, affine), output_fname)

    bval_filename, bval_ext = os.path.splitext(bvalFname)
    bval_out_fname = bval_filename + '_' + suffix + bval_ext
    np.savetxt(bval_out_fname, bvals_uniques, fmt='%i', delimiter='\n', newline=' ')
    
    bvec_filename, bvec_ext = os.path.splitext(bvecFname)
    bvec_out_fname = bvec_filename + '_' + suffix + bvec_ext
    np.savetxt(bvec_out_fname, bvecs_uniques, fmt='%1.13e', delimiter=' ', newline='\n')
    
    print('\nDone! Files created: \n\t\t' + output_fname + '\n\t\t' + bval_out_fname + '\n\t\t' + bvec_out_fname + '\n')



# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program averages a 4D volume per b-value according to bval file.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help='Path to 4D nifti image.', type=str, required=True)
    requiredArgs.add_argument('-bval', dest='bvalFname', help='Path to corresponding bval file.', type=str, required=True)
    requiredArgs.add_argument('-bvec', dest='bvecFname', help='Path to corresponding bvec file.', type=str, required=True)

    optionalArgs.add_argument('-suffix', dest='suffix', help='Suffix for the output averaged 4D volume, to be added to the input image file name.', type=str, required=False, default="bval")
    optionalArgs.add_argument('-operation', dest='operation', help='mean or median.', type=str, required=False, default="mean")

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
    main(iFname=args.iFname, bvalFname=args.bvalFname, bvecFname=args.bvecFname, suffix=args.suffix, operation=args.operation)
