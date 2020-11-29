#!/usr/bin/env python3.6

"""
=======================================================
Denoise images using Local PCA via empirical thresholds
=======================================================

PCA-based denoising algorithms are effective denoising methods because they
explore the redundancy of the multi-dimensional information of
diffusion-weighted datasets. In this example, we will show how to
perform the PCA-based denoising using the algorithm proposed by Manjon et al.
[Manjon2013]_.

This algorithm involves the following steps:

* First, we estimate the local noise variance at each voxel.

* Then, we apply PCA in local patches around each voxel over the gradient
  directions.

* Finally, we threshold the eigenvalues based on the local estimate of sigma
  and then do a PCA reconstruction


To perform PCA denoising without a prior noise standard deviation estimate
please see the following example instead: :ref:`denoise_mppca`

Let's load the necessary modules
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

import argparse


def main(dwiFname, bvalFname, bvecFname, oFname, plot):
    """Main."""

    """
    
    Load one of the datasets. These data were acquired with 63 gradients and 1
    non-diffusion (b=0) image.
    
    """

    data, affine = load_nifti(dwiFname)
    bvals, bvecs = read_bvals_bvecs(bvalFname, bvecFname)
    gtab = gradient_table(bvals, bvecs)

    print("Input Volume", data.shape)

    """
    Estimate the noise standard deviation
    =====================================
    
    We use the ``pca_noise_estimate`` method to estimate the value of sigma to be
    used in local PCA algorithm proposed by Manjon et al. [Manjon2013]_.
    It takes both data and the gradient table object as input and returns an
    estimate of local noise standard deviation as a 3D array. We return a smoothed
    version, where a Gaussian filter with radius 3 voxels has been applied to the
    estimate of the noise before returning it.
    
    We correct for the bias due to Rician noise, based on an equation developed by
    Koay and Basser [Koay2006]_.
    
    """

    t = time()
    sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)
    print("Sigma estimation time", time() - t)

    """
    Perform the localPCA using the function `localpca`
    ==================================================
    
    The localpca algorithm takes into account the multi-dimensional information of
    the diffusion MR data. It performs PCA on local 4D patch and
    then removes the noise components by thresholding the lowest eigenvalues.
    The eigenvalue threshold will be computed from the local variance estimate
    performed by the ``pca_noise_estimate`` function, if this is inputted in the
    main ``localpca`` function. The relationship between the noise variance
    estimate and the eigenvalue threshold can be adjusted using the input parameter
    ``tau_factor``. According to Manjon et al. [Manjon2013]_, this parameter is set
    to 2.3.
    """

    t = time()

    denoised_arr = localpca(data, sigma, tau_factor=2.3, patch_radius=3)

    print("Time taken for local PCA (slow)", -t + time())

    """
    The ``localpca`` function returns the denoised data which is plotted below
    (middle panel) together with the original version of the data (left panel) and
    the algorithm residual (right panel) .
    """
    if plot:

        sli = data.shape[2] // 2
        gra = data.shape[3] // 2
        orig = data[:, :, sli, gra]
        den = denoised_arr[:, :, sli, gra]
        rms_diff = np.sqrt((orig - den) ** 2)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(orig, cmap='gray', origin='lower', interpolation='none')
        ax[0].set_title('Original')
        ax[0].set_axis_off()
        ax[1].imshow(den, cmap='gray', origin='lower', interpolation='none')
        ax[1].set_title('Denoised Output')
        ax[1].set_axis_off()
        ax[2].imshow(rms_diff, cmap='gray', origin='lower', interpolation='none')
        ax[2].set_title('Residual')
        ax[2].set_axis_off()
        plt.savefig(oFname+'.png', bbox_inches='tight')

        print("Result figure saved to: "+oFname+".png")

    """
    .. figure:: denoised_localpca.png
       :align: center
    
    Below we show how the denoised data can be saved.
    """

    nib.save(nib.Nifti1Image(denoised_arr,
                             affine), oFname+'.nii.gz')

    print("Denoised dataset saved to: "+oFname+".nii.gz")

    """
    .. [Manjon2013] Manjon JV, Coupe P, Concha L, Buades A, Collins DL "Diffusion
                    Weighted Image Denoising Using Overcomplete Local PCA" (2013).
                    PLoS ONE 8(9): e73021. doi:10.1371/journal.pone.0073021.
    
    .. [Koay2006]  Koay CG, Basser PJ (2006). "Analytically exact correction scheme
                   for signal extraction from noisy magnitude MR signals". JMR 179:
                   317-322.
    
    .. include:: ../links_names.inc
    """


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program denoises diffusion-weighted images using Local PCA via'
                                                 'empirical thresholds. It is built from the script proposed by the '
                                                 'fantastic DIPY team (https://dipy.org/documentation/1.3.0./examples_built/denoise_localpca/#example-denoise-localpca).')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-dwi', dest='dwiFname', help="File name of the diffusion-weighted data.", type=str, required=True)
    requiredArgs.add_argument('-bval', dest='bvalFname', help="File name of corresponding b-value file.", type=str, required=True)
    requiredArgs.add_argument('-bvec', dest='bvecFname', help="File name of corresponding b-vector file.", type=str, required=True)

    optionalArgs.add_argument('-o', dest='oFname', help='Name for the output files (both figure and denoised data so exclude extension).', type=str, required=False, default='lpca_denoised')
    optionalArgs.add_argument('-plot', dest='bool', help='Boolean to decide whether a figure should be generated or not.', type=bool, required=False, default=True)

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # # print citation
    # print('\n\n'
    #       '\n****************************** <3 Thank you for using our toolbox! <3 ***********************************'
    #       '\n********************************* PLEASE CITE THE FOLLOWING PAPER ***************************************'
    #       '\nLévy S, Rapacchi S, Massire A, et al. Intravoxel Incoherent Motion at 7 Tesla to quantify human spinal '
    #       '\ncord perfusion: limitations and promises. Magn Reson Med. 2020;00:1-20. https://doi.org/10.1002/mrm.28195'
    #       '\n*********************************************************************************************************'
    #       '\n\n')

    # run main
    main(dwiFname=args.dwiFname, bvalFname=args.bvalFname, bvecFname=args.bvecFname, oFname=args.oFname, plot=args.bool)




