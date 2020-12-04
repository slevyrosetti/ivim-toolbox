#!/usr/bin/env python3.6

"""
===============================================================================
Suppress Gibbs oscillations
===============================================================================

Magnetic Resonance (MR) images are reconstructed from the Fourier coefficients
of acquired k-space images. Since only a finite number of Fourier coefficients
can be acquired in practice, reconstructed MR images can be corrupted by Gibbs
artefacts, which is manifested by intensity oscillations adjacent to edges of
different tissues types [1]_. Although this artefact affects MR images in
general, in the context of diffusion-weighted imaging, Gibbs oscillations
can be magnified in derived diffusion-based estimates [1]_, [2]_.

In the following example, we show how to suppress Gibbs artefacts of MR images.
This algorithm is based on an adapted version of a sub-voxel Gibbs suppression
procedure [3]_. Full details of the implemented algorithm can be found in
chapter 3 of [4]_  (please cite [3]_, [4]_ if you are using this code).

The algorithm to suppress Gibbs oscillations can be imported from the denoise
module of dipy:
"""

from dipy.denoise.gibbs import gibbs_removal
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti
from numpy import max


import argparse


def main(iFname, oFname, plot):
    """Main."""

    # load data
    data, affine = load_nifti(iFname)

    """
    Gibbs oscillation suppression of all multi-shell data and all slices
    can be performed in the following way:
    """

    data_corrected = gibbs_removal(data, slice_axis=2, num_threads=None)

    """
    Due to the high dimensionality of diffusion-weighted data, we recommend
    that you specify which is the axis of data matrix that corresponds to different
    slices in the above step. This is done by using the optional parameter
    'slice_axis'.
    
    Below we plot the results for an image acquired with b-value=0:
    """

    if plot:
        fig2, ax = plt.subplots(1, 3, figsize=(12, 6),
                                subplot_kw={'xticks': [], 'yticks': []})

        ax.flat[0].imshow(data[:, :, 0, 0].T, cmap='gray', origin='lower',
                          vmin=0, vmax=max(data))
        ax.flat[0].set_title('Uncorrected b0')
        ax.flat[1].imshow(data_corrected[:, :, 0, 0].T, cmap='gray',
                          origin='lower', vmin=0, vmax=max(data))
        ax.flat[1].set_title('Corrected b0')
        ax.flat[2].imshow(data_corrected[:, :, 0, 0].T - data[:, :, 0, 0].T,
                          cmap='gray', origin='lower', vmin=-500, vmax=500)
        ax.flat[2].set_title('Gibbs residuals')

        # plt.show()
        """
        .. figure:: Gibbs_suppression_b0.png
        :align: center
    
        Uncorrected (left panel) and corrected (middle panel) b-value=0 images. For
        reference, the difference between uncorrected and corrected images is shown
        in the right panel.
        """
        fig2.savefig(oFname+'.png')

        print("Result figure saved to: "+oFname+".png")

    # save as Nii
    nib.save(nib.Nifti1Image(data_corrected, affine), oFname+'.nii.gz')

    print("Ungibbsed dataset saved to: "+oFname+".nii.gz")

    """
    References
    ----------
    .. [1] Veraart, J., Fieremans, E., Jelescu, I.O., Knoll, F., Novikov, D.S.,
           2015. Gibbs Ringing in Diffusion MRI. Magn Reson Med 76(1): 301-314.
           https://doi.org/10.1002/mrm.25866
    .. [2] Perrone, D., Aelterman, J., Pižurica, A., Jeurissen, B., Philips, W.,
           Leemans A., 2015. The effect of Gibbs ringing artifacts on measures
           derived from diffusion MRI. Neuroimage 120, 441-455.
           https://doi.org/10.1016/j.neuroimage.2015.06.068.
    .. [3] Kellner, E., Dhital, B., Kiselev, V.G, Reisert, M., 2016. Gibbs‐ringing
           artifact removal based on local subvoxel‐shifts. Magn Reson Med
           76:1574–1581.
           https://doi.org/10.1002/mrm.26054.
    .. [4] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain
           (Doctoral thesis). https://doi.org/10.17863/CAM.29356
    .. [5] Valabrègue, R. (2015). Diffusion MRI measured at multiple b-values.
           Retrieved from:
           https://digital.lib.washington.edu/researchworks/handle/1773/33311
    
    .. include:: ../links_names.inc
    """


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program suppresses Gibbs oscillations in MRI data. It is built '
                                                 'from the script proposed by the fantastic DIPY '
                                                 'team (https://dipy.org/documentation/1.3.0./examples_built/denoise_gibbs/#example-denoise-gibbs).')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help="File name of the data to process.", type=str, required=True)

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
    main(iFname=args.iFname, oFname=args.oFname, plot=args.bool)




