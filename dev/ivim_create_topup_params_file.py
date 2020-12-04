#!/usr/bin/env python3.6

import argparse
import json

def main(fwd_jsonFname, rvs_jsonFname, oFname):
    """Main."""

    # load json files
    with open(fwd_jsonFname) as fwd_jFile:
        fwd_acqParams = json.load(fwd_jFile)
    with open(rvs_jsonFname) as rvs_jFile:
        rvs_acqParams = json.load(rvs_jFile)

    # print total readout time as written in the json file
    print("Total readout time for the FORWARD acquisition according to JSON file = {}".format(fwd_acqParams["TotalReadoutTime"]))
    print("Total readout time for the REVERSE acquisition according to JSON file = {}".format(rvs_acqParams["TotalReadoutTime"]))

    # compute total readout time
    fwd_trt = (fwd_acqParams["AcquisitionMatrixPE"] - 1)/fwd_acqParams["PixelBandwidth"]
    rvs_trt = (rvs_acqParams["AcquisitionMatrixPE"] - 1)/rvs_acqParams["PixelBandwidth"]
    print("Total calculated readout time for the FORWARD acquisition = {}".format(fwd_trt))
    print("Total calculated readout time for the REVERSE acquisition = {}".format(rvs_trt))


    # write parameter file to be used in TOPUP
    fileObj = open(oFname+'.txt', 'w')
    jsonCodePhaseEncDir = ["i", "j", "k", "i-", "j-", "k-"]
    phaseEncDir = ["1 0 0", "0 1 0", "0 0 1", "-1 0 0", "0 -1 0", "0 0 -1"]

    # forward phase-encoding direction
    fileObj.write("{} {:f}\n".format(phaseEncDir[jsonCodePhaseEncDir.index(fwd_acqParams["PhaseEncodingDirection"])], fwd_acqParams["TotalReadoutTime"]))
    # reverse phase-encoding direction
    fileObj.write("{} {:f}\n".format(phaseEncDir[jsonCodePhaseEncDir.index(rvs_acqParams["PhaseEncodingDirection"])], rvs_acqParams["TotalReadoutTime"]))

    fileObj.close()

    print('\n>>> TOPUP parameter file saved to: {}.txt'.format(oFname))


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program reads the JSON files extracted from the DICOM of forward '
                                                 'and reverse phase-encoded acquisitions and create the parameter file '
                                                 'necessary to run TOPUP.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-fwd', dest='fwd_jsonFname', help='File name of the JSON file from forward acquisition.', type=str, required=True)
    requiredArgs.add_argument('-rvs', dest='rvs_jsonFname', help='File name of the JSON file from reverse acquisition.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help="Output file name for the TOPUP parameter file.", type=str, required=True)

    # optionalArgs.add_argument('-order', dest='order', help='Enter \"bval\" for an order by b-value or \"rep\" for an order by repetition. Note that the order repetition-wise can only be performed if the same number of repetitions per b-value is requested.', type=str, required=False, default="bval")

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
    main(fwd_jsonFname=args.fwd_jsonFname, rvs_jsonFname=args.rvs_jsonFname, oFname=args.oFname)




