#

#  :mag_right: :bulb: ivim-toolbox :flashlight: :wrench: </font>

# 

<p align="center">
This toolbox is dedicated to model fitting and simulation for Intra-Voxel Incoherent Motion (IVIM) MR imaging. It gathers tools, on the one hand, to fit IVIM bi-exponential signal representation voxel-wise and, on the other hand, to run Monte Carlo simulations in order to assess the estimation error on parameters as well as to calculate the minimum required SNR to get accurate estimation.</p>

We thank you for choosing our toolbox! :heart: According to the MIT licence, please cite the following article:
> **Lévy S, Rapacchi S, Massire A, et al. Intravoxel Incoherent Motion at 7 Tesla to quantify human spinal cord perfusion: limitations and promises, *Magn Reson Med*. 2020;00:1–20. [https://doi.org/10.1002/mrm.28195](https://doi.org/10.1002/mrm.28195)**

---

# Table of Contents

- [Operating systems](#operating-systems)
- [Installation](#installation)
- [Data description](#data-description)
- [Code description](#code-description)
- [Get started](#get-started)
- [Funding](#funding)

---

# Operating systems
This toolbox is only based on Python. Therefore, it should run on any OS (Mac OSX, Linux, Windows). However, it has been developed and extensively tested on Mac OSX. Some light errors might pop up when using on Linux or Windows but **feel free to open an issue [here](https://github.com/slevyrosetti/ivim-toolbox/issues) or to contact me at simon.levy@mines-ales.org if you face any problem.**

---
# Installation
Only requirements are:
  1. Python
  2. Specific Python modules (i.e. not installed in default Python)
  3. ivim-toolbox files
  
  ## 1. Python
We recommend to install Python through the [Conda package](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). It is very easy, follow the link, select your Operating System and run the installer (either for Miniconda or Anaconda).
You can also download Python [here](https://www.python.org/downloads/).

  ## 2. Specific Python modules
Once Python is intalled, install the required Python modules:
  - lmfit
  - argparse
  - nibabel
  - multiprocessing
  - numpy
  - matplotlib
  - cPickle
  - wxpython
  
To do so, just open a Terminal and, for each module, type `conda install <module name>` where `<module name>` has to be replaced by the name of the module.

*Some modules such as `numpy`, `nibabel` or `matplotlib` are already included in the Conda package so no need to install them.
You can also update modules typing `conda update <module name>`.*

If you did not install Python through the Conda package, you can install those modules typing `pip install <module name>`.

*The `pip` command should be available on default Python but if not, you can find information on how to install it [here](https://packaging.python.org/tutorials/installing-packages/). The command to update modules with `pip` is `pip install <module name> --upgrade`.*

  ## 3. ivim-toolbox
Finally, [download](https://github.com/slevyrosetti/ivim-toolbox/archive/master.zip) this repository. Add the path to your `.bash_profile` (or `.bashrc` as preferred). To do so, type the following command in a Terminal:

`echo 'export PATH=<path to ivim-toolbox directory>:$PATH' >>~/.bash_profile` where `<path to ivim-toolbox directory>` has to be replaced by the path to the ivim-toolbox directory (e.g. `/Users/slevyrosetti/ivim-toolbox`).

---

# Data description

You will find in the directory `test_data` example data that you can use to test your installation:
 - `dwi_rl.nii.gz` are human spinal cord IVIM data acquired at 7T with diffusion encoding in the Right-Left direction and with b-values as defined in `bval_rl.txt`
 - `dwi_ap.nii.gz` are human spinal cord IVIM data acquired at 7T with diffusion encoding in the Anterior-Posterior direction and with b-values as defined in `bval_rl.txt`
 - `dwi_is.nii.gz` are human spinal cord IVIM data acquired at 7T with diffusion encoding in the Inferior-Superior direction and with b-values as defined in `bval_rl.txt`
 - `cord_seg_dilated.nii.gz` is a large mask including the cord (to use for voxel-wise fitting)
 - directory `results_you_should_get` includes the results you should get using the example commands detailed below

Those data are the single-subject data presented in Figure 4 of the paper *Levy S., Rapacchi S., Massire A., Troalen T., Feiweier T., Guye M., Callot V., Intra-Voxel Incoherent Motion at 7T to quantify human spinal cord microperfusion: pitfalls and promises, Proceedings of the 27th Annual Meeting of the International Society for Magnetic Resonance in Medicine. Montreal, Quebec, Canada; 2019:0301.*

---

# Code description

The tools available are:
  - `ivim_fitting.py`: fit IVIM biexponential signal representation to NIFTI data according to specified fitting approach
  - `ivim_view_fits.py`: display an IVIM parameter map and enable user to inspect fitting by clicking on any voxel and display corresponding fit plot
  - `ivim_simu_compute_error_nonoise.py`: compute error of a given fitting approach according to true IVIM values
  - `ivim_simu_plot_error_nonoise.py`: plot results from previous tool
  - `ivim_simu_compute_error_noise.py`: compute error of a given fitting approach according to true IVIM values for a given SNR (Monte Carlo simulations)
  - `ivim_simu_plot_error_noise.py`: plot results from previous tool
  - `ivim_simu_compute_required_snr.py`: compute required SNR to estimate parameters within 10% error margins for a given fitting approach and according to true IVIM values
  - `ivim_simu_plot_required_snr.py`: plot results from previous tool
  - `ivim_toolbox.py`: launch the graphical user interface
  
All tools can be used from the Terminal and some of the them can be run through a graphical user interface.

**To display all available options for any tool, type `<name of tool>.py --help` in a Terminal (e.g. `ivim_fitting.py --help`).**

---
# Get started

In this section, we will give examples of the commands that can be run from a Terminal window to show how to use the toolbox for your IVIM data.

## Open graphical user interface
```
ivim_toolbox.py
```
Two frames will open. The first is to fit the IVIM biexponential signal representation to NIFTI data according to specified fitting approach (similar to function `ivim_fitting.py`). The second is to compute required SNR to estimate parameters within 10% error margins for a given fitting approach and according to true IVIM values (similart to `ivim_simu_compute_required_snr.py`).

## Fit IVIM data
Go to the folder where the data are stored:
```
cd test_data
```

Fit the IVIM data acquired with diffusion encoding in the Right-Left direction, voxel-by-voxel for voxels within the mask only, using the one-step fitting approach and running on all threads available to speed up the calculation. IVIM maps will be output in a folder named `ivim_maps_rl`:
```
ivim_fitting.py -i dwi_rl.nii.gz -b bval_rl.txt -ma cord_seg_dilated.nii.gz -mo one-step -o ivim_maps_rl -mt 1
```

A folder named `<creation date>_plots` has been created in the result folder (`ivim_maps_rl`). This folder includes plots of the fit performed at the previous step, and as we would like to inspect the data and how the fit algorithm performed on each voxel, we type:
```
ivim_view_fits.py -i ivim_maps_rl/Fivim_map.nii.gz -plotdir ivim_maps_rl/190719170259_plots/ -param cmap=jet,clim=0\;0.3
```

Two windows open, the first displays the f<sub>IVIM</sub> map with colormap "jet" and with values from 0 to 30%. Now if you click on a voxel, the second window will display the corresponding fit plot. The Terminal will display the voxel coordinates and its value (if fit was performed on this voxel).
You can also zoom in using the :mag: icon.

*Note: This script was largely inspired from the script `sct_viewer.py` of the [Spinal Cord Toolbox project](https://github.com/neuropoly/spinalcordtoolbox)*

> *De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D, Callot V, Cohen-Adad J., SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord MRI data. Neuroimage 2017. [https://www.ncbi.nlm.nih.gov/pubmed/27720818](https://www.ncbi.nlm.nih.gov/pubmed/27720818)*

*We would like to truly thank the Spinal Cord Toolbox team, and in particular Benjamin De Leener @benjamindeleener and Julien Cohen-Adad @jcohenadad, for developing this script!*

## Check performance of fitting algorithm on perfect data
Compute estimation error of the one-step fit approach on perfect data with f<sub>IVIM</sub> values varying from 1 to 30%, D* values varying from 3 to 35e-3 mm<sup>2</sup>/s and D varying from 0.2 to 2.9e-3 mm<sup>2</sup>/s; a result file will be created in folder `one_step_fit_err`:
```
ivim_simu_compute_error_nonoise.py -model one-step -ofolder one_step_fit_err -bval 5,10,20,30,50,75,150,250,600,700,800
```
Plot the results in folder "one_step_fit_err" with "error_plot" as output file name:
```
ivim_simu_plot_error_nonoise.py -input one_step_fit_err/sim_results_*.pkl -oname one_step_fit_err/error_plot
```

## Compute estimation error of fitting algorithm for a given SNR
Compute estimation error of the two-step fit approach on simulated data with SNR=180 (Monte-Carlo simulations) with IVIM true values varying in the same range as the previous one; similarly, a result file will be created in folder `two_step_fit_err_snr180`:
```
ivim_simu_compute_error_noise.py -model two-step -snr 180 -ofolder two_step_fit_err_snr180 -bval 5,10,20,30,50,75,150,250,600,700,800
```
*NB: If you run it on a "standard" machine (with around 4 cores), this command will take a VERY LONG time; we recommend to run it on a machine with at least 8 cores.*

Plot the results in folder "two_step_fit_err_snr180" with "error_plot" as output file name:
```
ivim_simu_plot_error_noise.py -input two_step_fit_err_snr180/sim_results_*.pkl -oname two_step_fit_err_snr180/error_plot
```

## Calculate required SNR
Calculate the minimum required SNR to estimate the product of parameters f<sub>IVIM</sub> and D* within 10% error margins for f<sub>IVIM</sub> varying from 1 to 30% (with 10 steps), D* varying from 3 to 35e-3 mm<sup>2</sup>/s (with 10 steps) and D equals 0.3 and 1.5e-3 mm<sup>2</sup>/s
```
ivim_simu_compute_required_snr.py -model one-step -ofolder required_snr_one_step_fit -bval 5,10,20,30,50,75,150,250,600,700,800 -condition FDstar -F 0.01:10:0.30 -Dstar 3.0e-3:10:35e-3 -D 0.3e-3,1.5e-3
```
*NB: This command took **26h30min** on a linux machine with **40 Intel(R) Xeon(R) CPUs E5-2630 v4 @ 2.20GHz** so you might want to consider the specifications of the machine you want to use to run this command.*

A result file has been created to folder "required_snr_one_step_fit", let's plot the results now:
```
ivim_simu_plot_required_snr.py -input required_snr_one_step_fit/sim_results_*.pkl -oname required_snr_one_step_fit/required_snr_plot
```

---
# Publications where the IVIM-toolbox was used

## Scientific articles
- Lebret A, Lévy S, Freund P, Callot V, Seif M. **Test-retest repeatability of intravoxel incoherent motion (IVIM) measurements in the cervical cord**. *Imaging Neuroscience*. 2025; 3. [https://doi.org/10.1162/imag_a_00468](
https://doi.org/10.1162/imag_a_00468)
- Lebret A, Frese S, Lévy S, Curt A, Callot V, Freund P, Seif M. **Spinal Cord Blood Perfusion Deficit is Associated with Clinical Impairment after Spinal Cord Injury**. *Journal of Neurotrauma*. 2025;42:3-4, 280-291. [https://doi.org/10.1089/neu.2024.0267](https://doi.org/10.1089/neu.2024.0267).
- Lebret A, Lévy S, Pfender N, Farshad M, Altorfer FC, Callot V, Curt A, Freund P, Seif M. **Investigation of perfusion impairment in degenerative cervical myelopathy beyond the site of cord compression**. *Scientific Reports*. 2023;13(1):22660 [https://doi.org/10.1038/s41598-023-49896-3](https://doi.org/10.1038/s41598-023-49896-3).
- Lévy S, Rapacchi S, Massire A, et al. **Intravoxel Incoherent Motion at 7 Tesla to quantify human spinal cord perfusion: limitations and promises**. *Magn Reson Med*. 2020;00:1–20. [https://doi.org/10.1002/mrm.28195](https://doi.org/10.1002/mrm.28195)


## Conference proceedings
- Lebret A, Frese S, Lévy S, et al. **Intravoxel incoherent motion (IVIM) MRI in the cervical cord: application to traumatic spinal cord injury**. *In Proc. Int. Soc. Magn. Reson. Med. 33*, 2332. Singapore, 2024.
- Lebret A, Lévy S, Freund P, et al. **Test-retest repeatability of intravoxel incoherent motion (IVIM) MRI in the cervical cord at 3T**. *In Proc. Int. Soc. Magn. Reson. Med. 32*, 2591. Singapore, 2024.
- Lebret A, Lévy S, Pfender N, et al. **Investigating spinal cord perfusion impairment in degenerative cervical myelopathy (DCM) using Intravoxel Incoherent Motion (IVIM) MRI**. *In Proc. Int. Soc. Magn. Reson. Med. 32*, 1145. Toronto, ON, Canada, 2023.
- Lévy S, Freund P, Callot V, Seif M. **Spinal cord perfusion mapping using Intra-Voxel Incoherent Motion at 3T in healthy individuals and Degenerative Cervical Myelopathy patients**. *In Proc. Int. Soc. Magn. Reson. Med. 30*, 3462. Virtual Conference, 2021.
- Lévy S, Rapacchi S, Massire A, et al. **Intra-Voxel Incoherent Motion at 7T to Quantify Human Spinal Cord Microperfusion: Pitfalls and Promises**. *In Proc. Int. Soc. Magn. Reson. Med. 27*, 0301. Montreal, QC, Canada, 2018.
- 

---
# Funding

*This work was performed within the [CRMBM-CEMEREM](http://crmbm.univ-amu.fr/) (UMR 7339, CNRS / Aix-Marseille University), which is a laboratory member of France Life Imaging network (grant #ANR-11-INBS-0006). The project received funding from the European Union’s Horizon 2020 research and innovation program (Marie Skłodowska-Curie grant agreement #713750), the Regional Council of Provence-Alpes-Côte d’Azur, A\*MIDEX (#ANR-11-IDEX-0001-02, #7T-AMI-ANR-11-EQPX-0001, #A\*MIDEX-EI-13-07-130115-08.38-7T-AMISTART) and CNRS (Centre National de la Recherche Scientifique).*
