# ivim-toolbox
This toolbox is dedicated to model fitting and simulation for Intra-Voxel Incoherent Motion (IVIM) MR imaging. It gathers scripts, on the one hand, to fit IVIM bi-exponential signal representation voxel-wise and, on the other hand, to run Monte Carlo simulations in order to assess the estimation error on parameters as well as the minimum required SNR to get accurate estimation.

# Operating systems
This toolbox is only based on Python. Therefore, it should run on any OS (Mac OSX, Linux, Windows). However, it has been developed and extensively tested on Mac OSX. Some light errors might pop up when using on Linux or Windows but **feel free to open an issue [here](https://github.com/slevyrosetti/ivim-toolbox/issues) or to contact me at simon.levy@mines-ales.org if you face any problem.**

# Installation
Only requirements are:
  1. Python
  2. Specific Python modules (i.e. not installed in default Python)
  
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
