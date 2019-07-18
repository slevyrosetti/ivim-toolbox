# ivim-toolbox :100:
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

  ## ivim-toolbox
Finally, [download](https://github.com/slevyrosetti/ivim-toolbox/archive/master.zip) this repository. Add the path to your `.bash_profile` (or `.bashrc` as preferred). To do so, type the following command in a Terminal:

`echo 'export PATH=<path to ivim-toolbox directory>:$PATH' >>~/.bash_profile` where `<path to ivim-toolbox directory>` has to be replaced by the path to the ivim-toolbox directory (e.g. `/Users/slevyrosetti/ivim-toolbox`).

# Get started
*Some of the tools can be run through a graphical user interface; to jump there click [here](https://github.com/slevyrosetti/ivim-toolbox#installation).*

You will find in the directory `test_data` example data that you can use to test your installation:

All tools can be used from the Terminal and some of the them can be run through a graphical user interface; to jump there click [here](https://github.com/slevyrosetti/ivim-toolbox#installation).

Let's start by the Terminal use. So to get started, open a Terminal.

The tools available in this toolbox are the following:
  - `ivim_fitting.py`: fit IVIM biexponential signal representation to NIFTI data according to specified fitting approach
  - `ivim_simu_compute_error_nonoise.py`: compute error of a given fitting approach according to true IVIM values
  - `ivim_simu_plot_error_nonoise.py`: plot results from previous tool
  - `ivim_simu_compute_error_noise.py`: compute error of a given fitting approach according to true IVIM values for a given SNR (Monte Carlo simulations)
  - `ivim_simu_plot_error_noise.py`: plot results from previous tool
  - `ivim_simu_compute_required_snr.py`: compute required SNR to estimate parameters within 10% error margins for a given fitting approach and according to true IVIM values
  - `ivim_simu_plot_required_snr.py`: plot results from previous tool
  - `ivim_toolbox.py`: launch the graphical user interface
To display help for any tool, type `<name of tool>.py -h`.

## Example commands






