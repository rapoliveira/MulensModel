# Raphael's workspace within the MulensModel repository

The main script is [fit_1L2S_3steps.py](https://github.com/rapoliveira/MulensModel/blob/master/exploring_MulensModel/fit_1L2S_3steps.py), which does...

## Run binary source (1L2S) fitting

Add a folder containing the data, yaml file, folders for output...

The first key of the YAML file is `phot_settings`, which contains a list of dictionaries with at least `name`, `add_2450000` (True or False) and `phot_fmt` (mag or flux) keys.
For each item, if the value of `name` is a directory, all the photometry files inside it will be executed in the fitting process.
If `name` is a file, it will be used for the only fitting.
The photometry file should be in .dat format with at least three columns, i.e. time, magnitude and magnitude error.

## Run binary lens (2L1S) fitting

ulens_model_fit.py yaml file...

## To-Do List (Raphael)
- [X] Improve fixed value of blending_flux in the yaml input
- [X] Function write_tables() to write the chains, a yaml with the results, and a table with all results
- [X] Correct major bug related to source/blending fluxes retrieved from emcee.blobs
- [X] Best model with all digits in results.yaml and get back fixing the fluxes in fit figure
- [X] Implement priors, minimum values and starting parameters (if applicable) from the yaml file
- [X] Correct error with minimum values and priors... bad results
- [X] Add degree of freedom to the table with all results and the results yaml file
- [X] Correct issues on generating 2L1S yaml files. Plus: plot the 2L1S model (saved in same folder)
- [X] Convert the entire fit_1L2S_3steps.py code to 2450000 system, in agreement with 2L1S codes (26.dec)
- [ ] Unit tests, docstrings, **classes** (e.g. Data, Fitting, Tables)
- [ ] [...]
- [ ] Make the code more general for other databases... 1/2

### New script for 2xPSPL fits: split and subtract data from the beginning
- [X] NEW SCRIPT: split data after 1L2S to get the initial 2L1S parameters (end of October)
- [X] Split data as first step, before any fit (03.nov)
- [X] Improve pipeline/steps with new function with prefit and split (07-10.nov)
- [X] Script to plot multiple fits for the same data (22-28.nov)
- [ ] Check if there is a second peak above 3 sigma to go for 1L2S...
- [ ] Include and test all cases, including those with a single peak or skewed(?)
- [X] Apply gradient in scipy_minimize of PSPL fit

Last updated: 19 Aug 2024
