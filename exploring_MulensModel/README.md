# Raphael's workspace within the MulensModel repository

The main script is [fit_binary_source.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/fit_binary_source.py), where the FitBinarySource class inherits UlensModelFit from [example_16](https://github.com/rapoliveira/MulensModel/blob/develop/examples/example_16/ulens_model_fit.py), using a lot of its functions to parse inputs and outputs.

*Add a description of the fitting pipeline...*

## Run binary source (1L2S) fitting

*Add a folder containing the data, yaml file for 1L2S, folders for output...*

The first key of the YAML file is `phot_settings`, which contains a list of dictionaries with at least `name`, `add_2450000` (True or False) and `phot_fmt` (mag or flux) keys.
For each item, if the value of `name` is a directory, all the photometry files inside it will be executed in the fitting process.
If `name` is a file, it will be used for the only fitting.
The photometry file should be in .dat format with at least three columns, i.e. time, magnitude and magnitude error.

The rest of the YAML file is formatted as in example_16 except for the key `additional_inputs`, which can contain these optional keys:
- `t_peaks`: name of file with bump times of False. If the file is given, it should have columns with names 'obj_id', 't_peak' and 't_peak_2'. Default is False.
- `fix_blend`: float or False (default). If float is given, the PSPL fits have fixed blending_flux (usually at 0., in order to generate 2L1S initial parameters).
- `sigmas`: list of sigma values used to find starting values for EMCEE. The first item is used for PSPL(t_0, u_0, t_E) and the second for 1L2S (t_0_1, u_0_1, t_0_2, u_0_2, t_E). Default is [[0.01, 0.05, 1.0], [0.1, 0.01, 0.1, 0.01, 0.1]].
- `ans`: method to get the final solution from the EMCEE posteriors. The options are 'max_prob' (default) or 'median'.
- `yaml_files_2L1S`: information to generate input files for 2L1S. The keys are `t_or_f`, `yaml_template` and `yaml_dir_name`.

## Run binary lens (2L1S) fitting

*ulens_model_fit.py yaml file...*

## Final table with 1L2S vs 2L1S results (with evidence)

*Add description...*

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
- [X] Classes: FitBinarySource, Utils, PrepareBinaryLens...
- [ ] Add unit tests? How?
- [ ] Facilitate the addition of new data: different folders in [OGLE-evfinder/phot](https://github.com/rapoliveira/MulensModel/tree/develop/exploring_MulensModel/OGLE-evfinder/phot)?
- [ ] [...]

### New script for 2xPSPL fits: split and subtract data from the beginning
- [X] NEW SCRIPT: split data after 1L2S to get the initial 2L1S parameters (end of October)
- [X] Split data as first step, before any fit (03.nov)
- [X] Improve pipeline/steps with new function with prefit and split (07-10.nov)
- [X] Script to plot multiple fits for the same data (22-28.nov)
- [X] Apply gradient in scipy_minimize of PSPL fit
- [ ] Check if there is a second peak above 3 sigma to go for 1L2S...
- [ ] Include and test all cases, including those with a single peak or skewed(?)

Last updated: 18 Sep 2024
