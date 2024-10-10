# Raphael's workspace within MulensModel

This workspace contains the main script, auxiliary codes and templates to fit binary source (1L2S) models and generate input for subsequent binary lens (2L1S) modeling. The steps for the fitting are: *(i)* PSPL minimization using `scipy.minimize`; *(ii)* quick 1L2S fitting to find minimum flux between bumps; *(iii)* final PSPL fits with EMCEE, with data split and subtracted from one another; *(iv)* final 1L2S fit with EMCEE.

The main script is [fit_binary_source.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/fit_binary_source.py), where the `FitBinarySource` class inherits from `UlensModelFit` (see [example_16](https://github.com/rapoliveira/MulensModel/blob/develop/examples/example_16/ulens_model_fit.py)) to use many of its functions for parsing inputs and outputs. The auxiliary scripts are:
- [utils.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/.utils.py): stores utility functions that are used in various parts of the process;
- [save_results_binary_source.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/save_results_binary_source.py): generates the final outputs, including figures, chains and YAML files with results;
- [prepare_binary_lens.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/prepare_binary_lens.py): prepares the YAML input for 2L1S fitting.

Projects like "OGLE-evfinder" and "W16-59events" are organized into separate folders, each containing subfolders for photometry data and results. Use "project-name" folder to setup a new project, where you can edit the input YAML file ([1L2S_project-name.yaml](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/project-name/1L2S_project-name.yaml)) and the optional table with [peak times](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/project-name/t_peaks-file.dat), as well as adding more photometry files.

## Usage for binary source (1L2S) fitting

```
> cd exploring_MulensModel/
> python3 fit_binary_source.py OGLE-evfinder/1L2S_OGLE-evfinder.yaml
```

The configuration of the YAML file is as follows.
The key `phot_settings` contains a list of dictionaries with the key `name`, and optionally `add_2450000` (True or False) and `phot_fmt` (mag or flux).
If the value of `name` is a directory, all its photometry files will be executed in the fitting process.
If `name` is a file, it will be used for the only fitting.
Photometry files should be in `.dat` format with at least three columns, e.g., time, magnitude and magnitude error.

The rest of the YAML file is formatted as in example_16 except for the key `additional_inputs`, which can contain these optional keys:
- `t_peaks`: name of file with bump times or False. If the file is given, it should have columns with names 'obj_id', 't_peak' and 't_peak_2'. Default is False.
- `fix_blend`: float or False (default). If float is given, the PSPL fits have fixed blending_flux (usually at 0., in order to generate 2L1S initial parameters).
- `sigmas`: list of sigma values used to find starting values for EMCEE. The first item is used for PSPL(t_0, u_0, t_E) and the second for 1L2S (t_0_1, u_0_1, t_0_2, u_0_2, t_E). Default is [[0.01, 0.05, 1.0], [0.1, 0.01, 0.1, 0.01, 0.1]].
- `ans`: method to get the final solution from the EMCEE posteriors. The options are 'max_prob' (default) or 'median'.
- `yaml_files_2L1S`: information to generate input files for 2L1S. The keys are `t_or_f`, `yaml_template` and `yaml_dir_name`.

## Usage for binary lens (2L1S) fitting and others

Simplify a lot, refer to example_16. Mention the bash script like [run_multiple_fits.sh](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/run_multiple_fits.sh) can be used to run EMCEE or UltraNest for several events at once.

*Explain [generate_ultranest_files.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/OGLE-evfinder/generate_ultranest_files.py) (add check for negative t_E) and [combine_1L2S_2L1S_results.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/OGLE-evfinder/combine_1L2S_2L1S_results.py) scripts... Change everything for the example project, after adding the simulated data from example_11.*

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
- [X] Classes: FitBinarySource, SaveResultsBinarySource, Utils and PrepareBinaryLens.
- [ ] Add unit tests? How?
- [X] Facilitate the addition of new data: different folders in [OGLE-evfinder/phot](https://github.com/rapoliveira/MulensModel/tree/develop/exploring_MulensModel/OGLE-evfinder/phot)?
- [X] **URGENT:** Solve stuck EMCEE chains using blending flux...
- [X] Write priors setup to yaml files: 1L2S results and input file for 2L1S
- [X] Speed-up the saving of the chains with h5py and npy*
- [ ] Write instructions to run and improve README, with examples (**1/2 done**)
- [X] Limit to u_0 > 0 also in 2L1S: change alpha by 180 if initial u_0 < 0 (see Skowron+2011)
- [ ] [...]

### New script for 2xPSPL fits: split and subtract data from the beginning
- [X] NEW SCRIPT: split data after 1L2S to get the initial 2L1S parameters (end of October)
- [X] Split data as first step, before any fit (03.nov)
- [X] Improve pipeline/steps with new function with prefit and split (07-10.nov)
- [X] Script to plot multiple fits for the same data (22-28.nov)
- [X] Apply gradient in scipy_minimize of PSPL fit
- [ ] Check if there is a second peak above 3 sigma to go for 1L2S...
- [ ] Include and test all cases, including those with a single peak or skewed(?)

Last updated: 10 Oct 2024
