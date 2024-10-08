# Raphael's workspace within MulensModel

This workspace contains the main script, auxiliary codes and templates to fit binary source (1L2S) models and produce input to run binary lens (2L1S) models afterwards. The fitting is carried out following a number of steps: *(i)* quick PSPL minimization with scipy.minimize; *(ii)* quick 1L2S estimation to find minimum flux between bumps; *(iii)* final PSPL fits with EMCEE, with split data and subtracting one from the other; *(iv)* final 1L2S fit with EMCEE as result.

The main script is [fit_binary_source.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/fit_binary_source.py), where FitBinarySource class inherits UlensModelFit (see [example_16](https://github.com/rapoliveira/MulensModel/blob/develop/examples/example_16/ulens_model_fit.py)) to use a lot of its functions to parse inputs and outputs. The auxiliary codes are:
- [utils.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/.utils.py): stores utilitary functions, used in several parts of the process;
- [save_results_binary_source.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/save_results_binary_source.py): produces all the outputs (figures, chains and yaml files with results);
- [prepare_binary_lens.py](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/prepare_binary_lens.py): prepares the YAML input for 2L1S fitting.

Projects like "OGLE-evfinder" and "W16-59events" are separated in different folders, with folders for photometry and results inside it. Use "project-name" to setup a new project, where you must edit the input YAML file ([1L2S_project-name.yaml](https://github.com/rapoliveira/MulensModel/blob/develop/exploring_MulensModel/OGLE-evfinder/1L2S_OGLE-evfinder.yaml)), add photometry files and an optional table with the peak times.

### Instructions to run binary source (1L2S) fitting

```
> cd exploring_MulensModel/
> python3 fit_binary_source.py OGLE-evfinder/1L2S_OGLE-evfinder.yaml
```

The first key of the YAML file is `phot_settings`, which contains a list of dictionaries with the key `name`, and optionally the keys `add_2450000` (True or False) and `phot_fmt` (mag or flux).
If the value of `name` is a directory, all the photometry files inside it will be executed in the fitting process.
If `name` is a file, it will be used for the only fitting.
The photometry file should be in .dat format with at least three columns, i.e. time, magnitude and magnitude error.

The rest of the YAML file is formatted as in example_16 except for the key `additional_inputs`, which can contain these optional keys:
- `t_peaks`: name of file with bump times or False. If the file is given, it should have columns with names 'obj_id', 't_peak' and 't_peak_2'. Default is False.
- `fix_blend`: float or False (default). If float is given, the PSPL fits have fixed blending_flux (usually at 0., in order to generate 2L1S initial parameters).
- `sigmas`: list of sigma values used to find starting values for EMCEE. The first item is used for PSPL(t_0, u_0, t_E) and the second for 1L2S (t_0_1, u_0_1, t_0_2, u_0_2, t_E). Default is [[0.01, 0.05, 1.0], [0.1, 0.01, 0.1, 0.01, 0.1]].
- `ans`: method to get the final solution from the EMCEE posteriors. The options are 'max_prob' (default) or 'median'.
- `yaml_files_2L1S`: information to generate input files for 2L1S. The keys are `t_or_f`, `yaml_template` and `yaml_dir_name`.

## Instructions to run binary lens (2L1S) fitting

```
> cd exploring_MulensModel/
> python3 ../examples/example_16/ulens_model_fit.py OGLE-evfinder/yaml_files_1L2S/<event_id>-2L1S_traj_between.yaml
```

A .sh file like ... can also be used to run for several events at once.

## Generate UltraNest files, run it and make final table (with evidence)

Add generate_ultranest_files.py script? Add combine-1L2S-2L1S-results.py script? *Add description...*

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
- [ ] Speed-up the saving of the chains with backend, h5py
- [ ] Write instructions to run and improve README (**1/2 done**)
- [ ] [...]

### New script for 2xPSPL fits: split and subtract data from the beginning
- [X] NEW SCRIPT: split data after 1L2S to get the initial 2L1S parameters (end of October)
- [X] Split data as first step, before any fit (03.nov)
- [X] Improve pipeline/steps with new function with prefit and split (07-10.nov)
- [X] Script to plot multiple fits for the same data (22-28.nov)
- [X] Apply gradient in scipy_minimize of PSPL fit
- [ ] Check if there is a second peak above 3 sigma to go for 1L2S...
- [ ] Include and test all cases, including those with a single peak or skewed(?)

Last updated: 08 Oct 2024
