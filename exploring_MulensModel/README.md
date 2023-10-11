# Raphael's workspace within the MulensModel repository 

This script multiplies...

The main script is [fit_1L2S_3steps.py](https://github.com/rapoliveira/MulensModel/blob/master/exploring_MulensModel/fit_1L2S_3steps.py), which does...

<!-- - Give other references or details from the ESO proposal? -->

<!-- ## Instructions to run the script

If the Python libraries astropy, matplotlib, numpy and scipy are installed, a single command do the entire analysis and produces the two outputs:
```
python3 multiply_integrate_spectra_tcurves.py
```

Other transmission curves and spectra can be adopted, as the functions are general and a 1d-interpolation is applied to account for the different grids in wavelength.
New transmission curves should be added to the [transm_curves/](https://github.com/rapoliveira/ESO_prop_extinction/tree/master/transm_curves) folder, with the first two columns containing the wavelength and efficiency.
New stellar spectra in fits format should be added to the [spectra/](https://github.com/rapoliveira/ESO_prop_extinction/tree/master/spectra) folder and listed in the file spectra_coords.txt. The spectra should contain at least two columns named as WAVE and FLUX, as well as the X-Shooter spectra.

It is possible to compute JHKs magnitudes of the stellar spectra using the transmission curves from HAWK-I or 2MASS ([Skrutskie et al. 2006](https://ui.adsabs.harvard.edu/abs/2006AJ....131.1163S/abstract)), to be compared with values from Simbad or VVV. To do that for each of the JHKs filters, edit line 182 and uncomment line 265 of the main code, and print the array JHKmag to the terminal. -->

## To-Do List (Raphael)
<!-- - *old: I will also review the code one last time and improve the derivation of JHKs mags.* -->
<!-- - URGENT: Deal with more than one spectrum for the same star (weighted average) ->> Only duplicated spectra working so far. -->
- [X] Improve fixed value of blending_flux in the yaml input
- [X] Function write_tables() to write the chains, a yaml with the results, and a table with all results
- [X] Correct major bug related to source/blending fluxes retrieved from emcee.blobs
- [X] Best model with all digits in results.yaml and get back fixing the fluxes in fit figure
- [ ] Implement priors, minimum values and starting parameters (if applicable) from the yaml file
- [X] Add degree of freedom to the table with all results and the results yaml file
- [ ] Correct issue on generating 2L1S yaml files...
- [ ] Make unit tests and document the entire code
- [ ] Try to make it in classes (e.g. Data, Fitting, Tables)
- [ ] [...]
- [ ] Make the code more general for other databases...

Last updated: 04 Oct 2023
