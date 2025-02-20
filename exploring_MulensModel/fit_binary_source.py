"""
Class and script for fitting binary source model using MulensModel.
All settings are read from a YAML file.
"""
import copy
from itertools import chain
import os
import sys
import warnings

from astropy.table import Table
import emcee
import MulensModel as mm
import numpy as np
import yaml

from save_results_binary_source import SaveResultsBinarySource
try:
    ex16_path = os.path.join(mm.__path__[0], '../../examples/example_16')
    if os.path.abspath(ex16_path) not in sys.path:
        sys.path.append(os.path.abspath(ex16_path))
    from ulens_model_fit import UlensModelFit
except ImportError as err:
    print(err)
    print("Please install MulensModel in editable mode (-e) from within the "
          "directory cloned from GitHub. This will allow to import the class"
          " UlensModelFit from example_16.")
    sys.exit(1)
from utils import Utils


class FitBinarySource(UlensModelFit):
    """
    Class for fitting binary source (1L2S) microlensing models using
    *MulensModel* and *EMCEE* package.

    It is a child class of UlensModelFit from example_16, using several of
    its functions to check and parse inputs, setup the EMCEE fitting and
    parse the results. Additional functions are used to read the data, setup
    PSPL fits and run EMCEE with blobs.

    Parameters :
        additional_inputs: *dict*
            Additional inputs compared to UlensModelFit class.

            Currently accepted keys:

            ``'t_peaks'`` - file with peak times for 1L2S fits; if *False*,
            the peak times are taken from the brightest points in the data.
            The file should have columns with the names 'obj_id', 't_peak'
            and 't_peak_2' (see event_finder_modif).

            ``'fix_blend'`` - fix blending flux to given value; if *False*,
            the blending flux is fitted.

            ``'sigmas'`` - list of sigma values that provide starting values
            for EMCEE. The first item is used for PSPL fits and the second
            for the 1L2S fit.

            ``'ans'`` - select which method to get the final solution from
            EMCEE posteriors. If *'max_prob'*, the model with the highest
            probability is used; if *'median'*, the median values are used.

            ``'yaml_files_2L1S'`` - dictionary with information about the
            input files generated for 2L1S fitting. It has the keys 't_or_f',
            'yaml_template' and 'yaml_dir_name'. If 't_or_f' is *True*, the
            2L1S fits are saved in YAML files with the given template and
            directory name.
    """

    def __init__(self, photometry_files, additional_inputs, **kwargs):

        super().__init__(photometry_files, **kwargs)
        self.photometry_files_orig = photometry_files
        self.additional_inputs = additional_inputs

        self.starting_parameters = self._starting_parameters_input.copy()
        self.path = os.path.dirname(os.path.realpath(sys.argv[1]))
        self.t_E_prior = None
        if self._fit_constraints and 'prior' in self._fit_constraints:
            if self._fit_constraints['prior']['t_E'].startswith('lognormal'):
                prior = self._fit_constraints.pop('prior')
                self.t_E_prior = prior['t_E']
        self.read_data()
        self.setup_fitting_parameters()

    def read_data(self):
        """
        Read catalogue(s) and create MulensData instance(s).
        """
        self.phot_fmt = self.photometry_files_orig[0]['phot_fmt']
        phot_settings_aux = self.photometry_files_orig[0].copy()
        cat_path = os.path.join(self.path, phot_settings_aux.pop('file_name'))

        if os.path.isdir(cat_path):
            lst = os.listdir(cat_path)
            fnames = [os.path.join(cat_path, f) for f in lst if f[0] != '.']
        elif os.path.isfile(cat_path):
            fnames = [cat_path]
        else:
            raise RuntimeError(f'Check name of photometry file: {cat_path}.')
        self.file_names = sorted(fnames)

        self.data_list = []
        for fname in self.file_names:
            data = mm.MulensData(file_name=fname, **phot_settings_aux)
            data.bad = np.array(np.isnan(data.mag))
            self.data_list.append(data)
        self._datasets = [self.data_list[0]]

    def setup_fitting_parameters(self):
        """
        Set up the fitting parameters and additional inputs.
        """
        self._parse_fit_constraints()
        self._check_additional_inputs_types()
        self._check_additional_inputs_values()
        self._parse_fitting_parameters_EMCEE()
        self._get_n_walkers()
        self._n_burn = self._fitting_parameters.get('n_burn', 0)

        self._backend = None
        fname = self._other_output.get('models', {}).get('file name', '')
        self._backend_fname = os.path.join(self.path, fname)

    def _check_additional_inputs_types(self):
        """
        Check the types of additional inputs from YAML file.
        """
        dict_types = {'t_peaks': (str, bool), 'fix_blend': (float, bool),
                      'sigmas': (list,), 'ans': (str,),
                      'yaml_files_2L1S': (dict,)}
        msg = "Wrong type of `{}`. Expected: {}. Provided: {}"
        for (key, val) in dict_types.items():
            provided = self.additional_inputs.get(key)
            if not isinstance(provided, (*val, type(None))):
                raise ValueError(msg.format(key, str(val), type(provided)))

        def_ = {'t_or_f': False, 'yaml_template': '', 'yaml_dir_name': ''}
        self.yaml_2L1S = self.additional_inputs.get('yaml_files_2L1S', def_)
        for (key, val) in def_.items():
            provided = self.yaml_2L1S[key]
            if not isinstance(provided, type(val)):
                raise ValueError(msg.format(key, type(val), type(provided)))

    def _check_additional_inputs_values(self):
        """
        Check the values of additional inputs and set to variables.
        """
        self.t_peaks_in = self.additional_inputs.get('t_peaks', False)
        if isinstance(self.t_peaks_in, str):
            self.t_peaks_in = os.path.join(self.path, self.t_peaks_in)
            if not os.path.isfile(self.t_peaks_in):
                raise ValueError('File in t_peaks does not exist.')
        elif self.t_peaks is not False:
            raise ValueError('t_peaks should be a string or False.')

        self.fix_blend_in = self.additional_inputs.get('fix_blend', False)
        def_ = [[0.01, 0.05, 1.], [0.1, 0.01, 0.1, 0.01, 0.1]]
        self.sigmas_emcee = self.additional_inputs.get('sigmas', def_)
        self.ans_emcee = self.additional_inputs.get('ans', 'max_prob')

    def setup_datasets(self, data, name, phot_files):
        """
        Setup the variables for datasets, event_id and photometry files.
        """
        self._datasets = [data]
        self.event_id = name.split('/')[-1].split('.')[0]
        dir_name = phot_files[0]['file_name']
        if dir_name.endswith('.dat'):
            dir_name = os.path.dirname(phot_files[0]['file_name'])

        file_name = self.event_id + '.dat'
        phot_files[0]['file_name'] = os.path.join(dir_name, file_name)
        self._photometry_files = phot_files

    def run_initial_fits(self):
        """
        Run the initial fits: PSPL with scipy.minimize using original and
        subtracted data, then binary source with EMCEE.
        The final fits are done in run_final_fits().

        This function does not accept any parameters. All the settings
        are passed via __init__().
        """
        print(f'\n\033[1m * Running fit for {self.event_id}\033[0m\n')
        self._get_peak_times()
        self._quick_fits_pspl_subtract_pspl()
        start = {'t_0_1': self.pspl_1['t_0'], 'u_0_1': self.pspl_1['u_0'],
                 't_0_2': self.pspl_2['t_0'], 'u_0_2': self.pspl_2['u_0'],
                 't_E': self.t_E_init}
        t_E_scipy = Utils.scipy_minimize_t_E_only(self._datasets[0], start)
        start['t_E'] = t_E_scipy

        self._n_fitting = 0
        self._backend_fname = self._backend_fname.format(self.event_id)
        self._setup_and_run_emcee(self._datasets[0], start)
        self.res_pre_1L2S = self._result.copy()

    def _get_peak_times(self):
        """
        Get the peak times from the input `t_peaks` file.
        """
        if self.t_peaks_in is False:
            self.t_peaks, self.t_peaks_orig = [], []
            return

        tab = Table.read(self.t_peaks_in, format='ascii')
        event = self.event_id.replace('_OGLE', '')
        line = tab[tab['obj_id'] == event]
        self.t_peaks = np.array([line['t_peak'][0], line['t_peak_2'][0]])
        self.t_peaks_orig = self.t_peaks.copy()

    def _quick_fits_pspl_subtract_pspl(self):
        """
        Quick minimization of PSPL models, first with original data and
        then with data subtracted from the first fit.
        Auxiliary functions are all stored in utils.py.
        """
        self.t_E_init = Utils.guess_initial_t_E(self.t_E_prior)
        self.fix_blend = Utils.check_blending_flux(
            self.fix_blend_in, self._datasets[0])
        self.pspl_1, self.t_peaks = Utils.run_scipy_minimize(
            self._datasets[0], self.t_peaks, self.t_E_prior, self.fix_blend)

        subt_data = Utils.subtract_model_from_data(
            self._datasets[0], self.pspl_1, self.fix_blend)
        self.pspl_2, self.t_peaks = Utils.run_scipy_minimize(
            subt_data, self.t_peaks, self.t_E_prior, self.fix_blend)

    def _set_starting_params_emcee(self, dict_start):
        """
        Set starting parameters for the EMCEE fitting.
        The functions in the second block are from UlensModelFit class.
        """
        self._starting_parameters_input = {}
        for idx, (key, val) in enumerate(dict_start.items()):
            line = f"gauss {val} {self._sigma_emcee[idx]}"
            self._starting_parameters_input[key] = line

        self._check_starting_parameters_type()
        self._set_fit_parameters_unsorted()
        self._get_parameters_ordered()

    def _check_bounds_prior(self, theta):
        """
        Check if the parameters are within the bounds and priors. If not,
        it returns -np.inf. The bounds for t_0 should be a list, which sets
        the lower and upper limits around the range of the data.

        The last condition avoids that the guess for t_0_1 is higher than
        t_0_2 (or t_0_2 > t_0_1, depending on initial parameters).
        """
        if not isinstance(self._max_values['t_0'], list):
            raise ValueError('t_0 max_values should be of list type.')
        for (idx, param) in enumerate(self.params_to_fit):
            if param[:3] in self._min_values.keys():
                if theta[idx] < self._min_values[param[:3]]:
                    return -np.inf
            if 'u_0' in param and 'u_0' in self._max_values.keys():
                if theta[idx] > self._max_values['u_0']:
                    return -np.inf
            if 't_0' in param and 't_0' in self._max_values.keys():
                data_time = self._datasets[0].time
                t_range = data_time[::len(data_time)-1]
                t_range = t_range + np.array(self._max_values['t_0'])
                if not t_range[0] < theta[idx] < t_range[1]:
                    return -np.inf

        if self._event.model.n_sources == 2:
            init_t_0_1, init_t_0_2 = self._init_emcee[:3:2]
            if (init_t_0_1 > init_t_0_2) and (theta[0] < theta[2]):
                return -np.inf
            elif (init_t_0_2 > init_t_0_1) and (theta[2] < theta[0]):
                return -np.inf

        return 0.

    def _ln_like(self):
        """
        Get the value of the likelihood function from chi2 of event.

        NOTE: flux_ratio is not needed, but it is commented for future use.
        """
        # event = self._event
        # for (param, theta_) in zip(self.params_to_fit, theta):
        #     # Here we handle fixing source flux ratio:
        #     if param == 'flux_ratio':
        #         # implemented for a single dataset
        #         # event.fix_source_flux_ratio = {my_dataset: theta_}
        #         event.fix_source_flux_ratio = {event.datasets[0]: theta_}
        #     else:
        #         setattr(event.model.parameters, param, theta_)

        chi2 = self._event.chi2

        return -0.5 * chi2

    def _ln_prior(self, theta):
        """
        Apply all the priors (minimum, maximum, distributions), returning
        -np.inf if parameters are outside the bounds.
        The model parameters are set here, after check_bounds. The function
        get_chi2() calculates chi2 (used in _ln_like) and fluxes.
        """
        ln_prior_flux, ln_prior_t_E = 0., 0.
        bounds = self._check_bounds_prior(theta)
        if not np.isfinite(bounds):
            return -np.inf

        self._set_model_parameters(theta)
        self._event.get_chi2()
        self._fluxes_flat = self._get_fluxes()
        ln_prior_flux = self._run_flux_checks_ln_prior(self._fluxes_flat)
        if not np.isfinite(ln_prior_flux):
            return -np.inf

        if self._prior_t_E is not None:
            ln_prior_t_E = self._ln_prior_t_E()
        elif self.t_E_prior is not None:
            # lognormal prior in t_E (check log warnings...)
            # To-Do: Add lognormal in self._get_ln_prior_for_1_parameter()...
            t_E_val = self._model.parameters.t_E
            if 'lognormal' in self.t_E_prior:
                # if t_E >= 1.:
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                mean, sigma = [float(x) for x in self.t_E_prior.split()[1:]]
                ln_prior_t_E = - (np.log(t_E_val) - mean)**2 / (2*sigma**2)
                ln_prior_t_E -= np.log(t_E_val * np.sqrt(2*np.pi)*sigma)
            else:
                raise ValueError('t_E prior type not allowed.')

        return ln_prior_t_E + ln_prior_flux

    def _ln_prob(self, theta):
        """
        Combines likelihood value and priors for a given set of parameters.
        The argument theta is an array with the chain parameters to sample
        the likelihood + prior.
        Returns the logarithm of the probability and fluxes.
        """
        ln_prior = self._ln_prior(theta)
        if not np.isfinite(ln_prior):
            return self._return_inf
        ln_like = self._ln_like()
        if not np.isfinite(ln_like):
            return self._return_inf

        ln_prob = ln_prior + ln_like
        self._update_best_model_EMCEE(ln_prob, theta, self._fluxes_flat)
        source_fluxes, blending_flux = self._event.fluxes[0]

        return ln_prob, source_fluxes, blending_flux

    def _setup_and_run_emcee(self, data, dict_start):
        """
        Setup and run EMCEE, for given data and starting parameters.
        """
        self._setup_fit_pspl_binary(data, dict_start)
        self._run_emcee()
        self._make_burn_in_and_reshape()
        self._print_emcee_percentiles()
        self._get_best_params_emcee()

    def _setup_fit_pspl_binary(self, data, dict_start):
        """
        Setup the EMCEE fit for PSPL or binary source models.

        Settings such as the names and number of the fitted parameters,
        number of burn-in, and mean/sigmas of the starting values for the
        parameters are set here. An instance of _model and _event are also
        declared in order to access parameters and chi2.
        """
        self.params_to_fit = list(dict_start.keys())
        self._n_fit_parameters = len(self.params_to_fit)
        self._init_emcee = list(dict_start.values())

        if self._n_fit_parameters == 3:
            self._sigma_emcee = self.sigmas_emcee[0]
            fix = Utils.check_blending_flux(self.fix_blend_in, data)
            self._return_inf = -np.inf, np.array([-np.inf]), -np.inf
            self._blobs_dtype = np.dtype([('source_fluxes', np.float64, (1,)),
                                          ('blend_fluxes', np.float64)])
        elif self._n_fit_parameters == 5:
            self._sigma_emcee = self.sigmas_emcee[1]
            fix = None
            self._return_inf = -np.inf, np.array([-np.inf, -np.inf]), -np.inf
            self._blobs_dtype = np.dtype([('source_fluxes', np.float64, (2,)),
                                          ('blend_fluxes', np.float64)])
        else:
            raise ValueError("Number of fitting parameters should be 3 or 5.")

        self._set_starting_params_emcee(dict_start)
        self._model = mm.Model(dict_start)
        self._event = mm.Event(data, model=self._model, fix_blend_flux=fix)

    def _run_emcee(self):
        """
        Get initial state for walkers and run EMCEE sampler.

        NOTE: Multiprocessing was not effective, specially in Linux.
        """
        self._set_n_fluxes()
        rand_sample = np.random.randn(self._n_walkers, self._n_fit_parameters)
        self._rand_sample = rand_sample * self._sigma_emcee
        if self._n_fit_parameters == 5:
            self._run_quick_emcee()
        start = abs(np.array(self._init_emcee) + self._rand_sample)
        self._kwargs_EMCEE['initial_state'] = start

        if self._n_fitting == 4 and self._backend_fname.endswith('.h5'):
            self._backend = emcee.backends.HDFBackend(self._backend_fname)
            self._backend.reset(self._n_walkers, self._n_fit_parameters)

        self._sampler = emcee.EnsembleSampler(
            self._n_walkers, self._n_fit_parameters, self._ln_prob,
            backend=self._backend, blobs_dtype=self._blobs_dtype)
        self._sampler.run_mcmc(**self._kwargs_EMCEE)

    def _run_quick_emcee(self):
        """
        Quick EMCEE fitting, executed only for binary source, in order to
        get more definite starting values for the walkers.
        """
        start = abs(np.array(self._init_emcee) + 10*self._rand_sample)
        self._kwargs_EMCEE['initial_state'] = start

        sampler = emcee.EnsembleSampler(
            self._n_walkers, self._n_fit_parameters, self._ln_prob)
        nsteps_temp = int(self._kwargs_EMCEE['nsteps'] / 2)
        kwargs_temp = {**self._kwargs_EMCEE, 'nsteps': nsteps_temp}
        sampler.run_mcmc(**kwargs_temp)

        samples = sampler.chain[:, int(self._n_burn/2):, :]
        samples = samples.reshape((-1, self._n_fit_parameters))
        self._init_emcee = np.percentile(samples, 50, axis=0)

    def _make_burn_in_and_reshape(self):
        """
        Remove burn-in samples, reshape with the fluxes and probabilities.
        """
        samples = self._sampler.chain[:, self._n_burn:, :]
        samples = samples.reshape((-1, self._n_fit_parameters))
        blobs = self._sampler.get_blobs()[self._n_burn:]
        blobs = blobs.T.flatten()

        source_fluxes = np.array(list(chain.from_iterable(blobs))[::2])
        blend_flux = np.array(list(chain.from_iterable(blobs))[1::2])
        prob = self._sampler.lnprobability[:, self._n_burn:].ravel()
        arrays_to_stack = (samples, source_fluxes, blend_flux, prob)
        self._samples = np.column_stack(arrays_to_stack)

    def _print_emcee_percentiles(self):
        """
        Print the percentiles of the EMCEE samples.
        The fluxes names are added to the list params_to_fit here.
        """
        self._perc = np.percentile(self._samples, [16, 50, 84], axis=0)
        self.params_to_fit += self._get_fluxes_names_to_print()
        self._pars_perc = dict(zip(self.params_to_fit, self._perc.T))

        prints = {0: 'pre-fit, 1L2S to original data',
                  1: '1st fit, PSPL to split data',
                  2: '2nd fit, PSPL to split/subtracted data',
                  3: '1st fit, PSPL to split/subtracted data',
                  4: 'final fit, 1L2S to original data'}
        print(f"Fitted parameters ({prints[self._n_fitting]}):")
        self._n_fitting += 1

        for i in range(self._n_fit_parameters):
            r = self._perc[1, i]
            msg = self.params_to_fit[i] + ": {:.5f} +{:.5f} -{:.5f}"
            print(msg.format(r, self._perc[2, i]-r, r-self._perc[0, i]))

    def _get_best_params_emcee(self):
        """
        Get the best parameters from the EMCEE samples.
        """
        best_idx = np.argmax(self._samples[:, -1])
        pars_best = self._samples[best_idx, :-1]
        self._set_model_parameters(pars_best)

        print("Smallest chi2 model:")
        print(*[repr(b) if isinstance(b, float) else b for b in pars_best])
        n_dof = self._event.datasets[0].n_epochs - self._n_fit_parameters
        self._result_chi2 = self._event.get_chi2()
        print(f"chi2 = {self._result_chi2:.8f}, dof = {n_dof}\n")

        if self.ans_emcee == 'max_prob':
            self._result = dict(zip(self.params_to_fit, pars_best))
        elif self.ans_emcee == 'median':
            self._result = dict(zip(self.params_to_fit, self._perc[1]))

    def run_final_fits(self):
        """
        Split data and fit PSPL twice with EMCEE, then get final 1L2S fit.
        """
        if not hasattr(self, 't_peaks_orig'):
            t_peaks = [self.res_pre_1L2S['t_0_1'], self.res_pre_1L2S['t_0_2']]
            self.t_peaks_orig = np.array(t_peaks) - 2.45e6
        pre_ev = Utils.get_mm_event(self._datasets[0], self.res_pre_1L2S)[0]

        model_between_peaks = Utils.get_model_pts_between_peaks(
            pre_ev, self.t_peaks_orig)
        self._time_min_flux = Utils.detect_min_flux_in_model(
            self._datasets[0], self.t_peaks_orig, model_between_peaks,
            self.t_E_prior, self.fix_blend_in)
        self._data_left_right = Utils.split_in_min_flux(
            self._datasets[0], self._time_min_flux)

        self._fit_pspl_twice()
        pspl_1, pspl_2 = self.res_pspl_1[0], self.res_pspl_2[0]
        start = {'t_0_1': pspl_1['t_0'], 'u_0_1': pspl_1['u_0'],
                 't_0_2': pspl_2['t_0'], 'u_0_2': pspl_2['u_0'],
                 't_E': self.t_E_init}
        t_E_scipy = Utils.scipy_minimize_t_E_only(self._datasets[0], start)
        start['t_E'] = t_E_scipy

        self._setup_and_run_emcee(self._datasets[0], start)
        self.res_1L2S = [self._result.copy(), self._sampler, self._samples,
                         self._pars_perc]
        self.ev_1L2S = Utils.get_mm_event(
            self._datasets[0], self.res_1L2S[0])[0]

    def _fit_pspl_twice(self):
        """
        Fit PSPL to split data, then subtract and apply the PSPL fit to
        the other part of the data twice.
        """
        data_1, data_2 = self._data_left_right
        start = Utils.guess_pspl_params(
            self._datasets[0], t_E_prior=self.t_E_prior)[0]
        if not data_1.time.min() <= start['t_0'] <= data_1.time.max():
            data_1, data_2 = data_2, data_1

        # 1st PSPL (data_left or brighter)
        self._setup_and_run_emcee(data_1, start)
        res_pspl_1 = self._result.copy()
        model_1 = mm.Model(dict(list(res_pspl_1.items())[:3]))

        # ADD LATER ::: check if there is data > 3sigma above f_base...

        # 2nd PSPL (not to original data_2, but to data_2_subt)
        fix_2 = Utils.check_blending_flux(self.fix_blend_in, data_2)
        data_2_subt = Utils.subtract_model_from_data(data_2, model_1, fix_2)
        start = Utils.guess_pspl_params(data_2_subt, None, self.t_E_prior)[0]
        self._setup_and_run_emcee(data_2_subt, start)
        model_2 = mm.Model(dict(list(self._result.items())[:3]))
        self.res_pspl_2 = [self._result.copy(), self._sampler, self._samples,
                           self._pars_perc, data_2_subt]

        # 3rd PSPL (to data_1_subt)
        fix_1 = Utils.check_blending_flux(self.fix_blend_in, data_1)
        data_1_subt = Utils.subtract_model_from_data(data_1, model_2, fix_1)
        start = model_1.parameters.as_dict()
        start['t_E'] = model_1.parameters.t_E
        self._setup_and_run_emcee(data_1_subt, start)
        model_1 = mm.Model(dict(list(self._result.items())[:3]))
        self.res_pspl_1 = [self._result.copy(), self._sampler, self._samples,
                           self._pars_perc, data_1_subt]


if __name__ == '__main__':

    np.random.seed(12343)
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')

    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as data:
        settings = yaml.safe_load(data)
    stg_copy = copy.deepcopy(settings)
    fit_binary_source = FitBinarySource(**settings)

    phot_files = stg_copy.pop('photometry_files')
    del_keys = ['starting_parameters', 'min_values', 'max_values']
    stg_copy = {k: v for k, v in stg_copy.items() if k not in del_keys}
    dlist, fnames = fit_binary_source.data_list, fit_binary_source.file_names

    for data, name in zip(dlist, fnames):
        fit_binary_source.setup_datasets(data, name, phot_files)
        fit_binary_source.run_initial_fits()
        try:
            fit_binary_source.run_final_fits()
        except ValueError:
            print("The fit did not converge, skipping...")
            print("\n--------------------------------------------------")
            continue

        kwargs = copy.deepcopy(stg_copy)
        kwargs.update({'datasets': [data],
                       'event_id': fit_binary_source.event_id,
                       'res_pspl_1': fit_binary_source.res_pspl_1,
                       'res_pspl_2': fit_binary_source.res_pspl_2,
                       'res_1l2s': fit_binary_source.res_1L2S,
                       'time_min_flux': fit_binary_source._time_min_flux})
        save_results = SaveResultsBinarySource(phot_files, **kwargs)
        print("\n--------------------------------------------------")
