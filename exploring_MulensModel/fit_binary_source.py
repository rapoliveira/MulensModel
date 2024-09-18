"""
Class and script for fitting binary source model using MulensModel.
All settings are read from a YAML file.
"""
from itertools import chain
import os
import sys
import warnings

from astropy.table import Table, Column
import corner
import emcee
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# import multiprocessing
import numpy as np
import scipy.optimize as op
import yaml

import MulensModel as mm
try:
    # :: Add after UltraNest PR is merged ::
    # ex16_path = os.path.join(mm.__path__[0], '../../examples/example_16')
    # sys.path.append(os.path.abspath(ex16_path))
    from ulens_model_fit import UlensModelFit
    import split_data_for_binary_lens as split
except ImportError as err:
    print(err)
    print("Please install MulensModel in editable mode from within the"
          "directory cloned from GitHub. This will allow to import the class"
          "UlensModelFit from example_16.")
    sys.exit(1)
from utils import Utils


class FitBinarySource(UlensModelFit):
    """
    Class for fitting binary source (1L2S) microlensing models using
    *MulensModel* and *EMCEE* package.

    It is a child class of UlensModelFit from example_16, using several of
    its functions to check and parse inputs, setup the EMCEE fitting and
    parse the results. Additional functions (child methods) are used to
    read the data, setup PSPL fits and run EMCEE with blobs.

    Parameters :
        additional_inputs: *dict*
            Additional inputs compared to UlensModelFit class.

            Currently accepted keys:

            ``'t_peaks'`` - file with peak times for 1L2S fits; if *False*,
            the peak times are taken from the brightest points in the data.
            The file should have columns with the names 'obj_id', 't_peak'
            and 't_peak_2' (see event_finder_modif).

            ``'fix_blend'`` - fix blending flux to a given value; if *False*,
            the blending flux is fitted.

            ``'sigmas'`` - list of sigma values used to find starting values
            for EMCEE. The first value is used for the first PSPL fit and the
            second for the 1L2S fit.

            ``'ans'`` - select which method to get the final solution from
            the EMCEE posteriors. If *'max_prob'*, the model with the highest
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
        self._check_fit_constraints()
        self._parse_fit_constraints_keys()
        self.read_data()

        for data, name in zip(self.data_list, self.file_names):
            self._datasets = [data]
            self.event_id = name.split('/')[-1].split('.')[0]
            self.photometry_files = self.photometry_files_orig.copy()
            self.photometry_files[0]['file_name'] += self.event_id + '.dat'
            self.setup_fitting_parameters()
            self.run_initial_fits()
            self.run_final_fits()

    def read_data(self):
        """
        Read catalogue or list of catalogues and creates MulensData instance.
        """
        self.phot_fmt = self.photometry_files_orig[0]['phot_fmt']
        phot_settings_aux = self.photometry_files_orig[0].copy()
        dir_file = os.path.join(self.path, phot_settings_aux.pop('file_name'))

        if os.path.isdir(dir_file):
            fnames = [f for f in os.listdir(dir_file) if not f.startswith('.')]
            fnames = [os.path.join(dir_file, f) for f in fnames]
        elif os.path.isfile(dir_file):
            fnames = [dir_file]
        else:
            raise RuntimeError(f'Photometry file(s) {fnames} not available.')
        self.file_names = sorted(fnames)

        self.data_list = []
        for fname in self.file_names:
            data = mm.MulensData(file_name=fname, **phot_settings_aux)
            self.data_list.append(data)

    def setup_fitting_parameters(self):
        """
        Set up the fitting parameters and additional inputs.
        """
        self._check_additional_inputs()
        self.t_peaks, self.t_peaks_orig = [], []
        t_peaks_in = self.additional_inputs.get('t_peaks', False)
        if isinstance(t_peaks_in, str):
            self._get_peak_times()
        elif t_peaks_in is not False:
            raise ValueError('t_peaks should be a string or False.')

        self._parse_fitting_parameters_EMCEE()
        self._get_n_walkers()
        self._prior = self._fit_constraints['prior']

    def _check_additional_inputs(self):
        """
        Check the types of additional inputs and set the values.
        """
        dict_types = {'t_peaks': (str, bool), 'fix_blend': (float, bool),
                      'sigmas': (list,), 'ans': (str,),
                      'yaml_files_2L1S': (dict,)}
        msg = "Wrong type of `{}`. Expected: {}. Provided: {}"
        for (key, val) in dict_types.items():
            provided = self.additional_inputs.get(key)
            if not isinstance(provided, (*val, type(None))):
                raise ValueError(msg.format(key, str(val), type(provided)))

        self.fix_blend_in = self.additional_inputs.get('fix_blend', False)
        fix_dict = {self._datasets[0]: self.fix_blend_in}
        self.fix_blend = None if self.fix_blend_in is False else fix_dict
        def_ = [[0.01, 0.05, 1.], [0.1, 0.01, 0.1, 0.01, 0.1]]
        self.sigmas_emcee = self.additional_inputs.get('sigmas', def_)
        self.ans_emcee = self.additional_inputs.get('ans', 'max_prob')

        def_ = {'t_or_f': False, 'yaml_template': '', 'yaml_dir_name': ''}
        self.yaml_2L1S = self.additional_inputs.get('yaml_files_2L1S', def_)
        for (key, val) in def_.items():
            provided = self.yaml_2L1S[key]
            if not isinstance(provided, type(val)):
                raise ValueError(msg.format(key, type(val), type(provided)))

    def _get_peak_times(self):
        """
        Get the peak times from the input file.
        """
        tab_file = self.additional_inputs['t_peaks']
        tab = Table.read(os.path.join(self.path, tab_file), format='ascii')

        event = self.event_id.replace('_OGLE', '').replace('_', '.')
        line = tab[tab['obj_id'] == event]
        self.t_peaks = np.array([line['t_peak'][0], line['t_peak_2'][0]])
        self.t_peaks_orig = self.t_peaks.copy()

    def run_initial_fits(self):
        """
        Run the initial fits: PSPL to original data and subtracted data
        with scipy.minimize, then binary source with EMCEE.
        The final fits will be done in a separate function run_fits()...

        This function does not accept any parameters. All the settings
        are passed via __init__().
        """
        print(f'\n\033[1m * Running fit for {self.event_id}\033[0m')
        self._quick_fits_pspl_subtract_pspl()
        self.binary_source_start = {'t_0_1': self.quick_pspl_1['t_0'],
                                    'u_0_1': self.quick_pspl_1['u_0'],
                                    't_0_2': self.quick_pspl_2['t_0'],
                                    'u_0_2': self.quick_pspl_2['u_0'],
                                    't_E': self.t_E_init}

        t_E_scipy = Utils.scipy_minimize_t_E_only(self._datasets[0],
                                                  self.binary_source_start)
        self.binary_source_start['t_E'] = t_E_scipy
        model_start = mm.Model(self.binary_source_start)
        self.ev_st = mm.Event(self._datasets[0], model=model_start)

        self._setup_fit_emcee_binary()
        self._run_emcee()
        self._make_burn_in_and_reshape()
        self._print_emcee_percentiles()
        self._get_best_params_emcee()
        self.res_pre_1L2S = self._result.copy()
        self.ev_pre_1L2S = self._get_binary_source_event(self.res_pre_1L2S)

    def _quick_fits_pspl_subtract_pspl(self):
        """
        First step: quick estimate of PSPL models using scipy.minimize.
        Two fits are carried out: with original data and then with data
        subtracted from the first fit.
        These auxiliary functions are all stored in utils.py.
        """
        self.t_E_prior = self._fit_constraints['prior'].get('t_E')
        self.t_E_init = Utils.guess_initial_t_E(self.t_E_prior)
        self.quick_pspl_1, self.t_peaks = Utils.run_scipy_minimize(
            self._datasets[0], self.t_peaks, self.t_E_prior, self.fix_blend)

        subt_data = Utils.subtract_model_from_data(
            self._datasets[0], self.quick_pspl_1, self.fix_blend)
        self.quick_pspl_2, self.t_peaks = Utils.run_scipy_minimize(
            subt_data, self.t_peaks, self.t_E_prior, self.fix_blend)

    def _set_starting_params_emcee(self, dict_start):
        """
        Write docs later... ONLY AFTER STH!
        """
        self._starting_parameters_input = {}
        for idx, (key, val) in enumerate(dict_start.items()):
            line = f"gauss {val} {self._sigma_emcee[idx]}"
            self._starting_parameters_input[key] = line

        self._check_starting_parameters_type()
        self._set_fit_parameters_unsorted()
        self._get_parameters_ordered()
        self._parse_fit_constraints_fluxes()

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

    def _ln_like(self, theta):
        """
        Get the value of the likelihood function from chi2 of event.

        NOTE: if flux_ratio is not needed, use _ln_like from example_16
        """
        # *** DISCUSS TOMORROW IF FLUX RATIO WILL EVER BE USED ***
        # event = self._event
        # for (param, theta_) in zip(self.params_to_fit, theta):
        #     # Here we handle fixing source flux ratio:
        #     if param == 'flux_ratio':
        #         # implemented for a single dataset
        #         # event.fix_source_flux_ratio = {my_dataset: theta_} # original(?)
        #         event.fix_source_flux_ratio = {event.datasets[0]: theta_}
        #     else:
        #         setattr(event.model.parameters, param, theta_)

        self._set_model_parameters(theta)
        chi2 = self._event.get_chi2()

        return -0.5 * chi2

    def _ln_prior(self, theta):
        """
        Apply all the priors (minimum, maximum, distributions), returning
        -np.inf if parameters are outside the bounds.
        """
        bounds = self._check_bounds_prior(theta)
        fluxes = self._get_fluxes()
        ln_prior_flux = self._run_flux_checks_ln_prior(fluxes)
        if not np.isfinite(bounds + ln_prior_flux):
            return -np.inf

        # Prior in t_E (only lognormal so far, tbd: Mroz17/20)
        # OBS: Still need to remove the ignore warnings line (related to log?)
        # To-Do: Add lognormal in self._get_ln_prior_for_1_parameter()...
        ln_prior_t_E, ln_prior_flux = 0., 0.
        if 't_E' in self._prior.keys():
            t_E_prior = self._prior['t_E']
            t_E_val = theta[self.params_to_fit.index('t_E')]
            if 'lognormal' in t_E_prior:
                # if t_E >= 1.:
                warnings.filterwarnings("ignore", category=RuntimeWarning)  # bad!
                prior = [float(x) for x in t_E_prior.split()[1:]]
                ln_prior_t_E = - (np.log(t_E_val) - prior[0])**2 / (2*prior[1]**2)
                ln_prior_t_E -= np.log(t_E_val * np.sqrt(2*np.pi)*prior[1])
            elif 'Mroz et al.' in self._prior['t_E']:
                raise ValueError('Still implementing Mroz et al. priors.')
            else:
                raise ValueError('t_E prior type not allowed.')

        return 0.0 + ln_prior_t_E + ln_prior_flux

    def _ln_prob(self, theta):
        """
        Combines likelihood value and priors for a given set of parameters.
        The parameter theta is an array with the chain parameters to sample
        the likelihood + prior.
        Returns the logarithm of the probability and fluxes.
        """

        ln_like = self._ln_like(theta)
        ln_prior = self._ln_prior(theta)
        if not np.isfinite(ln_prior):
            return -np.inf, np.array([-np.inf, -np.inf]), -np.inf

        # In the cases that source fluxes are negative we want to return
        # these as if they were not in priors.
        if np.isnan(ln_like):
            return -np.inf, np.array([-np.inf, -np.inf]), -np.inf

        ln_prob = ln_prior + ln_like
        fluxes = self._get_fluxes()
        self._update_best_model_EMCEE(ln_prob, theta, fluxes)
        source_fluxes, blending_flux = self._event.fluxes[0]

        return ln_prob, source_fluxes, blending_flux

    def _setup_fit_emcee_binary(self):
        """
        Setup EMCEE fit for binary source model...
        # sigmas (list): sigma values used to find starting values.
        """
        self.params_to_fit = list(self.binary_source_start.keys())
        self._n_fit_parameters = len(self.params_to_fit)
        self._sigma_emcee = self.sigmas_emcee[1]
        self._n_burn = self._fitting_parameters.get('n_burn', 0)
        self._set_starting_params_emcee(self.binary_source_start)
        self._init_emcee = list(self.binary_source_start.values())

        pre_str = '3rd ' if hasattr(self, '_sampler') else 'Pre-'
        print(f'\n\033[1m -- {pre_str}fit: 1L2S to original data...\033[0m')

    def _run_emcee(self):
        """
        Setup and run EMCEE...
        TO-DO :::
        - Add backend=backend to sampler later!!!
        - Try multiprocessing once more...
        """
        self._event = self.ev_st
        self._model = self._event.model
        self._set_n_fluxes()

        rand_sample = np.random.randn(self._n_walkers, self._n_fit_parameters)
        self._rand_sample = rand_sample * self._sigma_emcee
        if self._n_fit_parameters == 5:
            self._run_quick_emcee()
        start = abs(np.array(self._init_emcee) + self._rand_sample)
        self._kwargs_EMCEE['initial_state'] = start

        blobs = [('source_fluxes', list), ('blend_fluxes', float)]
        self._sampler = emcee.EnsembleSampler(
            self._n_walkers, self._n_fit_parameters, self._ln_prob,
            blobs_dtype=blobs)
        self._sampler.run_mcmc(**self._kwargs_EMCEE)

        # Setting up multi-threading (std: fork in Linux, spawn in Mac)
        # multiprocessing.set_start_method("fork", force=True)
        # os.environ["OMP_NUM_THREADS"] = "1"
        # with multiprocessing.Pool() as pool:
        #     sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, pool=pool,
        #                                     args=(event, params_to_fit, spec))
        #     sampler.run_mcmc(start, nstep, progress=n_emcee['progress'])
        # pool.close()

    def _run_quick_emcee(self):
        """
        Write here... Only if 1L2S fit...
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
        """
        self._perc = np.percentile(self._samples, [16, 50, 84], axis=0)
        print("Fitted parameters:")
        for i in range(self._n_fit_parameters):
            r = self._perc[1, i]
            msg = self.params_to_fit[i] + ": {:.5f} +{:.5f} -{:.5f}"
            print(msg.format(r, self._perc[2, i]-r, r-self._perc[0, i]))

    def _get_best_params_emcee(self):
        """
        Get the best parameters from the EMCEE samples.

        NOTE: if flux_ratio is needed, recover lines from _ln_prob()
        """
        best_idx = np.argmax(self._samples[:, -1])
        pars_best = self._samples[best_idx, :-1]
        self._set_model_parameters(pars_best)
        print("\nSmallest chi2 model:")
        print(*[repr(b) if isinstance(b, float) else b for b in pars_best])
        n_dof = self._event.datasets[0].n_epochs - self._n_fit_parameters
        self._result_chi2 = self._event.get_chi2()
        print(f"chi2 = {self._result_chi2:.8f}, dof = {n_dof}")

        # Adding flux names to params_to_fit
        # if self._samples.shape[1] - 1 == self._n_fit_parameters + 2:
        #     self.params_to_fit += ['source_flux', 'blending_flux']
        # elif self._samples.shape[1] - 1 == self._n_fit_parameters + 3:
        #     self.params_to_fit += ['source_flux_1', 'source_flux_2',
        #                            'blending_flux']
        self.params_to_fit += self._get_fluxes_names_to_print()

        if self.ans_emcee == 'max_prob':
            self._result = dict(zip(self.params_to_fit, pars_best))
        elif self.ans_emcee == 'median':
            self._result = dict(zip(self.params_to_fit, self._perc[1]))

    def _get_binary_source_event(self, best):

        data = self._datasets[0]
        bst = dict(b_ for b_ in list(best.items()) if 'flux' not in b_[0])
        fix_source = {data: [best[p] for p in best if 'flux_s' in p]}
        event_1L2S = mm.Event(data, model=mm.Model(bst),
                              fix_source_flux=fix_source,
                              fix_blend_flux={data: best['flux_b_1']})
        event_1L2S.get_chi2()

        return event_1L2S

    def run_final_fits(self):
        """
        _summary_
        # *** Add all the fitting routine here... calling short functions!
        # 2) EMCEE to get a first estimate for 1L2S -- OK, solved!
        # 3) Split data and fit PSPL twice with EMCEE -- OK, check...
        # 4) Get final 1L2S fit...
        """
        if not hasattr(self, 't_peaks_orig'):
            t_peaks = [self.res_pre_1L2S['t_0_1'], self.res_pre_1L2S['t_0_2']]
            self.t_peaks_orig = np.array(t_peaks) - 2.45e6
        model_between_peaks = Utils.get_model_pts_between_peaks(
            self.ev_pre_1L2S, self.t_peaks_orig)
        time_min_flux = Utils.detect_min_flux_in_model(
            self._datasets[0], self.t_peaks_orig, model_between_peaks,
            self.t_E_prior, self.fix_blend)
        self._data_left_right = Utils.split_in_min_flux(
            self._datasets[0], time_min_flux)

        self.start_two_pspl = Utils.guess_pspl_params(
            self._datasets[0], t_E_prior=self.t_E_prior)[0]
        self._fit_pspl_twice()
        # two_pspl, subt_data = Utils.fit_PSPL_twice(data_left_right, settings, start=start_two_pspl)


        # OLD CODE :::
        # start = get_initial_t0_u0(data, settings)[0]
        # two_pspl, subt_data = split.fit_PSPL_twice(data_left_right, settings,
        #                                         start=start)
        # output_1, output_2 = two_pspl  # , output_3, data_left_right[0]
        #     breakpoint()

        # self._run_emcee_1L2S()
        # self._make_burn_in_and_reshape()
        # self._print_emcee_percentiles()
        # self._get_best_params_emcee()
        # self.ev_1L2S = self._get_binary_source_event(self._result)

        print("\n--------------------------------------------------")

    def _fit_pspl_twice(self):
        """
        Fit PSPL to data_left and another PSPL to the right subtracted data.
        Adapt later...

        data_left_right (tuple): two mm.Data instances (left and right)
        settings (dict): all settings from yaml file
        """

        # def fit_PSPL_twice(data_left_right, settings, result=[], start={}):
        """
        Fit PSPL to data_left and another PSPL to the right subtracted data.

        Args:
            result (tuple): all the results derived in binary source fit
            data_left_right (tuple): two mm.Data instances (left and right)
            settings (dict): all settings from yaml file

        Returns:
            tuple: two PSPL dictionaries with the result parameters
        """

        # 1st PSPL (data_left or brighter)
        data_1, data_2 = self._data_left_right
        # start = self.start_two_pspl
        n_emcee = self._fitting_parameters
        settings['123_fits'] = '1st fit'

        self.params_to_fit = list(self.start_two_pspl.keys())
        self._n_fit_parameters = len(self.params_to_fit)
        self._sigma_emcee = self.sigmas_emcee[0]
        self._set_starting_params_emcee(self.start_two_pspl)
        self._init_emcee = list(self.start_two_pspl.values())

        time_min, time_max = data_1.time.min(), data_1.time.max()
        if not time_min <= self.start_two_pspl['t_0'] <= time_max:
            data_1, data_2 = data_2, data_1  # e.g. BLG611.09.12112
        fix_dict = {data_1: self.fix_blend_in}
        fix_1 = None if self.fix_blend_in is False else fix_dict
        model_start = mm.Model(self.start_two_pspl)
        self.ev_st = mm.Event(data_1, model=model_start, fix_blend_flux=fix_1)

        print('\n\033[1m -- 1st fit: PSPL to original data...\033[0m')

        self._run_emcee()
        self._make_burn_in_and_reshape()
        self._print_emcee_percentiles()
        self._get_best_params_emcee()
        self.res_pspl_1 = self._result.copy()
        model_1 = mm.Model(dict(list(self.res_pspl_1.items())[:3]))
        breakpoint()

        # UP TO THIS POINT :::
        # The code copied from split.fit_split_twice is working well, just
        # as in the previous code. I will dedicate some time to clean and
        # shorten it.
        #
        # In the following lines, the data will be subtracted and two other
        # PSPL fits will be implemented:
        # - Subtract data_2 from the first fit
        # - ADD STEP OF CHECKING IF THERE IS DATA > 3sigma above f_base...?
        # - 2nd PSPL (not to original data_2, but to data_2_subt)
        # - Subtract data_1 from the second fit
        # - 3rd PSPL (to data_1_subt)
        # - Quick plot to check fits? Remove it...
        # - Add return statements
        #
        # OBS: Add a function to call all the repeated lines, with the data
        # and starting parameters as arguments!

    def _parse_results(self):
        """
        Still work on it after adding my functions...
        """
        self._parse_other_output_parameters()
        # value = self._other_output["models"]
        # self._parse_other_output_parameters_models(value)


def prefit_split_and_fit(data, settings, pdf=""):
    """
    NEW function for fitting PSPL-split, then 1L2S if there is a second peak.

    Args:
        data (mm.MulensData): data instance of a single event.
        settings (dict): all input settings from yaml file.
        pdf (str, optional): pdf file to save the plot. Defaults to "".

    Returns:
        tuple: emcee output, mm.Event, two PSPL results and corner plot.
    """

    # Split data before 1L2S fit, too specific (02-06/nov)
    # start, fm_base = get_initial_t0_u0(data, settings)
    # data_left_right = split.split_before_result(data, fm_base[0], fm_base[1])
    # two_pspl = split.fit_PSPL_twice(data_left_right, settings, start=start)

    # Radek's suggestion: scipy_minimize (06/nov-[...])
    # if settings['starting_parameters']['t_peaks'] is not False:
    #     settings = fit_utils('get_peak_times', data, settings)
    # model_1 = fit_utils('scipy_minimize', data, settings)
    # data_2_subt = fit_utils('subt_data', data, settings, model_1)
    # # fm_base = get_initial_t0_u0(data_2_subt, settings)[1]  # above 3sigma?
    # # if no data above 3sigma:
    # #   - run MCMC and return...
    # # else:
    # model_2 = fit_utils('scipy_minimize', data_2_subt, settings)
    # #
    # n_emcee = settings['fitting_parameters']
    # start = {'t_0_1': model_1['t_0'], 'u_0_1': model_1['u_0'], 't_0_2':
    #          model_2['t_0'], 'u_0_2': model_2['u_0'], 't_E': 25}
    # t_E_optimal = fit_utils('get_t_E_1L2S', data, settings, start)
    # start['t_E'] = round(t_E_optimal[4], 2)
    # ev_st = mm.Event(data, model=mm.Model(start))
    # output = fit_emcee(start, n_emcee['sigmas'][1], ln_prob, ev_st, settings)
    # event_1L2S = fit_utils('get_1L2S_event', data, settings, best=output[0])
    # data_left_right, t_min = split.split_after_result(event_1L2S, output, settings)

    # Fits 2xPSPL if data is good (u_0 < 3.)...
    start = get_initial_t0_u0(data, settings)[0]
    two_pspl, subt_data = split.fit_PSPL_twice(data_left_right, settings,
                                               start=start)
    output_1, output_2 = two_pspl  # , output_3, data_left_right[0]
    settings['xlim'] = get_xlim2(output_1[0], data, n_emcee, t_min)
    make_plots(output_1[:-1], subt_data[0], settings, data, pdf=pdf)
    make_plots(output_2[:-1], subt_data[1], settings, data, pdf=pdf)
    # make_plots(output_3[:-1], subt_data[0], settings, data, pdf=pdf)

    # Make 1L2S final fit and plot
    settings['123_fits'] = '3rd fit'
    start = {'t_0_1': output_1[0]['t_0'], 'u_0_1': output_1[0]['u_0'], 't_0_2':
             output_2[0]['t_0'], 'u_0_2': output_2[0]['u_0'], 't_E': 25}
    ev_st = mm.Event(data, model=mm.Model(start))
    output = fit_emcee(start, n_emcee['sigmas'][1], ln_prob, ev_st, settings)
    event_1L2S, cplot_1L2S = make_plots(output[:-1], data, settings, pdf=pdf)

    return output + (event_1L2S, (output_1[0], output_2[0])), cplot_1L2S

def make_plots(results_states, data, settings, orig_data=None, pdf=""):
    """
    Make three plots: tracer plot, corner plot and best model.

    Args:
        results_states (tuple): contains best results, sampler and states.
        data (mm.MulensData): data instance of a single event.
        settings (dict): all input settings from yaml file.
        orig_data (list, optional): Plot with original data. Defaults to None.
        pdf (str, optional): pdf file to save the plot. Defaults to "".

    Returns:
        tuple: mm.Event and corner plot instances, to be used later.
    """

    best, sampler, states = results_states
    n_emcee = settings['fitting_parameters']
    condition = (n_emcee['fix_blend'] is not False) and (len(best) != 8)
    c_states = states[:, :-2] if condition else states[:, :-1]
    params = list(best.keys())[:-1] if condition else list(best.keys())
    values = list(best.values())[:-1] if condition else list(best.values())
    tracer_plot(params, sampler, n_emcee['nburn'], pdf=pdf)
    if len(best) == 8:
        c_states, params, values = c_states[:, :-3], params[:5], values[:5]
    cplot = corner.corner(c_states, quantiles=[0.16, 0.50, 0.84],
                          labels=params, truths=values, show_titles=True)
    if pdf:
        pdf.savefig(cplot)
    else:
        plt.show()
    event = plot_fit(best, data, settings, orig_data, pdf=pdf)

    return event, cplot


def tracer_plot(params_to_fit, sampler, nburn, pdf=""):
    """
    Plot tracer plots (or time series) of the walkers.

    Args:
        params_to_fit (list): name of the parameters to be fitted.
        sampler (emcee.EnsembleSampler): sampler that contain the chains.
        (int): number of steps considered as burn-in (< n_steps).
        pdf (str, optional): pdf file to save the plot. Defaults to "".
    """

    npars = sampler.ndim
    fig, axes = plt.subplots(npars, 1, sharex=True, figsize=(10, 10))
    for i in range(npars):
        axes[i].plot(np.array(sampler.chain[:, :, i]).T, rasterized=True)
        axes[i].axvline(x=nburn, ls='--', color='gray', lw=1.5)
        axes[i].set_ylabel(params_to_fit[i], fontsize=16)
    axes[npars-1].set_xlabel(r'steps', fontsize=16)
    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
    else:
        plt.show()


def get_xlim2(best, data, n_emcee, ref=None):
    """
    Get the optimal range for the x-axis, considering the event results.

    Args:
        best (dict): results from PSPL (3+2 params) or 1L2S (5+3 params).
        data (mm.MulensData instance): object containing all the data.
        n_emcee (dict): parameters relevant to emcee fitting.
        ref (float, optional): reference for t_0. Defaults to None.

    Returns:
        list: range for the x-axis, without subtracting 2450000.
    """

    # only works for PSPL case... (A' should be considered for 1L2S)
    # Amax = (best['u_0']**2 + 2) / (best['u_0']*np.sqrt(best['u_0']**2 + 4))

    # Radek: using get_data_magnification from MulensModel
    bst = dict(item for item in list(best.items()) if 'flux' not in item[0])
    fix = None if n_emcee['fix_blend'] is False else {data:
                                                      n_emcee['fix_blend']}
    event = mm.Event(data, model=mm.Model(bst), fix_blend_flux=fix)
    event.get_flux_for_dataset(0)
    Amax = max(event.fits[0].get_data_magnification())
    dividend = best['source_flux']*Amax + best['blending_flux']
    divisor = best['source_flux'] + best['blending_flux']
    deltaI = 2.5*np.log10(dividend/divisor)  # deltaI ~ 3 for PAR-46 :: OK!

    # Get the magnitude at the model peak (mag_peak ~ comp? ok)
    idx_peak = np.argmin(abs(data.time-best['t_0']))
    model_mag = event.fits[0].get_model_magnitudes()
    mag_peak = model_mag[idx_peak]  # comp = data.mag[idx_peak]

    # Summing 0.85*deltaI to the mag_peak, then obtain t_range (+3%)
    mag_baseline = mag_peak + 0.85*deltaI
    idx1 = np.argmin(abs(mag_baseline - model_mag[:idx_peak]))
    idx2 = idx_peak + np.argmin(abs(mag_baseline - model_mag[idx_peak:]))
    t_range = [0.97*(data.time[idx1]-2450000) + 2450000,
               1.03*(data.time[idx2]-2450000) + 2450000]
    t_cen = best['t_0'] if ref is None else ref
    max_diff_t_0 = max(abs(np.array(t_range) - t_cen)) + 100

    if max_diff_t_0 > 250:
        return [t_cen-max_diff_t_0, t_cen+max_diff_t_0]
    return [t_cen-500, t_cen+500]


def plot_fit(best, data, settings, orig_data=None, best_50=None, pdf=""):
    """
    Plot the best-fitting model(s) over the light curve in mag or flux.

    Args:
        best (dict): results from PSPL (3+2 params) or 1L2S (5+3 params).
        data (mm.MulensData instance): object containing all the data.
        settings (dict): all input settings from yaml file.
        orig_data (list, optional): Plot with original data. Defaults to None.
        best_50 (list, optional): Additional percentile result. Defaults to [].
        pdf (str, optional): pdf file to save the plot. Defaults to "".

    Returns:
        mm.Event: final event containing the model and datasets.
    """

    ans, subtract = settings['fitting_parameters']['ans'], True
    fig = plt.figure(figsize=(7.5, 5.5))
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :])  # gs.new_subplotspec((0, 0), rowspan=2)
    event = fit_utils('get_1L2S_event', data, settings, best=best)
    data_label = "Original data" if not orig_data else "Subtracted data"
    event.plot_data(subtract_2450000=subtract, label=data_label)
    plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': subtract,
                   'color': 'black', 't_start': settings['xlim'][0],
                   't_stop': settings['xlim'][1], 'zorder': 10}
    if orig_data:
        orig_data.plot(phot_fmt='mag', color='gray', alpha=0.2,
                       label="Original data")

    txt = f'PSPL ({ans}):' if event.model.n_sources == 1 else f'1L2S ({ans}):'
    for key, val in best.items():
        txt += f'\n{key} = {val:.2f}' if 'flux' not in key else ""
    event.plot_model(label=rf"{txt}", **plot_params)  # % txt
    plt.tick_params(axis='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    ax2 = fig.add_subplot(gs[2:, :], sharex=ax1)
    event.plot_residuals(subtract_2450000=True, zorder=10)
    plt.tick_params(axis='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.xlim(*np.array(settings['xlim']) - subtract*2450000)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.axes(ax1)
    if best_50 is not None:
        model_x = dict((key, best_50[i]) for i, key in enumerate(best.keys()))
        event_x = mm.Event(model=model_x, datasets=[data])
        plot_params['color'] = 'orange'
        event_x.plot_model(label='50th_perc', **plot_params)

    ax1.legend(loc='best')
    if pdf:
        pdf.savefig(fig)
        plt.close('all')
    else:
        plt.show()
    return event


def write_tables(path, settings, name, two_pspl, result,
                 fmt="ascii.commented_header"):
    """
    Save the chains, yaml results and table with results, according with the
    paths and other informations provided in the settings file.

    Args:
        path (str): directory of the Python script and catalogues.
        settings (dict): all input settings from yaml file.
        name (str): name of the photometry file.
        result (tuple): contains the EMCEE outputs and mm.Event instance.
        fmt (str, optional): format of the ascii tables.
    """

    # saving the states to file
    best, name = result[0], name.split('.')[0]
    outputs, n_emcee = settings['other_output'], settings['fitting_parameters']
    bst = dict(item for item in list(best.items()) if 'flux' not in item[0])
    if 'models' in outputs.keys():
        fname = f'{path}/' + outputs['models']['file name'].format(name)
        idxs_remove = list(np.arange(len(bst), len(best)))
        chains = np.delete(result[2], idxs_remove, axis=1)
        chains = Table(chains, names=list(bst.keys())+['chi2'])
        chains.write(fname, format=fmt, overwrite=True)

    # organizing results to be saved in yaml file (as in example16)
    fluxes = dict(item for item in list(best.items()) if 'flux' in item[0])
    perc = dict(item for item in result[3].items() if 'flux' not in item[0])
    perc_fluxes = dict(item for item in result[3].items() if 'flux' in item[0])
    print()
    acc_fraction = np.mean(result[1].acceptance_fraction)
    acor = result[1].get_autocorr_time(quiet=True, discard=n_emcee['nburn'])
    deg_of_freedom = result[4].datasets[0].n_epochs - len(bst)
    pspl_1, pspl_2 = two_pspl
    pspl_1 = str([round(val, 7) for val in pspl_1.values()])
    pspl_2 = str([round(val, 7) for val in pspl_2.values()])
    xlim = str([round(val, 2) for val in settings['xlim']])
    lst = ['', pspl_1, pspl_2, xlim, acc_fraction, np.mean(acor), '', '',
           result[4].chi2, deg_of_freedom, '', '']
    dict_perc_best = {6: perc, 7: perc_fluxes, 10: bst, 11: fluxes}

    # filling and writing the template
    for idx, dict_obj in dict_perc_best.items():
        for key, val in dict_obj.items():
            if idx in [6, 7]:
                uncerts = f'+{val[2]-val[1]:.5f}, -{val[1]-val[0]:.5f}'
                lst[idx] += f'    {key}: [{val[1]:.5f}, {uncerts}]\n'
            else:
                lst[idx] += f'    {key}: {val}\n'
        lst[idx] = lst[idx][:-1]
    with open(f'{path}/../1L2S-result_template.yaml') as file_:
        template_result = file_.read()
    if 'yaml output' in outputs.keys():
        yaml_fname = outputs['yaml output']['file name'].format(name)
        yaml_path = os.path.join(path, yaml_fname)
        lst[0] = sys.argv[1]
        with open(yaml_path, 'w') as yaml_results:
            yaml_results.write(template_result.format(*lst))

    # saving results to table with all the events (e.g. W16)
    if 'table output' in outputs.keys():
        fname, columns, dtypes = outputs['table output'].values()
        if not os.path.isfile(f'{path}/{fname}'):
            result_tab = Table()
            for col, dtype in zip(columns, dtypes):
                result_tab[col] = Column(name=col, dtype=dtype)
        else:
            result_tab = Table.read(f'{path}/{fname}', format='ascii')
        bst_values = [round(val, 5) for val in bst.values()]
        lst = bst_values+[0., 0.] if len(bst) == 3 else bst_values
        lst = [name] + lst + [round(result[4].chi2, 4), deg_of_freedom]
        if name in result_tab['id']:
            idx_event = np.where(result_tab['id'] == name)[0]
            if result_tab[idx_event]['chi2'] > lst[-2]:
                result_tab[idx_event] = lst
        else:
            result_tab.add_row(lst)
        result_tab.sort('id')
        result_tab.write(f'{path}/{fname}', format=fmt, overwrite=True)


if __name__ == '__main__':

    np.random.seed(12343)
    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')

    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as data:
        settings = yaml.safe_load(data)

    fit_binary_source = FitBinarySource(**settings)
    # fit_binary_source.run_fit()
    breakpoint()

    # data_list, file_names = read_data(path, settings['phot_settings'][0])
    # for data, name in zip(data_list, file_names):

    #     name = name.split('/')[-1]
    #     print(f'\n\033[1m * Running fit for {name}\033[0m')
    #     breakpoint()
    #     pdf_dir = settings['plots']['all_plots']['file_dir']
    #     pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_result.pdf")
    #     result, cplot = prefit_split_and_fit(data, settings, pdf=pdf)
    #     # result, cplot = make_all_fittings(data, settings, pdf=pdf)
    #     res_event, two_pspl = result[4], result[5]
    #     pdf.close()
    #     write_tables(path, settings, name, two_pspl, result)

    #     pdf_dir = settings['plots']['triangle']['file_dir']
    #     pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_cplot.pdf")
    #     pdf.savefig(cplot)
    #     pdf.close()
    #     pdf_dir = settings['plots']['best model']['file_dir']
    #     pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_fit.pdf")
    #     plot_fit(result[0], data, settings, pdf=pdf)
    #     pdf.close()

    #     # Split data into two and fit PSPL to get 2L1S initial params
    #     make_2L1S = settings['other_output']['yaml_files_2L1S']['t_or_f']
    #     if make_2L1S and res_event.model.n_sources == 2:
    #         # data_lr = split.split_after_result(res_event, result)[0]
    #         # if isinstance(data_lr[0], list):  ### IMPORTANT LATER!
    #         #     continue
    #         # two_pspl = split.fit_PSPL_twice(data_lr, settings, result)
    #         split.generate_2L1S_yaml_files(path, two_pspl, name, settings)
    #     print("\n--------------------------------------------------")
    # # breakpoint()
