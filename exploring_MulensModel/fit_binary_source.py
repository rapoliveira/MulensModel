"""
Fits binary source model using EMCEE sampler.

The code simulates binary source light curve and fits the model twice:
with source flux ratio found via linear regression and
with source flux ratio as a chain parameter.
"""
from itertools import chain
import os
import sys
import warnings

from astropy.table import Table, Column
try:
    import corner
    import emcee
except ImportError as err:
    print(err)
    print("\nEMCEE or corner could not be imported.")
    print("Get it from: http://dfm.io/emcee/current/user/install/")
    print("and re-run the script")
    sys.exit(1)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# import multiprocessing
import numpy as np
import scipy.optimize as op
import yaml

import MulensModel as mm
import split_data_for_binary_lens as split
from ulens_model_fit import UlensModelFit


class FitBinarySource(UlensModelFit):
    """
    Add documentation later...
    """

    def __init__(self, photometry_files, fitting_parameters,
                 fit_method=None, starting_parameters=None,
                 prior_limits=None, fit_constraints=None, min_values=None,
                 max_values=None, plots=None, other_output=None):

        super().__init__(photometry_files,
                         fitting_parameters=fitting_parameters,
                         fit_method=fit_method,
                         starting_parameters=starting_parameters,
                         prior_limits=prior_limits,
                         fit_constraints=fit_constraints,
                         min_values=min_values,
                         max_values=max_values, plots=plots,
                         other_output=other_output)

        self.photometry_files_orig = photometry_files
        self.starting_parameters = starting_parameters

        self.path = os.path.dirname(os.path.realpath(sys.argv[1]))
        self._check_fit_constraints()
        self._parse_fit_constraints_keys()
        self.read_data()

        for data, name in zip(self.data_list, self.file_names):
            self.datasets = [data]
            self.event_id = name.split('/')[-1].split('.')[0]
            self.photometry_files = self.photometry_files_orig.copy()
            self.photometry_files[0]['file_name'] += self.event_id + '.dat'
            self.setup_fit()
            self.run_initial_fits()

    def read_data(self):
        """
        Read a catalogue or list of catalogues and creates MulensData instance.
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

    def setup_fit(self):
        """
        Write later... Only setup stuff here!!!
        """
        print(f'\n\033[1m * Running fit for {self.event_id}\033[0m')
        self.t_peaks, self.t_peaks_orig = [], []
        if self.starting_parameters['t_peaks'] is not False:
            self._get_peak_times()

        fix_blend = self._fitting_parameters.pop('fix_blend')
        fix_dict = {self.datasets[0]: fix_blend}
        self.fix_blend = None if fix_blend is False else fix_dict
        self.sigmas_emcee = self._fitting_parameters.pop('sigmas')
        self.ans_emcee = self._fitting_parameters.pop('ans')
        self.clean_cplot = self._fitting_parameters.pop('clean_cplot')
        self._parse_fitting_parameters_EMCEE()
        self._get_n_walkers()

    def _get_peak_times(self):
        """
        Write docstrings later...
        """
        tab_file = self.starting_parameters['t_peaks']
        tab = Table.read(os.path.join(self.path, tab_file), format='ascii')

        event = self.event_id.replace('_OGLE', '').replace('_', '.')
        line = tab[tab['obj_id'] == event]
        self.t_peaks = np.array([line['t_peak'][0], line['t_peak_2'][0]])
        self.t_peaks_orig = self.t_peaks.copy()

    def run_initial_fits(self):
        """
        Run the fit, print the output, and make the plots.

        This function does not accept any parameters. All the settings
        are passed via __init__().
        """
        self._quick_fits_pspl_subtract_pspl()
        self.binary_source_start = {'t_0_1': self.quick_pspl_1['t_0'],
                                    'u_0_1': self.quick_pspl_1['u_0'],
                                    't_0_2': self.quick_pspl_2['t_0'],
                                    'u_0_2': self.quick_pspl_2['u_0'],
                                    't_E': self.t_E_init}

        self._scipy_minimize_t_E_only(self.binary_source_start)
        self.binary_source_start['t_E'] = self.t_E_scipy
        model_start = mm.Model(self.binary_source_start)
        ev_st = mm.Event(self.datasets[0], model=model_start)

        self._fit_parameters_other = []
        output = self._fit_emcee(self.binary_source_start,
                                 self.sigmas_emcee[1],
                                 self._ln_prob, ev_st, settings)
        breakpoint()
        self.event_temp = self._get_binary_source_event(output[0])
        # event_1L2S = fit_utils('get_1L2S_event', data, settings, best=output[0])
        ### STOPPED HERE... Small problem with fitted fluxes as -np.inf

        # *** Add all the fitting routine here... calling short functions!
        # 2) EMCEE to get a first estimate for 1L2S -- OK!
        # 3) Split data and fit PSPL twice with EMCEE
        # 4) Get final 1L2S fit...

        print("\n--------------------------------------------------")

    def _quick_fits_pspl_subtract_pspl(self):
        """
        First step: quick estimate of PSPL models using scipy.minimize.
        Two fits are carried out: with original data and then with data
        subtracted from the first fit.
        """
        self.quick_pspl_1 = self._run_scipy_minimize()
        self._subtract_model_from_data()
        self.quick_pspl_2 = self._run_scipy_minimize(self.subt_data)

    def _run_scipy_minimize(self, data=None):
        """
        Write later... Decide about gradient and which...
        """
        if data is None:
            data = self.datasets[0]
        self._guess_starting_params(self.t_peaks)
        model = mm.Model(self.start_dict)
        ev_st = mm.Event(data, model=model, fix_blend_flux=self.fix_blend)

        # Nelder-Mead (no gradient)
        x0 = list(self.start_dict.values())
        arg = (list(self.start_dict.keys()), ev_st)
        bnds = [(x0[0]-50, x0[0]+50), (1e-5, 3), (1e-2, None)]
        r_ = op.minimize(split.chi2_fun, x0=x0, args=arg, bounds=bnds,
                         method='Nelder-Mead')
        results = [{'t_0': r_.x[0], 'u_0': r_.x[1], 't_E': r_.x[2]}]

        # Options with gradient from jacobian
        # L-BFGS-B and TNC accept `bounds``, Newton-CG doesn't
        r_ = op.minimize(split.chi2_fun, x0=x0, args=arg, method='L-BFGS-B',
                         bounds=bnds, jac=split.jacobian, tol=1e-3)
        results.append({'t_0': r_.x[0], 'u_0': r_.x[1], 't_E': r_.x[2]})

        return results[1]

    def _guess_starting_params(self, t_bright=None):
        """
        Guess PSPL parameters: t_0 from brightest points, u_0 from the
        flux difference between baseline and brightest point and t_E from
        a quick Nelder-Mead minimization.
        """
        self._guess_initial_t_0(t_bright)
        self._guess_initial_u_0()
        self._guess_initial_t_E()
        start = {'t_0': round(self.t_init, 1), 'u_0': self.u_init,
                 't_E': self.t_E_init}

        self._scipy_minimize_t_E_only(start)
        self.start_dict = start.copy()
        self.start_dict['t_E'] = round(self.t_E_scipy, 2)

    def _guess_initial_t_0(self, t_bright=None):
        """
        Write later...
        """
        data_time = self.datasets[0].time
        data_mag = self.datasets[0].mag
        t_brightest = np.median(data_time[np.argsort(data_mag)][:9])

        if isinstance(t_bright, np.ndarray):
            if len(t_bright) > 0 and np.all(t_bright != 0):
                subt = np.abs(t_bright - t_brightest + 2450000)
                t_brightest = 2450000 + t_bright[np.argmin(subt)]
                self.t_peaks = np.delete(t_bright, np.argmin(subt))
        elif t_bright not in [None, False]:
            raise ValueError('t_bright should be a list of peak times, False'
                             ' or None.')

        self.t_init = t_brightest

    def _guess_initial_u_0(self):
        """
        Write later...
        """
        data = self.datasets[0]

        # Starting the baseline (360d or half-data window?)
        t_diff = data.time - self.t_init
        t_window = [self.t_init + min(t_diff)/2., self.t_init + max(t_diff)/2.]
        t_mask = (data.time < t_window[0]) | (data.time > t_window[1])
        flux_base = np.median(data.flux[t_mask])
        # flux_mag_base = (flux_base, np.std(data.flux[t_mask]),
        #                  np.median(data.mag[t_mask]),
        #                  np.std(data.mag[t_mask]))

        # Get the brightest flux around self.t_init (to avoid outliers)
        idx_min = np.argmin(abs(t_diff))
        min_, max_ = max(idx_min-5, 0), min(idx_min+6, len(data.mag))
        # mag_peak = min(data.mag[min_:max_]) # [idx_min-5:idx_min+6]
        flux_peak = max(data.flux[min_:max_])

        # Compute magnification and corresponding u_0(A)
        magnif_A = flux_peak / flux_base
        self.u_init = np.sqrt(2*magnif_A / np.sqrt(magnif_A**2 - 1) - 2)
        self.u_init = round(self.u_init, 3)

    def _guess_initial_t_E(self):
        """
        Write later...
        """
        if 't_E' in self._fit_constraints['prior'].keys():
            t_E_prior = self._fit_constraints['prior']['t_E']
            self.t_E_init = round(np.exp(float(t_E_prior.split()[1])), 1)
        else:
            self.t_E_init = 25.

    def _scipy_minimize_t_E_only(self, model_dict):
        """
        Write...
        """
        aux_event = mm.Event(self.datasets[0], model=mm.Model(model_dict))
        x0, arg = [model_dict['t_E']], (['t_E'], aux_event)
        bnds = [(0.1, None)]
        r_ = op.minimize(split.chi2_fun, x0=x0, args=arg, bounds=bnds,
                         method='Nelder-Mead')

        self.t_E_scipy = r_.x[0]

    def _subtract_model_from_data(self):
        """
        Write later...
        """
        data = self.datasets[0]
        model = mm.Model(self.quick_pspl_1)
        aux_event = mm.Event(data, model=model, fix_blend_flux=self.fix_blend)
        (flux, blend) = aux_event.get_flux_for_dataset(0)

        fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
        subt_data = np.c_[data.time, fsub, data.err_flux][fsub > 0]
        self.subt_data = mm.MulensData(subt_data.T, phot_fmt='flux')

    def _ln_like(self, theta, event, params_to_fit):
        """
        Get the value of the likelihood function from chi2 of event.
        NOTE: Still adapt it to use example_16 functions!!!

        Args:
            theta (np.array): chain parameters to sample the likelihood.
            event (mm.Event): event instance containing the model and datasets.
            params_to_fit (list): microlensing parameters to be fitted.

        Returns:
            tuple: likelihood value (-0.5*chi2) and fluxes of the event.
        """

        for (param, theta_) in zip(params_to_fit, theta):
            # Here we handle fixing source flux ratio:
            if param == 'flux_ratio':
                # implemented for a single dataset
                # event.fix_source_flux_ratio = {my_dataset: theta_} # original(?)
                event.fix_source_flux_ratio = {event.datasets[0]: theta_}
            else:
                setattr(event.model.parameters, param, theta_)
        event.get_chi2()

        return -0.5 * event.chi2, event.fluxes[0]

    def _ln_prior(self, theta, event, params_to_fit):
        """
        Apply all the priors (minimum, maximum and distributions).

        Args:
            theta (np.array): chain parameters to sample the prior.
            event (mm.Event): Event instance containing the model and datasets.
            params_to_fit (list): name of the parameters to be fitted.

        Raises:
            ValueError: if the maximum values for t_0 are not a list.
            ValueError: if Mróz priors for t_E are selected.
            ValueError: if input prior is not implemented.

        Returns:
            float: -np.inf or prior value to be added on the likelihood.
        """

        stg_priors = self._fit_constraints
        stg_min_max = [self._min_values, self._max_values]
        if not isinstance(stg_min_max[1]['t_0'], list):
            raise ValueError('t_0 max_values should be of list type.')
        ln_prior_t_E, ln_prior_fluxes = 0., 0.

        # Ensuring that t_0_1 < t_0_2 or t_0_1 > t_0_2
        if event.model.n_sources == 2:
            init_t_0_1, init_t_0_2 = self._init_params_emcee[:3:2]
            if (init_t_0_1 > init_t_0_2) and (theta[0] < theta[2]):
                return -np.inf
            elif (init_t_0_2 > init_t_0_1) and (theta[2] < theta[0]):
                return -np.inf

        # Limiting min and max values (all minimum, then t_0 and u_0 maximum)
        for (idx, param) in enumerate(params_to_fit):
            if param[:3] in stg_min_max[0].keys():
                if theta[idx] < stg_min_max[0][param[:3]]:
                    return -np.inf
            if 't_0' in param and 't_0' in stg_min_max[1].keys():
                data_time = event.datasets[0].time
                t_range = data_time[::len(data_time)-1]
                t_range = t_range + np.array(stg_min_max[1]['t_0'])
                if not t_range[0] < theta[idx] < t_range[1]:
                    return -np.inf
            if 'u_0' in param and 'u_0' in stg_min_max[1].keys():
                if theta[idx] > stg_min_max[1]['u_0']:
                    return -np.inf

        # Prior in t_E (only lognormal so far, tbd: Mroz17/20)
        # OBS: Still need to remove the ignore warnings line (related to log?)
        if 't_E' in stg_priors['prior'].keys():
            t_E_prior = stg_priors['prior']['t_E']
            t_E_val = theta[params_to_fit.index('t_E')]
            if 'lognormal' in t_E_prior:
                # if t_E >= 1.:
                warnings.filterwarnings("ignore", category=RuntimeWarning)  # bad!
                prior = [float(x) for x in t_E_prior.split()[1:]]
                ln_prior_t_E = - (np.log(t_E_val) - prior[0])**2 / (2*prior[1]**2)
                ln_prior_t_E -= np.log(t_E_val * np.sqrt(2*np.pi)*prior[1])
            elif 'Mroz et al.' in stg_priors['ln_prior']['t_E']:
                raise ValueError('Still implementing Mroz et al. priors.')
            else:
                raise ValueError('t_E prior type not allowed.')

        # Avoiding negative source/blending fluxes (Radek)
        _ = event.get_chi2()
        if 'negative_blending_flux_sigma_mag' in stg_priors.keys():
            sig_ = stg_priors['negative_blending_flux_sigma_mag']
            for flux in [*event.source_fluxes[0], event.blend_fluxes[0]]:
                if flux < 0.:
                    ln_prior_fluxes += -1/2 * (flux/sig_)**2  # 1000, 100 or less?
        elif self._fit_constraints['no_negative_blending_flux']:
            if event.blend_fluxes[0] < 0:
                return -np.inf

        return 0.0 + ln_prior_t_E + ln_prior_fluxes

    def _ln_prob(self, theta, event, params_to_fit):
        """
        Combines likelihood value and priors for a given set of parameters.

        Args:
            theta (np.array): chain parameters to sample the likelihood+prior.
            event (mm.Event): Event instance containing the model and datasets.
            params_to_fit (list): name of the parameters to be fitted.

        Returns:
            tuple: value of prior+likelihood, source and blending fluxes.
        """

        ln_prior_ = self._ln_prior(theta, event, params_to_fit)
        if not np.isfinite(ln_prior_):
            return -np.inf, np.array([-np.inf, -np.inf]), -np.inf
        ln_like_, fluxes = self._ln_like(theta, event, params_to_fit)

        # In the cases that source fluxes are negative we want to return
        # these as if they were not in priors.
        if np.isnan(ln_like_):
            return -np.inf, np.array([-np.inf, -np.inf]), -np.inf

        return ln_prior_ + ln_like_, fluxes[0], fluxes[1]

    def _setup_fit_emcee_binary(self):
        """
        Setup EMCEE fit for binary source model.
        """
        self.params_to_fit = list(self.binary_source_start.keys())
        self._n_fit_parameters = len(self.params_to_fit)
        sigmas = self.sigmas_emcee[1]
        # self._n_burn = self._fitting_parameters['n_burn']
        breakpoint()
        # CONTINUE FROM HERE :: working, adapting to example_16!!!

    # def _setup_fit_EMCEE(self):
    #     """
    #     Setup EMCEE fit  --  COPIED FROM EXAMPLE_16 so far...
    #     """
    #     # self._sampler = emcee.EnsembleSampler(
    #     #     self._n_walkers, self._n_fit_parameters, self._ln_prob)

    def _fit_emcee(self, dict_start, sigmas, ln_prob, event, settings):
        """
        Fit model using EMCEE (Foreman-Mackey et al. 2013) and print results.

        Args:
            dict_start (dict): dict that specifies values of these parameters.
            sigmas (list): sigma values used to find starting values.
            ln_prob (func): function returning logarithm of probability.
            event (mm.Event): Event instance containing the model and datasets.
            settings (dict): all settings from yaml file.

        Raises:
            RuntimeError: if number of dimensions different than 3 or 5 is given

        Returns:
            tuple: with EMCEE results (best, sampler, samples, percentiles)
        """

        params_to_fit = list(dict_start.keys())  # OK!
        n_dim, sigmas = len(params_to_fit), np.array(sigmas)  # OK!
        n_emcee = self._fitting_parameters  # OK!
        nwlk, nstep = n_emcee['n_walkers'], n_emcee['n_steps']  # OK!
        nburn = n_emcee['n_burn']  # OK!
        emcee_args = (event, params_to_fit)  # Continue from here...
        nfit = settings.get('123_fits', '3rd fit')  # to be removed!
        term = ['PSPL to original', 'PSPL to subtracted', '1L2S to original']
        print(f'\n\033[1m -- {nfit}: {term[int(nfit[0])-1]} data...\033[0m')
        breakpoint()

        # Doing the 1L2S fitting in two steps (or all? best in 1st and 3rd fits)
        # if nfit in ['1st fit', '3rd fit']:
        if nfit in ['3rd fit']:
            init_params = np.array(list(dict_start.values()))
            random_sample = np.random.randn(nwlk, n_dim) * 10 * sigmas
            start = abs(init_params + random_sample)
            sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, args=emcee_args)
            sampler.run_mcmc(start, int(nstep/2), progress=n_emcee['progress'])
            samples = sampler.chain[:, int(nburn/2):, :].reshape((-1, n_dim))
            init_params = np.percentile(samples, 50, axis=0)
            # prob_temp = sampler.lnprobability[:, int(nburn/2):].reshape((-1))
            # mean = samples[np.argmax(prob_temp)]
        start = abs(init_params + random_sample / 10)

        # Run emcee (this can take some time):
        blobs = [('source_fluxes', list), ('blend_fluxes', float)]
        sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, blobs_dtype=blobs,
                                        args=emcee_args)  # backend=backend)
        sampler.run_mcmc(start, nstep, progress=n_emcee['progress'])

        # Setting up multi-threading (std: fork in Linux, spawn in Mac)
        # multiprocessing.set_start_method("fork", force=True)
        # os.environ["OMP_NUM_THREADS"] = "1"
        # with multiprocessing.Pool() as pool:
        #     sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, pool=pool,
        #                                     args=(event, params_to_fit, spec))
        #     sampler.run_mcmc(start, nstep, progress=n_emcee['progress'])
        # pool.close()

        # Remove burn-in samples and reshape:
        samples = sampler.chain[:, nburn:, :].reshape((-1, n_dim))
        blobs = sampler.get_blobs()[nburn:].T.flatten()
        source_fluxes = np.array(list(chain.from_iterable(blobs))[::2])
        blend_flux = np.array(list(chain.from_iterable(blobs))[1::2])
        prob = sampler.lnprobability[:, nburn:].reshape((-1))
        if len(params_to_fit) == 3:
            samples = np.c_[samples, source_fluxes[:, 0], blend_flux, prob]
        elif len(params_to_fit) == 5:
            samples = np.c_[samples, source_fluxes[:, 0], source_fluxes[:, 1],
                            blend_flux, prob]
        else:
            raise RuntimeError('Wrong number of dimensions')

        # Print results from median and 1sigma-perc:
        perc = np.percentile(samples, [16, 50, 84], axis=0)
        print("Fitted parameters:")
        for i in range(n_dim):
            r = perc[1, i]
            msg = params_to_fit[i] + ": {:.5f} +{:.5f} -{:.5f}"
            print(msg.format(r, perc[2, i]-r, r-perc[0, i]))

        # Adding fluxes to params_to_fit and setting up pars_perc
        if samples.shape[1]-1 == len(params_to_fit) + 2:
            params_to_fit += ['source_flux', 'blending_flux']
        elif samples.shape[1]-1 == len(params_to_fit) + 3:
            params_to_fit += ['source_flux_1', 'source_flux_2', 'blending_flux']
        pars_perc = dict(zip(params_to_fit, perc.T))

        # We extract best model parameters and chi2 from event:
        best_idx = np.argmax(prob)
        best = samples[best_idx, :-1] if self.ans_emcee == 'max_prob' else perc[1]
        for (key, value) in zip(params_to_fit, best):
            if key == 'flux_ratio':
                event.fix_source_flux_ratio = {event.datasets[0]: value}
            else:
                setattr(event.model.parameters, key, value)
        print("\nSmallest chi2 model:")
        print(*[repr(b) if isinstance(b, float) else b for b in best])
        deg_of_freedom = event.datasets[0].n_epochs - n_dim
        print(f"chi2 = {event.chi2:.8f}, dof = {deg_of_freedom}")
        best = dict(zip(params_to_fit, best))

        # Cleaning posterior and cplot if required by the user
        if n_emcee['clean_cplot']:
            new_states, new_perc = clean_posterior_emcee(sampler, best, nburn)
            if new_states is not None:
                pars_perc = dict(zip(params_to_fit, new_perc.T))
                samples = new_states

        # return best, pars_best, event.get_chi2(), states, sampler
        return best, sampler, samples, pars_perc

    def _get_binary_source_event(self, best):

        data = self.datasets[0]
        bst = dict(b_ for b_ in list(best.items()) if 'flux' not in b_[0])
        fix_source = {data: [best[p] for p in best if 'source' in p]}
        event_1L2S = mm.Event(data, model=mm.Model(bst),
                              fix_source_flux=fix_source,
                              fix_blend_flux={data: best['blending_flux']})
        event_1L2S.get_chi2()

        return event_1L2S


def fit_utils(method, data, settings, model=None, best=None):
    """
    Useful short functions to be applied in mm.Data instance.

    Args:
        method (str): which method to execute.
        data (mm.MulensData): data instance of a single event.
        settings (dict): all input settings from yaml file.
        model (dict, optional): parameters to get mm.Model. Defaults to None.
        best (dict, optional): parameters to get mm.Event. Defaults to None.

    Raises:
        ValueError: if invalid method is given.

    Returns:
        [...]: depends on the method (dict, list or mm.Event)
    """

    n_emcee = settings['fitting_parameters']
    fix = None if n_emcee['fix_blend'] is False else {data:
                                                      n_emcee['fix_blend']}

    if method == 'get_t_E_1L2S':
        aux_event = mm.Event(data, model=mm.Model(model))
        x0, arg = [model['t_E']], (['t_E'], aux_event)
        bnds = [(0.1, None)]
        r_ = op.minimize(split.chi2_fun, x0=x0, args=arg, bounds=bnds,
                         method='Nelder-Mead')
        model['t_E'] = r_.x[0]
        return list(model.values())

    elif method == 'get_1L2S_event':
        bst = dict(b_ for b_ in list(best.items()) if 'flux' not in b_[0])
        fix_source = {data: [best[p] for p in best if 'source' in p]}
        event_1L2S = mm.Event(data, model=mm.Model(bst),
                              fix_source_flux=fix_source,
                              fix_blend_flux={data: best['blending_flux']})
        event_1L2S.get_chi2()
        return event_1L2S

    else:
        raise ValueError('Invalid method sent to function fit_utils().')


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
    if settings['starting_parameters']['t_peaks'] is not False:
        settings = fit_utils('get_peak_times', data, settings)
    model_1 = fit_utils('scipy_minimize', data, settings)
    data_2_subt = fit_utils('subt_data', data, settings, model_1)
    # fm_base = get_initial_t0_u0(data_2_subt, settings)[1]  # above 3sigma?
    # if no data above 3sigma:
    #   - run MCMC and return...
    # else:
    model_2 = fit_utils('scipy_minimize', data_2_subt, settings)
    #
    n_emcee = settings['fitting_parameters']
    start = {'t_0_1': model_1['t_0'], 'u_0_1': model_1['u_0'], 't_0_2':
             model_2['t_0'], 'u_0_2': model_2['u_0'], 't_E': 25}
    t_E_optimal = fit_utils('get_t_E_1L2S', data, settings, start)
    start['t_E'] = round(t_E_optimal[4], 2)
    ev_st = mm.Event(data, model=mm.Model(start))
    output = fit_emcee(start, n_emcee['sigmas'][1], ln_prob, ev_st, settings)
    event_1L2S = fit_utils('get_1L2S_event', data, settings, best=output[0])
    data_left_right, t_min = split.split_after_result(event_1L2S, output, settings)

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


# def ln_like(theta, event, params_to_fit):
#     """
#     Get the value of the likelihood function from chi2 of event.
#     NOTE: Still adapt it to use example_16 functions!!!

#     Args:
#         theta (np.array): chain parameters to sample the likelihood.
#         event (mm.Event): event instance containing the model and datasets.
#         params_to_fit (list): microlensing parameters to be fitted.

#     Returns:
#         tuple: likelihood value (-0.5*chi2) and fluxes of the event.
#     """

#     for (param, theta_) in zip(params_to_fit, theta):
#         # Here we handle fixing source flux ratio:
#         if param == 'flux_ratio':
#             # implemented for a single dataset
#             # event.fix_source_flux_ratio = {my_dataset: theta_} # original(?)
#             event.fix_source_flux_ratio = {event.datasets[0]: theta_}
#         else:
#             setattr(event.model.parameters, param, theta_)
#     event.get_chi2()

#     return -0.5 * event.chi2, event.fluxes[0]


# def ln_prior(theta, event, params_to_fit, settings):
#     """
#     Apply all the priors (minimum, maximum and distributions).

#     Args:
#         theta (np.array): chain parameters to sample the prior.
#         event (mm.Event): Event instance containing the model and datasets.
#         params_to_fit (list): name of the parameters to be fitted.
#         settings (dict): all settings from yaml file.

#     Raises:
#         ValueError: if the maximum values for t_0 are not a list.
#         ValueError: if Mróz priors for t_E are selected.
#         ValueError: if input prior is not implemented.

#     Returns:
#         float: -np.inf or prior value to be added on the likelihood.
#     """

#     stg_priors = settings['fit_constraints']
#     stg_min_max = [settings['min_values'], settings['max_values']]
#     if not isinstance(stg_min_max[1]['t_0'], list):
#         raise ValueError('t_0 max_values should be of list type.')
#     ln_prior_t_E, ln_prior_fluxes = 0., 0.

#     # Ensuring that t_0_1 < t_0_2 or t_0_1 > t_0_2
#     if event.model.n_sources == 2:
#         init_params = settings['init_params_1L2S']
#         if (init_params[0] > init_params[2]) and (theta[0] < theta[2]):
#             return -np.inf
#         elif (init_params[2] > init_params[0]) and (theta[2] < theta[0]):
#             return -np.inf

#     # Limiting min and max values (all minimum, then t_0 and u_0 maximum)
#     for param in params_to_fit:
#         if param[:3] in stg_min_max[0].keys():
#             if theta[params_to_fit.index(param)] < stg_min_max[0][param[:3]]:
#                 return -np.inf
#         if 't_0' in param and 't_0' in stg_min_max[1].keys():
#             t_range = event.datasets[0].time[::len(event.datasets[0].time)-1]
#             t_range = t_range + np.array(stg_min_max[1]['t_0'])
#             if not t_range[0] < theta[params_to_fit.index(param)] < t_range[1]:
#                 return -np.inf
#         if 'u_0' in param and 'u_0' in stg_min_max[1].keys():
#             if theta[params_to_fit.index(param)] > stg_min_max[1]['u_0']:
#                 return -np.inf

#     # Prior in t_E (only lognormal so far, tbd: Mroz17/20)
#     # OBS: Still need to remove the ignore warnings line (related to log?)
#     if 't_E' in stg_priors['prior'].keys():
#         t_E_prior = stg_priors['prior']['t_E']
#         t_E_val = theta[params_to_fit.index('t_E')]
#         if 'lognormal' in t_E_prior:
#             # if t_E >= 1.:
#             warnings.filterwarnings("ignore", category=RuntimeWarning)  # bad!
#             prior = [float(x) for x in t_E_prior.split()[1:]]
#             ln_prior_t_E = - (np.log(t_E_val) - prior[0])**2 / (2*prior[1]**2)
#             ln_prior_t_E -= np.log(t_E_val * np.sqrt(2*np.pi)*prior[1])
#         elif 'Mroz et al.' in stg_priors['ln_prior']['t_E']:
#             raise ValueError('Still implementing Mroz et al. priors.')
#         else:
#             raise ValueError('t_E prior type not allowed.')

#     # Avoiding negative source/blending fluxes (Radek)
#     _ = event.get_chi2()
#     if 'negative_blending_flux_sigma_mag' in stg_priors.keys():
#         sig_ = stg_priors['negative_blending_flux_sigma_mag']
#         for flux in [*event.source_fluxes[0], event.blend_fluxes[0]]:
#             if flux < 0.:
#                 ln_prior_fluxes += -1/2 * (flux/sig_)**2  # 1000, 100 or less?
#     elif 'no_negative_blending_flux' in stg_priors.keys():
#         if event.blend_fluxes[0] < 0:
#             return -np.inf

#     return 0.0 + ln_prior_t_E + ln_prior_fluxes


# def ln_prob(theta, event, params_to_fit, settings):
#     """
#     Combines likelihood value and priors for a given set of parameters.

#     Args:
#         theta (np.array): chain parameters to sample the likelihood+prior.
#         event (mm.Event): Event instance containing the model and datasets.
#         params_to_fit (list): name of the parameters to be fitted.
#         settings (dict): all settings from yaml file.

#     Returns:
#         tuple: value of prior+likelihood, source and blending fluxes.
#     """

#     ln_prior_ = ln_prior(theta, event, params_to_fit, settings)
#     if not np.isfinite(ln_prior_):
#         return -np.inf, np.array([-np.inf, -np.inf]), -np.inf
#     ln_like_, fluxes = ln_like(theta, event, params_to_fit)

#     # In the cases that source fluxes are negative we want to return
#     # these as if they were not in priors.
#     if np.isnan(ln_like_):
#         return -np.inf, np.array([-np.inf, -np.inf]), -np.inf

#     return ln_prior_ + ln_like_, fluxes[0], fluxes[1]


def fit_emcee(dict_start, sigmas, ln_prob, event, settings):
    """
    Fit model using EMCEE (Foreman-Mackey et al. 2013) and print results.

    Args:
        dict_start (dict): dict that specifies values of these parameters.
        sigmas (list): sigma values used to find starting values.
        ln_prob (func): function returning logarithm of probability.
        event (mm.Event): Event instance containing the model and datasets.
        settings (dict): all settings from yaml file.

    Raises:
        RuntimeError: if number of dimensions different than 3 or 5 is given

    Returns:
        tuple: with EMCEE results (best, sampler, samples, percentiles)
    """

    params_to_fit, mean = list(dict_start.keys()), list(dict_start.values())
    n_dim, sigmas = len(params_to_fit), np.array(sigmas)
    n_emcee = settings['fitting_parameters']
    nwlk, nstep, nburn = n_emcee['nwlk'], n_emcee['nstep'], n_emcee['nburn']
    emcee_args = (event, params_to_fit, settings)
    nfit = settings['123_fits'] if '123_fits' in settings.keys() else '3rd fit'
    term = ['PSPL to original', 'PSPL to subtracted', '1L2S to original']
    print(f'\n\033[1m -- {nfit}: {term[int(nfit[0])-1]} data...\033[0m')

    # Doing the 1L2S fitting in two steps (or all? best in 1st and 3rd fits)
    # if nfit in ['1st fit', '3rd fit']:
    if nfit in ['3rd fit']:
        data = event.datasets[0]
        mean = fit_utils('get_t_E_1L2S', data, settings, model=dict_start)
        settings['init_params_1L2S'] = mean
        start = [mean + np.random.randn(n_dim)*10*sigmas for i in range(nwlk)]
        start = abs(np.array(start))
        sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, args=emcee_args)
        sampler.run_mcmc(start, int(nstep/2), progress=n_emcee['progress'])
        samples = sampler.chain[:, int(nburn/2):, :].reshape((-1, n_dim))
        mean = np.percentile(samples, 50, axis=0)
        settings['init_params_1L2S'] = mean
        # prob_temp = sampler.lnprobability[:, int(nburn/2):].reshape((-1))
        # mean = samples[np.argmax(prob_temp)]
    start = [mean + np.random.randn(n_dim) * sigmas for i in range(nwlk)]
    start = abs(np.array(start))

    # Run emcee (this can take some time):
    blobs = [('source_fluxes', list), ('blend_fluxes', float)]
    sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, blobs_dtype=blobs,
                                    args=emcee_args)  # backend=backend)
    # sampler = emcee.EnsembleSampler(
    #     nwlk, n_dim, ln_prob,
    #     moves=[(emcee.moves.DEMove(),0.8),(emcee.moves.DESnookerMove(),0.2)],
    #     args=(event, params_to_fit, spec))
    sampler.run_mcmc(start, nstep, progress=n_emcee['progress'])

    # Setting up multi-threading (std: fork in Linux, spawn in Mac)
    # multiprocessing.set_start_method("fork", force=True)
    # os.environ["OMP_NUM_THREADS"] = "1"
    # with multiprocessing.Pool() as pool:
    #     sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, pool=pool,
    #                                     args=(event, params_to_fit, spec))
    #     sampler.run_mcmc(start, nstep, progress=n_emcee['progress'])
    # pool.close()

    # Remove burn-in samples and reshape:
    samples = sampler.chain[:, nburn:, :].reshape((-1, n_dim))
    blobs = sampler.get_blobs()[nburn:].T.flatten()
    source_fluxes = np.array(list(chain.from_iterable(blobs))[::2])
    blend_flux = np.array(list(chain.from_iterable(blobs))[1::2])
    prob = sampler.lnprobability[:, nburn:].reshape((-1))
    if len(params_to_fit) == 3:
        samples = np.c_[samples, source_fluxes[:, 0], blend_flux, prob]
    elif len(params_to_fit) == 5:
        samples = np.c_[samples, source_fluxes[:, 0], source_fluxes[:, 1],
                        blend_flux, prob]
    else:
        raise RuntimeError('Wrong number of dimensions')

    # Print results from median and 1sigma-perc:
    perc = np.percentile(samples, [16, 50, 84], axis=0)
    print("Fitted parameters:")
    for i in range(n_dim):
        r = perc[1, i]
        msg = params_to_fit[i] + ": {:.5f} +{:.5f} -{:.5f}"
        print(msg.format(r, perc[2, i]-r, r-perc[0, i]))

    # Adding fluxes to params_to_fit and setting up pars_perc
    if samples.shape[1]-1 == len(params_to_fit) + 2:
        params_to_fit += ['source_flux', 'blending_flux']
    elif samples.shape[1]-1 == len(params_to_fit) + 3:
        params_to_fit += ['source_flux_1', 'source_flux_2', 'blending_flux']
    pars_perc = dict(zip(params_to_fit, perc.T))

    # We extract best model parameters and chi2 from event:
    best_idx = np.argmax(prob)
    best = samples[best_idx, :-1] if n_emcee['ans'] == 'max_prob' else perc[1]
    for (key, value) in zip(params_to_fit, best):
        if key == 'flux_ratio':
            event.fix_source_flux_ratio = {event.datasets[0]: value}
        else:
            setattr(event.model.parameters, key, value)
    print("\nSmallest chi2 model:")
    print(*[repr(b) if isinstance(b, float) else b for b in best])
    deg_of_freedom = event.datasets[0].n_epochs - n_dim
    print(f"chi2 = {event.chi2:.8f}, dof = {deg_of_freedom}")
    best = dict(zip(params_to_fit, best))

    # Cleaning posterior and cplot if required by the user
    if n_emcee['clean_cplot']:
        new_states, new_perc = clean_posterior_emcee(sampler, best, nburn)
        if new_states is not None:
            pars_perc = dict(zip(params_to_fit, new_perc.T))
            samples = new_states

    # return best, pars_best, event.get_chi2(), states, sampler
    return best, sampler, samples, pars_perc


def clean_posterior_emcee(sampler, params, n_burn):
    """
    OLD: manipulate emcee chains to reject stray walkers and clean posterior.

    Args:
        sampler (emcee.EnsembleSampler): sampler that contain the chains.
        params (...): set of best parameters from fit_emcee function.
        n_burn (int): number of steps considered as burn-in (< n_steps).

    Returns:
        tuple: new states and 1sigma quantiles of these states.
    """

    # here I will put all the manipulation of emcee chains...

    # Rejecting stray walkers (copied from King)
    # acc = sampler.acceptance_fraction
    # w = (abs(acc-np.median(acc)) < max(np.std(acc), 0.1))
    # gwlk = len(acc[w])
    #  print(gwlk,'{:4.1f}'.format(ksize.to(u.arcsec).value), \
    #       np.median(acc),np.median(sig_den/den_kern))
    # states = sampler.chain[w, burn_end:, :]
    # states = np.reshape(states,[gwlk*nburn,npars])

    # Rejecting stray walkers (copied from rad_profile)
    sampler.get_autocorr_time(tol=0)
    acc = sampler.acceptance_fraction
    w = np.where(abs(acc-np.median(acc)) < min(0.1, 3*np.std(acc)))

    if len(w[0]) == 0:
        return None, None
    # states = sampler.chain[w[0],burnin::thin,:].copy()  # with thinning
    new_states = sampler.chain[w[0], n_burn::, :].copy()  # no thinning
    gwlk, nthin, npars = np.shape(new_states)
    new_states = np.reshape(new_states, [gwlk*nthin, npars])

    # Trying to add the source_fluxes after flattening...
    blobs = sampler.get_blobs().T  # [:10]
    blobs = blobs[w[0], n_burn:].reshape(-1)
    source_fluxes = np.array(list(chain.from_iterable(blobs))[::2]).T
    blend_flux = np.array(list(chain.from_iterable(blobs))[1::2])
    prob = sampler.lnprobability[w[0], n_burn:].reshape(-1)
    if npars == 3:
        new_states = np.c_[new_states, source_fluxes[0], blend_flux, prob]
    elif npars == 5:
        new_states = np.c_[new_states, source_fluxes[0], source_fluxes[1],
                           blend_flux, prob]

    # new_samples = samples[w[0]].copy()[npars:.]
    # breakpoint()

    n_rej, perc_rej = sampler.nwalkers-gwlk, 100*(1-gwlk/sampler.nwalkers)
    print(f'Obs: {n_rej} walkers ({round(perc_rej)}%) were rejected')

    # Finding median values and confidence intervals... FIT SKEWED GAUSSIANS!!!
    # To-Do... [...]
    # test = params # [...]
    # do things here and return best only here!!! COPIED BELOW

    # Getting states and reshaping: e.g. (20, 1500, 3) -> (30000, 3)
    # states = sampler.chain[:, n_burn:, :]
    # # states = np.reshape(states,[gwlk*n_burn,len(params_to_fit)])
    # after_burn = n_steps-n_burn
    # states = np.reshape(states,[n_walkers*after_burn, len(params_to_fit)])
    # breakpoint()
    # w = np.quantile(new_states,[0.16,0.50,0.84],axis=0)
    # pars_best = w[1,:]
    # perr_low  = w[0,:]-pars_best
    # perr_high = w[2,:]-pars_best

    return new_states, np.quantile(new_states, [0.16, 0.50, 0.84], axis=0)


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
        fname = f'{path}/' + outputs['models']['file_dir'].format(name)
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
    fit_binary_source.run_fit()
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
