"""
Fits binary source model using EMCEE sampler.

The code simulates binary source light curve and fits the model twice:
with source flux ratio found via linear regression and
with source flux ratio as a chain parameter.
"""
import sys
try:
    import emcee
    import corner
except ImportError as err:
    print(err)
    print("\nEMCEE or corner could not be imported.")
    print("Get it from: http://dfm.io/emcee/current/user/install/")
    print("and re-run the script")
    sys.exit(1)

from astropy.table import Table, Column
from itertools import chain
import MulensModel as mm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import multiprocessing
import numpy as np
import os
import scipy.optimize as op
import yaml
import warnings
import split_data_for_binary_lens as split

def read_data(path, phot_settings, plot=False):
    """Read a catalogue or list of catalogues and creates MulensData instance

    Args:
        path (str): directory of the Python script and catalogues
        phot_settings (dict): photometry settings from the yaml file
        plot (bool, optional): Plot the catalogues or not. Defaults to False.

    Raises:
        RuntimeError: if photometry files(s) are not available.

    Returns:
        tuple of lists: list of data instances and filenames to be looped
    """

    filenames, subtract, phot_fmt = phot_settings.values()
    if os.path.isdir(f"{path}/{filenames}"):
        data_list = []
        for fname in sorted(os.listdir(f'{path}/{filenames}')):
            tab = Table.read(f'{path}/{filenames}/{fname}', format='ascii')
            dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
            dataset[0] = dataset[0]-2450000 if subtract else dataset[0]
            data_list.append(mm.MulensData(dataset, phot_fmt=phot_fmt))
        filenames = sorted(os.listdir(f'{path}/{filenames}'))

    elif os.path.isfile(f"{path}/{filenames}"):
        tab = Table.read(f"{path}/{filenames}", format='ascii')
        dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
        dataset[0] = dataset[0]-2450000 if subtract else dataset[0]
        data_list = [mm.MulensData(dataset, phot_fmt=phot_fmt)]
        filenames = [filenames.split('/')[1]]

    else:
        raise RuntimeError(f'Photometry file(s) {filenames} not available.')

    if plot:
        plt.figure(tight_layout=True)
        for dataset in data_list:
            dataset.plot(phot_fmt=phot_fmt, alpha=0.5)
        plt.gca().set(**{'xlabel': 'Time', 'ylabel': phot_fmt})
        plt.show()

    return data_list, filenames

def get_initial_t0_u0(data, settings, t_brightest=0.):
    """
    _summary_

    Args:
        data (_type_): _description_
        t_brightest (_type_, optional): _description_. Defaults to 0..

    Returns:
        _type_: _description_
    """

    if t_brightest == 0.:
        t_brightest = np.mean(data.time[np.argsort(data.mag)][:10])
    if 't_E' in settings['fit_constraints']['ln_prior'].keys():
        t_E_prior = settings['fit_constraints']['ln_prior']['t_E']
        t_E_init = round(np.exp(float(t_E_prior.split()[1])), 1)
    else:
        t_E_init = 25.

    # Starting the baseline (360d or half-data window?)
    # t_window = [t_brightest - 180, t_brightest + 180] # 360d window
    t_diff = data.time - t_brightest
    t_window = [t_brightest + min(t_diff)/2., t_brightest + max(t_diff)/2.]
    t_mask = (data.time < t_window[0]) | (data.time > t_window[1])
    flux_base = np.median(data.flux[t_mask])
    flux_mag_base = (flux_base, np.std(data.flux[t_mask]),
                     np.median(data.mag[t_mask]), np.std(data.mag[t_mask]))

    # Get the brightest flux around t_brightest (to avoid outliers)
    idx_min = np.argmin(abs(t_diff))
    min_, max_ = max(idx_min-5, 0), min(idx_min+6, len(data.mag))
    # mag_peak = min(data.mag[min_:max_]) # [idx_min-5:idx_min+6]
    flux_peak = max(data.flux[min_:max_])

    # Compute magnification and corresponding u_0(A)
    magnif_A = flux_peak / flux_base
    u_init = round(np.sqrt(2*magnif_A / np.sqrt(magnif_A**2 - 1) - 2), 3)
    start = {'t_0': round(t_brightest, 1), 'u_0': u_init, 't_E': t_E_init}

    return start, flux_mag_base

def make_all_fittings(data, settings, pdf=""):
    '''
    Missing description for this function...
    '''
    # 1st fit: Fitting a PSPL/1L1S without parallax...
    start, f_base = get_initial_t0_u0(data, settings)
    n_emcee = settings['fitting_parameters']
    fix = None if n_emcee['fix_blend'] is False else {data: n_emcee['fix_blend']}
    ev_st = mm.Event(data, model=mm.Model(start), fix_blend_flux=fix)
    settings['123_fits'] = '1st fit'
    output = fit_EMCEE(start, n_emcee['sigmas'][0], ln_prob, ev_st, settings)
    xlim = get_xlim2(output[0], data, n_emcee)
    event, cplot = make_plots(output[:-1], n_emcee, data, xlim, pdf=pdf)

    # Subtracting light curve from first fit
    model = mm.Model(dict(list(output[0].items())[:3]))
    aux_event = mm.Event(data, model=model, fix_blend_flux=fix)
    (flux, blend) = aux_event.get_flux_for_dataset(0)
    fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
    subt_data = [data.time[fsub > 0], fsub[fsub > 0], data.err_flux[fsub > 0]]
    subt_data = mm.MulensData(subt_data, phot_fmt='flux')

    # 2nd fit: PSPL to the subtracted data
    t_brightest = np.mean(subt_data.time[np.argsort(subt_data.mag)][:10])
    start = {'t_0': round(t_brightest,1), 'u_0':0.1, 't_E': output[0]['t_E']}
    fix = None if n_emcee['fix_blend'] is False else {subt_data: n_emcee['fix_blend']}
    ev_st = mm.Event(subt_data, model=mm.Model(start), fix_blend_flux=fix)
    settings['123_fits'], settings['xlim'] = '2nd fit', xlim
    output_1 = fit_EMCEE(start, n_emcee['sigmas'][0], ln_prob, ev_st, settings)
    two_pspl = (output[0], output_1[0])
    try:
        make_plots(output_1[:-1], n_emcee, subt_data, xlim, data, pdf=pdf)
        t_range = [min(data.time) - 500, max(data.time) + 500]
        # if output_1[-1]['u_0'][2] > 20:  # fix that 15? 20?
        # if output_1[0]['u_0'] > 5:
        if output_1[0]['u_0'] > 5 and output_1[-1]['u_0'][2] > 20:
            return output + (event, two_pspl), cplot, xlim
        elif output_1[0]['t_0'] < t_range[0] or output_1[0]['t_0'] > t_range[1]:
            return output + (event, two_pspl), cplot, xlim
    except ValueError:
        return output + (event, two_pspl), cplot, xlim

    # Third fit: 1L2S, source flux ratio not set yet (regression)
    start = {'t_0_1': output[0]['t_0'], 'u_0_1': output[0]['u_0'], 't_0_2':
             output_1[0]['t_0'], 'u_0_2': output_1[0]['u_0'], 't_E': 25}
    ev_st = mm.Event(data, model=mm.Model(start))
    settings['123_fits'] = '3rd fit'
    output_2 = fit_EMCEE(start, n_emcee['sigmas'][1], ln_prob, ev_st, settings)
    event_2, cplot_2 = make_plots(output_2[:-1], n_emcee, data, xlim, pdf=pdf)
    
    if max(output_2[0]['u_0_1'], output_2[0]['u_0_2']) > 3.:
    # if max(output_2[-1]['u_0_1'][2], output_2[-1]['u_0_2'][2]) > 3.:
    # if max(output_2[-1]['u_0_1'][1], output_2[-1]['u_0_2'][1]) > 4.:
        return output + (event, two_pspl), cplot, xlim        
    return output_2 + (event_2, two_pspl), cplot_2, xlim

def fit_utils(method, data, settings, model=''):

    n_emcee = settings['fitting_parameters']
    fix = None if n_emcee['fix_blend'] is False else {data: n_emcee['fix_blend']}

    if method == 'scipy_minimize':
        fix = {data: 0.}
        start = get_initial_t0_u0(data, settings)[0]
        ev_st = mm.Event(data, model=mm.Model(start), fix_blend_flux=fix)
        x0, arg = list(start.values()), (list(start.keys()), ev_st)
        r_ = op.minimize(split.chi2_fun, x0=x0, args=arg, method='Nelder-Mead')
        model = {'t_0': r_.x[0], 'u_0': r_.x[1], 't_E': r_.x[2]}

        return model

    elif method == 'get_t_E_1L2S':
        aux_event = mm.Event(data, model=mm.Model(model))
        x0, arg = [model['t_E']], (['t_E'], aux_event)
        r_ = op.minimize(split.chi2_fun, x0=x0, args=arg, method='Nelder-Mead')
        model['t_E'] = r_.x[0]
        return list(model.values())

    elif method == 'subt_data':
        aux_event = mm.Event(data, model=mm.Model(model), fix_blend_flux=fix)
        (flux, blend) = aux_event.get_flux_for_dataset(0)
        fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
        subt_data = np.c_[data.time, fsub, data.err_flux][fsub > 0]
        subt_data = mm.MulensData(subt_data.T, phot_fmt='flux')

        return subt_data

    else:
        raise Exception('Invalid method sent to function fit_utils().')

def prefit_split_and_fit(data, settings, pdf=""):
    """
    Raphael: I started doing the split before the result. Radek thinks the
    scipy argrelextrema is not general for all cases.

    Radek suggestion (06 Nov): Do PSPL quickly with scipy.optimize or
    chi2_gradient (exs 02, 09) on the original data, then do quick 1L2S(?),
    then get the minimum between peaks and split to start the process...
    In this case, I will test my code and try the fits, then apply his method
    just to check both results. In Radek's case, I would need to...

    Args:
        data (_type_): _description_
        settings (_type_): _description_
        pdf (str, optional): _description_. Defaults to "".
    """

    ### My method ::: 02-06/nov
    # start, fm_base = get_initial_t0_u0(data, settings)
    # data_left_right = split.split_before_result(data, fm_base[0], fm_base[1])
    # two_pspl = split.fit_PSPL_twice(data_left_right, settings, start=start)

    ### Radek's method ::: 06/nov on...
    model_1 = fit_utils('scipy_minimize', data, settings)
    data_2_subt = fit_utils('subt_data', data, settings, model_1)
    fm_base = get_initial_t0_u0(data_2_subt, settings)[1]  # above 3sigma?
    model_2 = fit_utils('scipy_minimize', data_2_subt, settings)
    #
    n_emcee = settings['fitting_parameters']
    start = {'t_0_1': model_1['t_0'], 'u_0_1': model_1['u_0'], 't_0_2':
             model_2['t_0'], 'u_0_2': model_2['u_0'], 't_E': 25}
    # start = {'t_0_1': model_2['t_0'], 'u_0_1': model_2['u_0'], 't_0_2':
    #          model_1['t_0'], 'u_0_2': model_1['u_0'], 't_E': 25} # inverted!!!
    ev_st = mm.Event(data, model=mm.Model(start))
    # result = op.minimize(split.chi2_fun, x0=list(start.values()),
    #                      args=(list(start.keys()), ev_st), method='Nelder-Mead')
    output = fit_EMCEE(start, n_emcee['sigmas'][1], ln_prob, ev_st, settings)
    bst = dict(b_ for b_ in list(output[0].items()) if 'flux' not in b_[0])
    fix_source = {data: [output[0][key] for key in output[0] if 'source' in key]}
    event_1L2S = mm.Event(data, model=mm.Model(bst),
                          fix_source_flux=fix_source,
                          fix_blend_flux={data: output[0]['blending_flux']})
    event_1L2S.get_chi2()
    data_left_right = split.split_after_result(event_1L2S, output)

    # Fits 2xPSPL if data is good (u_0 < 3.)...
    start = get_initial_t0_u0(data, settings)[0]
    two_pspl, subt_data = split.fit_PSPL_twice(data_left_right, settings, start=start)

    # Plotting 2 PSPL
    output_1, output_2 = two_pspl
    # output_1, output_2, output_3 = two_pspl
    data_1_subt, data_2_subt = subt_data
    xlim = get_xlim2(output_1[0], data, n_emcee)
    make_plots(output[:-1], n_emcee, data, xlim)  # TESTING HERE...
    settings['xlim'] = xlim
    event_1, cplot_1 = make_plots(output_1[:-1], n_emcee, data_1_subt, xlim,
                                  data, pdf=pdf)
    # event_1, cplot_1 = make_plots(output_1[:-1], n_emcee, data_left_right[0],
    #                               xlim, data, pdf=pdf)
    make_plots(output_2[:-1], n_emcee, data_2_subt, xlim, data, pdf=pdf)
    # make_plots(output_3[:-1], n_emcee, data_1_subt, xlim, data, pdf=pdf)
    pdf.close()

    # Stopped here: 07.nov @ 15h42
    # [...]
    breakpoint()

    pass

def ln_like(theta, event, params_to_fit):
    """ likelihood function """
    for (param, theta_) in zip(params_to_fit, theta):
        # Here we handle fixing source flux ratio:
        if param == 'flux_ratio':
            # implemented for a single dataset
            # event.fix_source_flux_ratio = {my_dataset: theta_} # original: wrong?
            event.fix_source_flux_ratio = {event.datasets[0]: theta_}
        else:
            setattr(event.model.parameters, param, theta_)
    event.get_chi2()

    return -0.5 * event.chi2, event.fluxes[0]

def ln_prior(theta, event, params_to_fit, settings):
    """
    Apply all the priors (minimum, maximum and distributions)

    Args:
        theta (np.array): chain parameters to sample the likelihood/prior.
        event (mm.Event): Event instance containing the model and datasets.
        params_to_fit (list): name of the parameters to be fitted.
        settings (dict): all settings from yaml file.
        spec (str, optional): _description_. Defaults to "". (TO EDIT...)

    Raises:
        ValueError: if input prior is not implemented

    Returns:
        float: -np.inf or prior value to be added on the likelihood
    """

    stg_priors = settings['fit_constraints']
    stg_min_max = [settings['min_values'], settings['max_values']]
    if not isinstance(stg_min_max[1]['t_0'], list):
        raise ValueError('t_0 max_values should be of list type.')
    ln_prior_t_E, ln_prior_fluxes = 0., 0.

    # breakpoint()    # settings['init_params_1L2S'] to check t01 > t02 initially

    # Limiting min and max values (all minimum, then t_0 and u_0 maximum)
    for param in params_to_fit:
        if param[:3] in stg_min_max[0].keys():
            if theta[params_to_fit.index(param)] < stg_min_max[0][param[:3]]:
                return -np.inf
        if 't_0' in param and 't_0' in stg_min_max[1].keys():
            t_range = event.datasets[0].time[::len(event.datasets[0].time)-1]
            t_range = t_range + np.array(stg_min_max[1]['t_0'])
            if t_range[0] > theta[params_to_fit.index(param)] > t_range[1]:
                return -np.inf
        if 'u_0' in param and 'u_0' in stg_min_max[1].keys():
            if theta[params_to_fit.index(param)] > stg_min_max[1]['u_0']:
                return -np.inf
    
    # Prior in t_E (only lognormal so far, tbd: Mroz17/20)
    # OBS: Still need to remove the ignore warnings line (related to log?)
    if 't_E' in stg_priors['ln_prior'].keys():
        t_E = theta[params_to_fit.index('t_E')]
        if 'lognormal' in stg_priors['ln_prior']['t_E']:
            # if t_E >= 1.:
            warnings.filterwarnings("ignore", category=RuntimeWarning)  # bad!
            prior = [float(x) for x in stg_priors['ln_prior']['t_E'].split()[1:]]
            ln_prior_t_E = - (np.log(t_E) - prior[0])**2 / (2*prior[1]**2)
            ln_prior_t_E -= np.log(t_E * np.sqrt(2*np.pi)*prior[1])
        elif 'Mroz et al.' in stg_priors['ln_prior']['t_E']:
            raise ValueError('Still implementing Mroz et al. (2017, 2020) prior.')
        else:
            raise ValueError('t_E prior type not allowed.')

    # Avoiding negative source/blending fluxes (Radek)
    _ = event.get_chi2()
    if 'negative_blending_flux_sigma_mag' in stg_priors.keys():
        sig_ = stg_priors['negative_blending_flux_sigma_mag']
        for flux in [*event.source_fluxes[0], event.blend_fluxes[0]]:
            if flux < 0.:
                ln_prior_fluxes += -1/2 * (flux/sig_)**2  # 1000, 100 or less?
    elif 'no_negative_blending_flux' in stg_priors.keys():
        if event.blend_fluxes[0] < 0:
            return -np.inf
    
    # # Raphael's prior in min_flux (OLD CODE)
    # min_flux = min([min(event.source_fluxes[0]), event.blend_fluxes[0]])
    # # # min_flux = min(event.source_fluxes[0])
    # if min_flux < 0:
    #     # ln_prior_fluxes = - 1/2 * (np.log(abs(min_flux))/np.log(2))**2
    #     # ln_prior_fluxes += np.log(1/(np.sqrt(2*np.pi)*np.log(2)))
    #     # ln_prior_fluxes = - 1/2 * (min_flux/2)**2  # sigma = 2
    #     ln_prior_fluxes = - 1/2 * (min_flux/10)**2  # sigma = 0.1
    #     # breakpoint()  # min_flux -> abs(min_flux)
    #     # print(_)  # Radek: printing the chi2 value (144k -> 6000)
    # else:
    #     ln_prior_fluxes = 0.

    return 0.0 + ln_prior_t_E + ln_prior_fluxes

def ln_prob(theta, event, params_to_fit, settings):
    """ combines likelihood and priors"""
    ln_prior_ = ln_prior(theta, event, params_to_fit, settings)
    if not np.isfinite(ln_prior_):
        return -np.inf, np.array([-np.inf,-np.inf]), -np.inf
    ln_like_, fluxes = ln_like(theta, event, params_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
        return -np.inf, np.array([-np.inf,-np.inf]), -np.inf

    return ln_prior_ + ln_like_, fluxes[0], fluxes[1]

def fit_EMCEE(dict_start, sigmas, ln_prob, event, settings):
    """
    Fit model using EMCEE and print results.
    Arguments:
        params_to_fit - list of parameters (REMOVED?)
        dict_start - dict that specifies values of these parameters
        sigmas - list of sigma values used to find starting values
        ln_prob - function returning logarithm of probability
        event - MulensModel.Event instance
        n_emcee - Dictionary with number of walkers, steps, burn-in
        n_walkers - number of walkers in EMCEE -> inactive
        n_steps - number of steps per walker   -> inactive
        n_burn - number of steps considered as burn-in ( < n_steps)  -> inactive
    """
    params_to_fit, mean = list(dict_start.keys()), list(dict_start.values())
    n_dim, sigmas = len(params_to_fit), np.array(sigmas)
    n_emcee = settings['fitting_parameters']
    nwlk, nstep, nburn = n_emcee['nwlk'], n_emcee['nstep'], n_emcee['nburn']
    emcee_args = (event, params_to_fit, settings)
    x_fit = settings['123_fits'] if '123_fits' in settings.keys() else '3rd fit'
    term = ['PSPL to original', 'PSPL to subtracted', '1L2S to original']
    print(f'\n\033[1m -- {x_fit}: {term[int(x_fit[0])-1]} data...\033[0m')

    # Doing the 1L2S fitting in two steps (or all? best in 1st and 3rd fits)
    # if x_fit in ['1st fit', '3rd fit']:
    if x_fit in ['3rd fit']:
        mean = fit_utils('get_t_E_1L2S', data, settings, model=dict_start)  ## NEW LINE!!!
        settings['init_params_1L2S'] = mean
        start = [mean + np.random.randn(n_dim)*10*sigmas for i in range(nwlk)]
        start = abs(np.array(start))
        sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, args=emcee_args)
        sampler.run_mcmc(start, int(nstep/2), progress=n_emcee['progress'])
        samples = sampler.chain[:, int(nburn/2):, :].reshape((-1, n_dim))
        mean = np.percentile(samples, 50, axis=0)
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
    blobs = sampler.get_blobs()[nburn:].T.flatten() # [:10]
    source_fluxes = np.array(list(chain.from_iterable(blobs))[::2])
    blend_flux = np.array(list(chain.from_iterable(blobs))[1::2])
    prob = sampler.lnprobability[:, nburn:].reshape((-1))
    if len(params_to_fit) == 3:
        samples = np.c_[samples, source_fluxes[:,0], blend_flux, prob]
    elif len(params_to_fit) == 5:
        samples = np.c_[samples, source_fluxes[:,0], source_fluxes[:,1],
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
    best = samples[best_idx,:-1] if n_emcee['ans'] == 'max_prob' else perc[1]
    best = dict(zip(params_to_fit, best))
    for (key, value) in zip(params_to_fit, best.values()):
        if key == 'flux_ratio':
            event.fix_source_flux_ratio = {event.datasets[0]: value}
        else:
            setattr(event.model.parameters, key, value)
    print("\nSmallest chi2 model:")
    print(*[repr(b) if isinstance(b, float) else b.value for b in best.values()])
    deg_of_freedom = event.datasets[0].n_epochs - n_dim
    print(f"chi2 = {event.chi2:.8f}, dof = {deg_of_freedom}")

    # Cleaning posterior and cplot if required by the user
    if n_emcee['clean_cplot']:
        new_states, new_perc = clean_posterior_emcee(sampler, best, nburn)
        if new_states is not None:
            pars_perc = dict(zip(params_to_fit, new_perc.T))
            samples = new_states
    
    # return best, pars_best, event.get_chi2(), states, sampler    
    return best, sampler, samples, pars_perc #, samples, sampler

def clean_posterior_emcee(sampler, params, n_burn):
    """
    Manipulate emcee chains to reject stray walkers and clean posterior
    Arguments:
        sampler - ensemble sampler from EMCEE
        ## params - set of best parameters from fit_EMCEE function -> deactivate
        n_burn - number of steps considered as burn-in ( < n_steps)
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
    tau = sampler.get_autocorr_time(tol=0)
    acc = sampler.acceptance_fraction
    w = np.where(abs(acc-np.median(acc)) < min(0.1,3*np.std(acc)))

    if len(w[0]) == 0:
        return None, None
    # states = sampler.chain[w[0],burnin::thin,:].copy()  # with thinning
    new_states = sampler.chain[w[0],n_burn::,:].copy()  # no thinning
    gwlk, nthin, npars = np.shape(new_states)
    new_states = np.reshape(new_states,[gwlk*nthin, npars])

    # Trying to add the source_fluxes after flattening...
    blobs = sampler.get_blobs().T # [:10]
    blobs = blobs[w[0],n_burn:].reshape(-1)
    source_fluxes = np.array(list(chain.from_iterable(blobs))[::2])
    blend_flux = np.array(list(chain.from_iterable(blobs))[1::2])
    prob = sampler.lnprobability[w[0], n_burn:].reshape(-1)
    if npars == 3:
        new_states = np.c_[new_states, source_fluxes[:,0], blend_flux, prob]
    elif npars == 5:
        new_states = np.c_[new_states, source_fluxes[:,0], source_fluxes[:,1],
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

    return new_states, np.quantile(new_states,[0.16,0.50,0.84],axis=0)

def make_plots(results_states, n_emcee, dataset, xlim, orig_data=[], pdf=""):
    """
    plot results
    """
    best, sampler, states = results_states
    condition = (n_emcee['fix_blend'] is not False) and (len(best) != 8)
    c_states = states[:,:-2] if condition else states[:,:-1]
    params = list(best.keys())[:-1] if condition else list(best.keys())
    values = list(best.values())[:-1] if condition else list(best.values())
    tracer_plot(params, sampler, n_emcee['nburn'], pdf=pdf)
    cplot = corner.corner(c_states, quantiles=[0.16,0.50,0.84], labels=params,
                          truths=values, show_titles=True)
    if pdf:
        pdf.savefig(cplot)
    else:
        plt.show()
    event = plot_fit(best, dataset, n_emcee, xlim, orig_data, pdf=pdf)

    return event, cplot

def tracer_plot(params_to_fit, sampler, nburn, pdf=""):
    """
    Plot tracer plots (or walkers' time series)
    """
    npars = sampler.ndim
    fig, axes = plt.subplots(npars, 1, sharex=True, figsize=(10,10))
    for i in range(npars):
        axes[i].plot(np.array(sampler.chain[:,:,i]).T,rasterized=True)
        axes[i].axvline(x=nburn, ls='--', color='gray', lw=1.5)
        axes[i].set_ylabel(params_to_fit[i], fontsize=16)
    axes[npars-1].set_xlabel(r'steps', fontsize=16)
    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
    else:
        plt.show()

def get_xlim2(best, data, n_emcee):

    # only works for PSPL case... (A' should be considered for 1L2S)
    # Amax = (best['u_0']**2 + 2) / (best['u_0']*np.sqrt(best['u_0']**2 + 4))

    # Radek: using get_data_magnification from MulensModel
    bst = dict(item for item in list(best.items()) if 'flux' not in item[0])
    fix = None if n_emcee['fix_blend'] is False else {data: n_emcee['fix_blend']}
    event = mm.Event(data, model=mm.Model(bst), fix_blend_flux=fix)
    event.get_flux_for_dataset(0)
    Amax = max(event.fits[0].get_data_magnification())
    dividend = best['source_flux']*Amax + best['blending_flux']
    divisor = best['source_flux'] + best['blending_flux']
    deltaI = 2.5*np.log10(dividend/divisor)  # deltaI ~ 3 for PAR-46 :: OK!

    # Get the magnitude at the model peak (mag_peak ~ comp? ok)
    idx_peak = np.argmin(abs(data.time-best['t_0']))
    model_mag = event.fits[0].get_model_magnitudes()
    mag_peak, comp = model_mag[idx_peak], data.mag[idx_peak]

    # Summing 0.85*deltaI to the mag_peak, then obtain t_range (+2%)
    mag_baseline = mag_peak + 0.85*deltaI
    idx1 = np.argmin(abs(mag_baseline - model_mag[:idx_peak]))
    idx2 = idx_peak + np.argmin(abs(mag_baseline - model_mag[idx_peak:]))
    t_range = np.array([0.97*data.time[idx1], 1.03*data.time[idx2]])
    max_diff_t_0 = max(abs(t_range - best['t_0'])) + 100
    xlim = [best['t_0']-max_diff_t_0, best['t_0']+max_diff_t_0]

    if np.diff(xlim)[0] < 500:
        xlim = [best['t_0']-500, best['t_0']+500]
    
    return xlim

def plot_fit(best, data, n_emcee, xlim, orig_data=[], best_50=[], pdf=""):
    """
    Plot the best-fitting model(s) over the light curve in mag or flux.

    Args:
        best (dict): results from PSPL (3+2 params) or 1L2S (5+3 params).
        data (mm.MulensData instance): object containing all the data.
        n_emcee (dict): parameters relevant to emcee fitting.
        xlim (list): time interval to be plotted.
        orig_data (list, optional): Plot with subtracted data. Defaults to [].
        best_50 (list, optional): Additional percentile result. Defaults to [].
        pdf (str, optional): pdf file to save the plot. Defaults to "".

    Returns:
        mm.Event: final event containing the model and datasets.
    """

    fig = plt.figure(figsize=(7.5,5.5))
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :]) # or gs.new_subplotspec((0, 0), rowspan=2)
    bst = dict(item for item in list(best.items()) if 'flux' not in item[0])
    fix_source = {data: [best[key] for key in best if 'source' in key]}
    event = mm.Event(data, model=mm.Model(bst), fix_source_flux=fix_source,
                     fix_blend_flux={data: best['blending_flux']})
    data_label = "Original data" if not orig_data else "Subtracted data"
    event.plot_data(subtract_2450000=False, label=data_label)
    plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': False,
                   't_start': xlim[0], 't_stop': xlim[1], 'zorder': 10,
                   'color': 'black'}
    if orig_data:
        orig_data.plot(phot_fmt='mag', color='gray', alpha=0.2, label="Original data")

    label = 'PSPL' if event.model.n_sources==1 else '1L2S'
    label += f" ({n_emcee['ans']}):\n"
    for item in bst:
        label += f'{item} = {bst[item]:.2f}\n'
    event.plot_model(label=r"%s"%label[:-1], **plot_params)
    plt.tick_params(axis='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    
    ax2 = fig.add_subplot(gs[2:, :], sharex=ax1)
    event.plot_residuals(subtract_2450000=False, zorder=10) # fix zorder
    plt.tick_params(axis='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.xlim(xlim[0], xlim[1])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.axes(ax1)
    if len(best_50) > 0:
        if len(best_50) == 3:
            model_x = mm.Model({'t_0': best_50[0], 'u_0': best_50[1],
                                't_E': best_50[2]})
        elif len(best_50) == 5:
            model_x = mm.Model({'t_0_1': best_50[0], 'u_0_1': best_50[1],
                                't_0_2': best_50[2], 'u_0_2': best_50[3],
                                't_E': best_50[4]})
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

def write_tables(path, settings, name, result, fmt="ascii.commented_header"):
    """
    Save the chains, yaml results and table with results, according with the
    paths and other informations provided in the settings file.

    Args:
        path (str): directory of the Python script and catalogues
        settings (dict): all settings from yaml file
        name (str): name of the photometry file
        result (tuple): contains the EMCEE outputs and mm.Event instance
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
    acor = result[1].get_autocorr_time(quiet=True, discard=n_emcee['nburn'])
    deg_of_freedom = result[4].datasets[0].n_epochs - len(bst)
    lst = [np.mean(result[1].acceptance_fraction), np.mean(acor), '', '',
           result[4].chi2, deg_of_freedom, '', '']
    dict_perc_best = {2: perc, 3: perc_fluxes, 6: bst, 7: fluxes}

    # filling and writing the template
    for idx, dict_obj in dict_perc_best.items():
        for key, val in dict_obj.items():
            if idx in [2, 3]:
                uncerts = f'+{val[2]-val[1]:.5f}, -{val[1]-val[0]:.5f}'
                lst[idx] += f'    {key}: [{val[1]:.5f}, {uncerts}]\n'
            else:
                lst[idx] += f'    {key}: {val}\n'
        lst[idx] = lst[idx][:-1]
    with open(f'{path}/../1L2S-result_template.yaml') as file_:
        template_result = file_.read()
    if 'yaml output' in outputs.keys():
        yaml_fname =  outputs['yaml output']['file name'].format(name)
        with open(f'{path}/{yaml_fname}', 'w') as yaml_results:
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
        lst = bst_values+[0.,0.] if len(bst)==3 else bst_values
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
    path = os.path.dirname(os.path.realpath(sys.argv[1]))
    with open(sys.argv[1]) as in_data:
        settings = yaml.safe_load(in_data)

    data_list, filenames = read_data(path, settings['phot_settings'])
    for data, name in zip(data_list[5:6], filenames[5:6]):
        
        print(f'\n\033[1m * Running fit for {name}\033[0m')
        # breakpoint()
        pdf_dir = settings['plots']['all_plots']['file_dir']
        pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_result.pdf")
        # prefit_split_and_fit(data, settings, pdf=pdf)
        result, cplot, xlim = make_all_fittings(data, settings, pdf=pdf)
        res_event, two_pspl = result[4], result[5]
        pdf.close()
        write_tables(path, settings, name, result)

        pdf_dir = settings['plots']['triangle']['file_dir']
        pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_cplot.pdf")
        pdf.savefig(cplot)
        pdf.close()

        pdf_dir = settings['plots']['best model']['file_dir']
        pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_fit.pdf")
        plot_fit(result[0], data, settings['fitting_parameters'], xlim, pdf=pdf)
        pdf.close()

        # Split data into two and fit PSPL to get 2L1S initial params
        make_2L1S = settings['other_output']['yaml_files_2L1S']['t_or_f']
        if make_2L1S and res_event.model.n_sources == 2:
            data_left_right = split.split_after_result(res_event, result)
            if isinstance(data_left_right[0], list):
                continue
            two_pspl = split.fit_PSPL_twice(data_left_right, settings, result)
            split.generate_2L1S_yaml_files(path, two_pspl, name, settings)
        print("\n--------------------------------------------------")
    # breakpoint()
