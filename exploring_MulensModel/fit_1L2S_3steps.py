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

from astropy.table import Table
from itertools import chain
import MulensModel as mm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import multiprocessing
import numpy as np
# import os

def make_all_fittings(data, n_emcee, pdf=""):

    # 1st fit: Fitting a PSPL/1L1S without parallax...
    t_brightest = np.mean(data.time[np.argsort(data.mag)][:10])
    # still missing u(A) from baseline to get an initial u_0 !!!
    start = {'t_0': round(t_brightest, 1), 'u_0':0.1, 't_E': 25}
    fixed = {data: 0.} if n_emcee['fix_blend_flux'] else None
    event = mm.Event(datasets=data, model=mm.Model(start), fix_blend_flux=fixed)
    # sigmas = [1., 0.05, 1.]  # [1., 0.05, 1.] or [10., 0.5, 10.]
    print("\n\033[1m -- 1st fit: PSPL to original data...\033[0m")
    output = fit_EMCEE(start, n_emcee['sigmas'][0], ln_prob, event, n_emcee)
    xlim = get_xlim2(output[0], data, n_emcee)  # checking with Radek... OK
    cplot = make_plots(output[:-1], n_emcee, data, xlim, pdf=pdf)[1]

    # Subtracting light curve from first fit
    model = mm.Model(dict(list(output[0].items())[:3]))
    aux_event = mm.Event(model=model, datasets=data, fix_blend_flux=fixed) # repeated?
    (flux, blend) = aux_event.get_flux_for_dataset(0)
    fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
    subt_data = [data.time[fsub > 0], fsub[fsub > 0], data.err_flux[fsub > 0]]
    subt_data = mm.MulensData(subt_data, phot_fmt='flux')

    # 2nd fit: PSPL to the subtracted data
    t_brightest = np.mean(subt_data.time[np.argsort(subt_data.mag)][:10])
    start = {'t_0': round(t_brightest,1), 'u_0':0.1, 't_E': output[0]['t_E']}
    fixed = {subt_data: 0.} if n_emcee['fix_blend_flux'] else None
    event = mm.Event(subt_data, model=mm.Model(start), fix_blend_flux=fixed)
    # sigmas = [1., 0.05, 1.]  # [1., 0.05, 1.] or [10., 0.5, 10.]
    print("\n\033[1m -- 2nd fit: PSPL to subtracted data...\033[0m")
    output_1 = fit_EMCEE(start, n_emcee['sigmas'][1], ln_prob, event, n_emcee,
                         spec="u_0")
    make_plots(output_1[:-1], n_emcee, subt_data, xlim, data, pdf=pdf)
    if output_1[-1]['u_0'][2] > 20:  # fix that 15? 20?
    # if output_1[0]['u_0'] > 5:
        return (output[0], output[2], event), cplot, xlim

    # Third fit: 1L2S, source flux ratio not set yet (regression)
    start = {'t_0_1': output[0]['t_0'], 'u_0_1': output[0]['u_0'], 't_0_2':
             output_1[0]['t_0'], 'u_0_2': output_1[0]['u_0'], 't_E': 25}
    event = mm.Event(datasets=data, model=mm.Model(start))
    # # sigmas = [0.1, 0.05, 0.1, 0.01, 0.1]
    # sigmas = [0.1, 0.01, 0.1, 0.01, 0.1] # [1., 0.1, 1., 0.1, 1.]
    print("\n\033[1m -- 3rd fit: 1L2S to original data...\033[0m")
    output_2 = fit_EMCEE( start, n_emcee['sigmas'][2], ln_prob, event, n_emcee)
    event_2, cplot_2 = make_plots(output_2[:-1], n_emcee, data, xlim, pdf=pdf)
    
    # if max(output_2[0][1], output_2[0][3]) > 2.9:     ### or after cleaning chains...
    # if max(output_2[-1]['u_0_1'][2], output_2[-1]['u_0_2'][2]) > 3.:
    if max(output_2[-1]['u_0_1'][1], output_2[-1]['u_0_2'][1]) > 3.:
        return (output[0], output[2], event), cplot, xlim
    
    return (output_2[0], output_2[0], event_2), cplot_2, xlim

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

    return -0.5 * event.get_chi2()

def ln_prior(theta, event, params_to_fit, spec=""):
    """priors - we only reject obviously wrong models"""
    for param in ['t_E', 'u_0', 'u_0_1', 'u_0_2']:
        if param in params_to_fit:
            if theta[params_to_fit.index(param)] < 0.:
                return -np.inf

    # Additional priors distributions:
    t_range = [min(event.datasets[0].time), max(event.datasets[0].time)]
    if spec:    # 15, 100, 1000 or nothing?
        # if theta[params_to_fit.index('u_0')] > 15. or \
        if theta[params_to_fit.index('t_0')] < t_range[0]-100 or \
            theta[params_to_fit.index('t_0')] > t_range[1]+100:
            return -np.inf
    t_E = theta[params_to_fit.index('t_E')]
    sigma = 2 if len(params_to_fit) == 3 else 5
    ln_prior_t_E = - (np.log(t_E) - np.log(25))**2 / (2*np.log(sigma)**2)
    ln_prior_t_E += np.log(1/(np.sqrt(2*np.pi)*np.log(sigma)))

    # Trying to limit negative source/blending fluxes:
    _ = event.get_chi2()
    # if min(event.source_fluxes[0]) < -50000 or event.blend_fluxes[0] < -50000:
    #     return -np.inf
    # if max(event.source_fluxes[0]) > 10000:
    #     return -np.inf
    
    # Radek's prior in fluxes
    ln_prior_fluxes = 0
    # for flux in [*event.source_fluxes[0], event.blend_fluxes[0]]:
    #    if flux < 0.:
    #     #    ln_prior_fluxes += - 1/2 * (flux/1000)**2
    #        ln_prior_fluxes += -1/2 * (flux/10)**2 # 1000* or 100 or less...
    
    # Raphael's prior in min_flux
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

def ln_prob(theta, event, params_to_fit, spec=""):
    """ combines likelihood and priors"""
    ln_prior_ = ln_prior(theta, event, params_to_fit, spec)
    if not np.isfinite(ln_prior_):
        return -np.inf, np.array([-np.inf,-np.inf]), -np.inf
    ln_like_ = ln_like(theta, event, params_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
        return -np.inf, np.array([-np.inf,-np.inf]), -np.inf

    return ln_prior_ + ln_like_, event.source_fluxes[0], event.blend_fluxes[0]

def fit_EMCEE(dict_start, sigmas, ln_prob, event, n_emcee, spec=""):
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
    nwlk, nstep, nburn = n_emcee['nwlk'], n_emcee['nstep'], n_emcee['nburn']

    # Doing the 1L2S fitting in two steps (or all? best in 1st and 3rd fits)
    if not spec: # n_dim == 5:
        start = [mean + np.random.randn(n_dim)*10*sigmas for i in range(nwlk)]
        start = abs(np.array(start))
        sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob,
                                        args=(event, params_to_fit, spec))
        sampler.run_mcmc(start, int(nstep/2), progress=n_emcee['tqdm'])
        samples = sampler.chain[:, int(nburn/2):, :].reshape((-1, n_dim))
        mean = np.percentile(samples, 50, axis=0)
        # prob_temp = sampler.lnprobability[:, int(nburn/2):].reshape((-1))
        # mean = samples[np.argmax(prob_temp)]
    start = [mean + np.random.randn(n_dim) * sigmas for i in range(nwlk)]
    start = abs(np.array(start))

    # Run emcee (this can take some time):
    blobs = [('source_fluxes', list), ('blend_fluxes', float)]
    sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, blobs_dtype=blobs,
                                    args=(event, params_to_fit, spec))
                                    # backend=backend)
    # sampler = emcee.EnsembleSampler(
    #     nwlk, n_dim, ln_prob,
    #     moves=[(emcee.moves.DEMove(),0.8),(emcee.moves.DESnookerMove(),0.2)],
    #     args=(event, params_to_fit, spec))
    sampler.run_mcmc(start, nstep, progress=n_emcee['tqdm'])
    
    # Setting up multi-threading (std: fork in Linux, spawn in Mac)
    # multiprocessing.set_start_method("fork", force=True)
    # os.environ["OMP_NUM_THREADS"] = "1"
    # with multiprocessing.Pool() as pool:
    #     sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, pool=pool,
    #                                     args=(event, params_to_fit, spec))
    #     sampler.run_mcmc(start, nstep, progress=n_emcee['tqdm'])
    # pool.close()

    # Remove burn-in samples and reshape:
    samples = sampler.chain[:, nburn:, :].reshape((-1, n_dim))
    blobs = sampler.get_blobs()[nburn:].reshape(-1) # [:10]
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
    print("chi2 =", event.get_chi2())

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
    condition = (n_emcee['fix_blend_flux'] and len(best) != 8)
    c_states = states[:,:-2] if condition else states[:,:-1]
    params = list(best.keys())[:-1] if condition else list(best.keys())
    values = list(best.values())[:-1] if condition else list(best.values())
    # params, values = list(best.keys()), list(best.values())
    tracer_plot(params, sampler, n_emcee['nburn'], pdf=pdf)
    # if len(best) == 8:
    #     cplot = corner.corner(states[:,:-1], labels=params, truths=values,
    #                           quantiles=[0.16,0.50,0.84], show_titles=True)
    # else:
    #     cplot = corner.corner(states[:,:-2], labels=params[:-1], truths=values[:-1],
    #                           quantiles=[0.16,0.50,0.84], show_titles=True)
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

def get_xlim(best, dataset, t_brightest=0., prev={}):

    if len(best)==5+3:
        xlim = sorted([best['t_0_1'], best['t_0_2']])
        return [xlim[0]-3*best['t_E'], xlim[1]+3*best['t_E']]

    if prev:
        xlim = [(prev['t_0']+best['t_0'])/2 - 2.5*abs(prev['t_0']-best['t_0']),
                (prev['t_0']+best['t_0'])/2 + 2.5*abs(prev['t_0']-best['t_0'])]
    else:
        xlim = [best['t_0'] - 3*best['t_E'], best['t_0'] + 3*best['t_E']]
        if xlim[0] < min(dataset.time) and t_brightest:
            xlim = [best['t_0'] - 2*abs(best['t_0'] - t_brightest),
                    best['t_0'] + 2*abs(best['t_0'] - t_brightest)]
    
    # if np.diff(xlim)[0] < 250:
    #     xlim = [best['t_0'] - 500, best['t_0']+500]

    # Obs: Still doesn't cover the PSPL/1L2S cases where t_E is too low/high...       
    return xlim

def get_xlim2(best, dataset, n_emcee):

    # only works for PSPL case... (A' should be considered for 1L2S)
    # Amax = (best['u_0']**2 + 2) / (best['u_0']*np.sqrt(best['u_0']**2 + 4))

    # Radek: using get_data_magnification from MulensModel
    bst = dict(item for item in list(best.items()) if 'flux' not in item[0])
    fixed = {dataset: 0.} if n_emcee['fix_blend_flux'] else None
    event = mm.Event(model=mm.Model(bst), datasets=[dataset], fix_blend_flux=fixed)
    event.get_flux_for_dataset(0)
    Amax = max(event.fits[0].get_data_magnification())
    dividend = best['source_flux']*Amax + best['blending_flux']
    divisor = best['source_flux'] + best['blending_flux']
    deltaI = 2.5*np.log10(dividend/divisor)  # deltaI ~ 3 for PAR-46 :: OK!

    # Get the magnitude at the model peak (mag_peak ~ comp? ok)
    idx_peak = np.argmin(abs(dataset.time-best['t_0']))
    model_mag = event.fits[0].get_model_magnitudes()
    mag_peak, comp = model_mag[idx_peak], dataset.mag[idx_peak]

    # Summing 0.85*deltaI to the mag_peak, then obtain t_range (+2%)
    mag_baseline = mag_peak + 0.85*deltaI
    idx1 = np.argmin(abs(mag_baseline - model_mag[:idx_peak]))
    idx2 = idx_peak + np.argmin(abs(mag_baseline - model_mag[idx_peak:]))
    t_range = np.array([0.97*dataset.time[idx1], 1.03*dataset.time[idx2]])
    max_diff_t_0 = max(abs(t_range - best['t_0'])) + 100
    xlim = [best['t_0']-max_diff_t_0, best['t_0']+max_diff_t_0]

    if np.diff(xlim)[0] < 500:
        xlim = [best['t_0']-500, best['t_0']+500]
    
    return xlim

def plot_fit(best, dataset, n_emcee, xlim, orig_data=[], best_50=[], pdf=""):

    fig = plt.figure(figsize=(7.5,5.5))
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :]) # or gs.new_subplotspec((0, 0), rowspan=2)
    best = dict(item for item in list(best.items()) if 'flux' not in item[0])
    if best == 5:
        event = mm.Event(model=mm.Model(best), datasets=[dataset])
    else:
        fixed = {dataset: 0.} if n_emcee['fix_blend_flux'] else None
        event = mm.Event(model=mm.Model(best), datasets=[dataset],
                         fix_blend_flux=fixed)
    data_label = "Original data" if not orig_data else "Subtracted data"
    event.plot_data(subtract_2450000=False, label=data_label)
    plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': False,
                   't_start': xlim[0], 't_stop': xlim[1], 'zorder': 10,
                   'color': 'black'}
    if orig_data:
        orig_data.plot(phot_fmt='mag', color='gray', alpha=0.2, label="Original data")

    label = 'PSPL' if event.model.n_sources==1 else '1L2S'
    label += f" ({n_emcee['ans']}):\n"
    for item in best:
        label += f'{item} = {best[item]:.2f}\n'
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
        event_x = mm.Event(model=model_x, datasets=[dataset])
        plot_params['color'] = 'orange'
        event_x.plot_model(label='50th_perc', **plot_params)

    ax1.legend(loc='best')
    if pdf:
        pdf.savefig(fig)
        plt.close('all')
    else:
        plt.show()
    return event
