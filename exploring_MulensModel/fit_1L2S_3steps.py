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

def make_all_fittings(my_dataset, n_emcee, pdf=""):

    # 1st fit: Fitting a PSPL/1L1S without parallax...
    mag_ids = np.argsort(my_dataset.mag)
    t_brightest = np.mean(my_dataset.time[mag_ids][:10])
    start = {'t_0': round(t_brightest, 1), 'u_0':0.1, 't_E': 20}
    my_event = mm.Event(datasets=my_dataset, model=mm.Model(start))
    params_to_fit = ['t_0', 'u_0', 't_E']
    sigmas = [1., 0.05, 1.]
    print("\n\033[1m -- First fit: PSPL to raw data...\033[0m")
    nburn = n_emcee['nburn']
    output = fit_EMCEE(params_to_fit, start, sigmas, ln_prob, my_event,
                       n_emcee)
    best, pars_quant, samples, states, sampler = output
    model_0 = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]})
    event_0 = mm.Event(model=model_0, datasets=[my_dataset])    # repeated!!!

    # Subtracting light curve from first fit
    (flux, blend) = event_0.get_flux_for_dataset(0)
    flux_subt = my_dataset.flux - event_0.fits[0].get_model_fluxes() + flux + blend
    # subtracted_data = [my_dataset.time, flux_subt, my_dataset.err_flux]
    subtracted_data = [my_dataset.time[flux_subt > 0], flux_subt[flux_subt > 0],
                       my_dataset.err_flux[flux_subt > 0]]
    my_dataset_2 = mm.MulensData(subtracted_data, phot_fmt='flux')

    mag_ids = np.argsort(my_dataset_2.mag)
    t_brightest = np.mean(my_dataset_2.time[mag_ids][:10])
    lims = [best[0] - 3*best[2], best[0] + 3*best[2]]
    if lims[0] < min(my_dataset.time):
        lims = [best[0] - 2*abs(best[0] - t_brightest),
                best[0] + 2*abs(best[0] - t_brightest)]
    labels = ["no pi_E, max_prob", "no pi_E, 50th_perc"]
    event_0, cplot = make_three_plots(params_to_fit, sampler, samples, states, nburn,
                                      best, my_dataset, labels, lims, pdf=pdf)

    # 2nd fit: PSPL to the subtracted data
    start = {'t_0': round(t_brightest,1), 'u_0':0.1, 't_E': best[2]}
    my_event = mm.Event(datasets=my_dataset_2, model=mm.Model(start))
    params_to_fit = ['t_0', 'u_0', 't_E']
    sigmas = [1., 0.05, 1.]
    print("\n\033[1m -- Second fit: PSPL to subtracted data.\033[0m")
    output = fit_EMCEE(params_to_fit, start, sigmas, ln_prob, my_event,
                       n_emcee, spec="u_0")
    best_1, pars_quant_1, samples, states_1, sampler_1 = output
    # if np.quantile(states_1, 0.84, axis=0)[1] > 15: # fix that 15?
    if pars_quant_1['u_0'][2] > 15:
        return event_0, best, cplot
    lims = [np.mean([best[0],best_1[0]]) - 2.5*abs(best[0]-best_1[0]),
            np.mean([best[0],best_1[0]]) + 2.5*abs(best[0]-best_1[0])]
    event_1, cplot = make_three_plots(params_to_fit, sampler_1, samples, states_1,
                                      nburn, best_1, my_dataset_2, labels, lims,
                                      my_dataset, pdf=pdf)

    # Third fit: 1L2S, source flux ratio not set yet (regression)
    start = {'t_0_1': best[0], 'u_0_1': best[1], 't_0_2': best_1[0],
              'u_0_2': best_1[1], 't_E': best_1[2]}
    my_event = mm.Event(datasets=my_dataset, model=mm.Model(start))
    # params['flux_ratio'] = 1 # 0.02
    params_to_fit = ["t_0_1", "u_0_1", "t_0_2", "u_0_2", "t_E"] #, "flux_ratio"]
    sigmas = [0.1, 0.05, 0.1, 0.01, 0.1] #, 0.001]
    print("\n\033[1m -- Third fit: 1L2S to original data.\033[0m")
    output = fit_EMCEE(params_to_fit, start, sigmas, ln_prob, my_event,
                       n_emcee)
    best_2, pars_quant_2, samples, states_2, sampler_2 = output
    labels = ["1L2S, max_prob", "1L2S, 50th_perc"]
    lims = sorted([best_2[0], best_2[2]])
    lims = [lims[0]-3*best_2[4], lims[1]+3*best_2[4]]
    event_2, cplot = make_three_plots(params_to_fit, sampler_2, samples, states_2,
                                      nburn, best_2, my_dataset, labels, lims,
                                      pdf=pdf)
    
    # if max(np.quantile(states_2[:,1],0.84), np.quantile(states_2[:,3],0.84)) > 3:
    # if max(best_2[1], best_2[3]) > 2.9:     ### or after cleaning chains...
    if max(pars_quant_2['u_0_1'][2], pars_quant_2['u_0_2'][2]) > 3.:
        return event_0, best, cplot
    return event_2, best_2, cplot

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
        # if theta[params_to_fit.index('u_0')] > 1000. or \
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
    #ln_prior_fluxes = 0
    #for flux in [*event.source_fluxes[0], event.blend_fluxes[0]]:
    #    if flux < 0.:
    #        ln_prior_fluxes += - 1/2 * (flux/1000)**2
    
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

    return 0.0 + ln_prior_t_E # + ln_prior_fluxes

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

def fit_EMCEE(params_to_fit, starting_params, sigmas, ln_prob, event,
              n_emcee, spec=""):
    """
    Fit model using EMCEE and print results.
    Arguments:
        params_to_fit - list of parameters
        starting_params - dict that specifies values of these parameters
        sigmas - list of sigma values used to find starting values
        ln_prob - function returning logarithm of probability
        event - MulensModel.Event instance
        n_emcee - Dictionary with number of walkers, steps, burn-in
        n_walkers - number of walkers in EMCEE -> inactive
        n_steps - number of steps per walker   -> inactive
        n_burn - number of steps considered as burn-in ( < n_steps)  -> inactive
    """
    n_dim = len(params_to_fit)
    mean = [starting_params[p] for p in params_to_fit]
    nwlk, nstep, nburn = n_emcee['nwlk'], n_emcee['nstep'], n_emcee['nburn']
    start = [mean + np.random.randn(n_dim) * sigmas for i in range(nwlk)]
    start = abs(np.array(start))

    # if params_to_fit

    # Run emcee (this can take some time):
    # dtype = [("event_fluxes", list)] #, ("log_prior", float)]
    blobs_type = [('source_fluxes', list), ('blend_fluxes', float)]
    sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob,
                                    blobs_dtype=blobs_type,
                                    args=(event, params_to_fit, spec))
                                    # backend=backend)
    # sampler = emcee.EnsembleSampler(
    #     nwlk, n_dim, ln_prob,
    #     moves=[(emcee.moves.DEMove(),0.8),(emcee.moves.DESnookerMove(),0.2)],
    #     args=(event, params_to_fit, spec))
    sampler.run_mcmc(start, nstep, progress=True)
    
    # Setting up multi-threading (std: fork in Linux, spawn in Mac)
    # multiprocessing.set_start_method("fork", force=True)
    # os.environ["OMP_NUM_THREADS"] = "1"
    # with multiprocessing.Pool() as pool:
    #     sampler = emcee.EnsembleSampler(nwlk, n_dim, ln_prob, pool=pool,
    #                                     args=(event, params_to_fit, spec))
    #     sampler.run_mcmc(start, nstep, progress=True)
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

    # Results:
    results = np.percentile(samples, [16, 50, 84], axis=0)
    print("Fitted parameters:")
    for i in range(n_dim):
        r = results[1, i]
        msg = params_to_fit[i] + ": {:.5f} +{:.5f} -{:.5f}"
        print(msg.format(r, results[2, i]-r, r-results[0, i]))

    # We extract best model parameters and chi2 from event:
    best_index = np.argmax(prob)
    # breakpoint()
    best = samples[best_index, :-1]
    for (key, value) in zip(params_to_fit, best):
        if key == 'flux_ratio':
            event.fix_source_flux_ratio = {event.datasets[0]: value}
        else:
            setattr(event.model.parameters, key, value)
    print("\nSmallest chi2 model:")
    print(*[repr(b) if isinstance(b, float) else b.value for b in best])
    print("chi2 = ", event.get_chi2())

    if n_emcee['clean_cplot']:
        new_states, pars_quant = clean_posterior_emcee(sampler, best, nburn)
        pars_quant = dict(zip(params_to_fit, pars_quant.T))
        if new_states is None:
            return best, results, samples, samples, sampler
        return best, pars_quant, samples, new_states, sampler

    # return best, pars_best, event.get_chi2(), states, sampler    
    return best, results, samples, samples, sampler

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
    if npars == 3:
        new_states = np.c_[new_states, source_fluxes[:,0], blend_flux]
    elif npars == 5:
        new_states = np.c_[new_states, source_fluxes[:,0], source_fluxes[:,1],
                           blend_flux]
    # breakpoint()

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

def make_three_plots(params, sampler, samples, new_states, nburn, best, dataset, labels,
                     lims, orig_data=[], pdf=""):
    """
    plot results
    """
    tracer_plot(params, sampler, nburn, pdf=pdf)
    if new_states.shape[1] == len(params) + 2:
        params += ['source_flux', 'blending_flux']
    elif new_states.shape[1] == len(params) + 3:
        params += ['source_flux_1', 'source_flux_2', 'blending_flux']
    
    # saving the states to file
    table = Table(samples, names=params+['ln_prob'])
    table.write("chains.txt", format='ascii', overwrite=True)
    # breakpoint()

    cplot = corner.corner(new_states, labels=params, truths=best,
                          quantiles=[0.16,0.50,0.84], show_titles=True)
    if pdf:
        pdf.savefig(cplot)
    else:
        plt.show()
    event = plot_fit(best[:-2], dataset, labels, lims, orig_data, pdf=pdf)

    return event, cplot

def tracer_plot(params_to_fit, sampler, nburn, pdf=""):
    """
    Plot tracer plots (or walkers' time series)
    """
    npars = len(params_to_fit)
    fig, axes = plt.subplots(npars, 1, sharex=True, figsize=(10,10) )
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

def plot_fit(best, dataset, labels, lims, orig_data=[], best_50=[], pdf=""):

    fig = plt.figure(figsize=(7.5,5.5))
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :]) # or gs.new_subplotspec((0, 0), rowspan=2)
    breakpoint()
    if len(best) == 3+2:
        model = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]})
    elif len(best) == 5+3:
        model = mm.Model({'t_0_1': best[0], 'u_0_1': best[1], 't_0_2': best[2],
                          'u_0_2': best[3], 't_E': best[4]})
    event = mm.Event(model=model, datasets=[dataset])
    data_label = "Original data" if not orig_data else "Subtracted data"
    event.plot_data(subtract_2450000=False, label=data_label)
    plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': False,
                   't_start': lims[0], 't_stop': lims[1], 'zorder': 10,
                   'color': 'black'}

    if orig_data:
        orig_data.plot(phot_fmt='mag', color='gray', alpha=0.2, label="Original data")
    if labels[1] == "":
        event.plot_model(label=labels[0], **plot_params)
    else:
        event.plot_model(label=r"%s"%labels[0], **plot_params)
    plt.tick_params(axis='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    
    ax2 = fig.add_subplot(gs[2:, :], sharex=ax1)
    event.plot_residuals(subtract_2450000=False, zorder=10) # fix zorder
    plt.tick_params(axis='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.xlim(lims[0], lims[1])
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
        event_x.plot_model(label=labels[1], **plot_params)

    ax1.legend(loc='best')
    if pdf:
        pdf.savefig(fig)
        plt.close('all')
    else:
        plt.show()
    return event
