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

import MulensModel as mm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm

def make_all_fittings(my_dataset, n_emcee, pdf=""):

    # 1st fit: Fitting a PSPL/1L1S without parallax...
    mag_ids = np.argsort(my_dataset.mag)
    t_brightest = np.mean(my_dataset.time[mag_ids][:10])
    # params = {'t_0': round(t_brightest, 1), 'u_0':0.1, 't_E': t_0_2-t_0_1}
    params = {'t_0': round(t_brightest, 1), 'u_0':0.1, 't_E': 20}
    my_model = mm.Model(params)
    my_event = mm.Event(datasets=my_dataset, model=my_model)
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    sigmas = [1., 0.05, 1.]
    msg = "This can take some time..."
    print(f"\n\033[1m -- First fit: PSPL to raw data. {msg}\033[0m")
    nwlk, nstep, nburn = n_emcee.values()
    output = fit_EMCEE(parameters_to_fit, params, sigmas, ln_prob, my_event,
                       n_walkers=nwlk, n_steps=nstep, n_burn=nburn)
    best, pars_best, states, sampler = output
    model_0 = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]})
    event_0 = mm.Event(model=model_0, datasets=[my_dataset])    # repeated!!!

    # Subtracting light curve from first fit (before plotting to get t_0_2)
    (_, blend_flux_0) = event_0.get_flux_for_dataset(0)
    flux_subt = my_dataset.flux - event_0.fits[0].get_model_fluxes() + _ + blend_flux_0
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
    event_0 = make_three_plots(parameters_to_fit, sampler, nburn, best,
                               my_dataset, labels, lims, pdf=pdf)

    # 2nd fit: ... PSPL to the subtracted data
    params = {'t_0': round(t_brightest,1), 'u_0':0.1, 't_E': best[2]}
    my_model = mm.Model(params)
    my_event = mm.Event(datasets=my_dataset_2, model=my_model)
    parameters_to_fit = ['t_0', 'u_0', 't_E']
    sigmas = [1., 0.05, 1.]
    print(f"\n\033[1m -- Second fit: PSPL to subtracted data. {msg}\033[0m")
    output = fit_EMCEE(parameters_to_fit, params, sigmas, ln_prob, my_event,
                       n_walkers=nwlk, n_steps=nstep, n_burn=nburn, spec="u_0")
    best_1, pars_best_1, states_1, sampler_1 = output
    if np.quantile(states_1, 0.84, axis=0)[1] > 15: # fix that 15?
        return event_0, best
    lims = [np.mean([best[0],best_1[0]]) - 2.5*abs(best[0]-best_1[0]),
            np.mean([best[0],best_1[0]]) + 2.5*abs(best[0]-best_1[0])]
    event_1 = make_three_plots(parameters_to_fit, sampler_1, nburn, best_1,
                            my_dataset_2, labels, lims, my_dataset, pdf=pdf)

    # Third fit: 1L2S, source flux ratio not set yet (regression)
    params = {'t_0_1': best[0], 'u_0_1': best[1], 't_0_2': best_1[0],
              'u_0_2': best_1[1], 't_E': best_1[2]}
    my_model = mm.Model(params)
    my_event = mm.Event(datasets=my_dataset, model=my_model)
    parameters_to_fit = ["t_0_1", "u_0_1", "t_0_2", "u_0_2", "t_E"]
    sigmas = [0.1, 0.05, 1., 0.01, 0.1]
    print(f"\n\033[1m -- Third fit: 1L2S to original data. {msg}\033[0m")
    output = fit_EMCEE(parameters_to_fit, params, sigmas, ln_prob, my_event,
                       n_walkers=nwlk, n_steps=nstep, n_burn=nburn)
    best_2, pars_best_2, states_2, sampler_2 = output
    labels = ["1L2S, max_prob", "1L2S, 50th_perc"]
    lims = sorted([best_2[0], best_2[2]])
    lims = [lims[0]-3*best_2[4], lims[1]+3*best_2[4]]
    event_2 = make_three_plots(parameters_to_fit, sampler_2, nburn, best_2,
                               my_dataset, labels, lims, pdf=pdf)
    
    if max(np.quantile(states_2[:,1],0.84), np.quantile(states_2[:,3],0.84)) > 3:
        return event_0, best
    return event_2, best_2

def make_three_plots(params, sampler, nburn, best, dataset, labels, lims,
                     orig_data=[], pdf=""):
    """
    plot results
    """
    tracer_plot(params, sampler, nburn, pdf=pdf)
    new_states = clean_posterior_emcee(sampler, best, nburn)
    fig1 = corner.corner(new_states, labels=params, truths=best, 
                        quantiles=[0.16,0.50,0.84], show_titles=True)
    if pdf:
        pdf.savefig(fig1)
    else:
        plt.show()
    event = plot_fit(best, dataset, labels, lims, orig_data, pdf=pdf)

    return event

def ln_like(theta, event, parameters_to_fit):
    """ likelihood function """
    for (param, theta_) in zip(parameters_to_fit, theta):
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
    return 0.0 + ln_prior_t_E

def ln_prob(theta, event, parameters_to_fit, spec=""):
    """ combines likelihood and priors"""
    ln_prior_ = ln_prior(theta, event, parameters_to_fit, spec)
    if not np.isfinite(ln_prior_):
        return -np.inf
    ln_like_ = ln_like(theta, event, parameters_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
        return -np.inf

    return ln_prior_ + ln_like_

def fit_EMCEE(parameters_to_fit, starting_params, sigmas, ln_prob, event,
              n_walkers=20, n_steps=3000, n_burn=1500, spec=""):
    """
    Fit model using EMCEE and print results.
    Arguments:
        parameters_to_fit - list of parameters
        starting_params - dict that specifies values of these parameters
        sigmas - list of sigma values used to find starting values
        ln_prob - function returning logarithm of probability
        event - MulensModel.Event instance
        n_walkers - number of walkers in EMCEE
        n_steps - number of steps per walker
        n_burn - number of steps considered as burn-in ( < n_steps)
    """
    n_dim = len(parameters_to_fit)
    mean = [starting_params[p] for p in parameters_to_fit]
    start = [mean + np.random.randn(n_dim) * sigmas for i in range(n_walkers)]
    start = abs(np.array(start))

    # Run emcee (this can take some time):
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_prob, args=(event, parameters_to_fit, spec))
    # sampler = emcee.EnsembleSampler(
    #     n_walkers, n_dim, ln_prob,
    #     moves=[(emcee.moves.DEMove(),0.8),(emcee.moves.DESnookerMove(),0.2)],
    #     args=(event, parameters_to_fit, spec))
    # sampler.run_mcmc(start, n_steps, progress=True)
    
    # Setting up multi-threading
    # import multiprocessing
    # multiprocessing.set_start_method("fork", force=True) # already fork
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    from multiprocessing import Pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob, pool=pool,
                                        args=(event, parameters_to_fit, spec))
        sampler.run_mcmc(start, n_steps, progress=True)
    # breakpoint()

    # Remove burn-in samples and reshape:
    samples = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))

    # Results:
    results = np.percentile(samples, [16, 50, 84], axis=0)
    print("Fitted parameters:")
    for i in range(n_dim):
        r = results[1, i]
        msg = parameters_to_fit[i] + ": {:.5f} +{:.5f} -{:.5f}"
        print(msg.format(r, results[2, i]-r, r-results[0, i]))

    # We extract best model parameters and chi2 from event:
    prob = sampler.lnprobability[:, n_burn:].reshape((-1))
    best_index = np.argmax(prob)
    best = samples[best_index, :]
    for (key, value) in zip(parameters_to_fit, best):
        if key == 'flux_ratio':
            event.fix_source_flux_ratio = {event.datasets[0]: value}
        else:
            setattr(event.model.parameters, key, value)
    print("\nSmallest chi2 model:")
    print(*[repr(b) if isinstance(b, float) else b.value for b in best])
    print("chi2 = ", event.get_chi2())
    
    # Getting states and reshaping (20, 1500, 3) -> (30000, 3)
    states = sampler.chain[:, n_burn:, :]
    # states = np.reshape(states,[gwlk*n_burn,len(parameters_to_fit)])
    after_burn = n_steps-n_burn
    states = np.reshape(states,[n_walkers*after_burn, len(parameters_to_fit)])
    w = np.quantile(states,[0.16,0.50,0.84],axis=0)
    pars_best = w[1,:]
    perr_low  = w[0,:]-pars_best
    perr_high = w[2,:]-pars_best
    # breakpoint()

    # return best, pars_best, event.get_chi2(), states, sampler
    return best, pars_best, states, sampler

def tracer_plot(parameters_to_fit, sampler, nburn, pdf=""):
    """
    Plot tracer plots (or walkers' time series)
    """
    npars = len(parameters_to_fit)
    fig, axes = plt.subplots(npars, 1, sharex=True, figsize=(10,10) )
    for i in range(npars):
        axes[i].plot(np.array(sampler.chain[:,:,i]).T,rasterized=True)
        axes[i].axvline(x=nburn, ls='--', color='gray', lw=1.5)
        axes[i].set_ylabel(parameters_to_fit[i], fontsize=16)
    axes[npars-1].set_xlabel(r'steps', fontsize=16)
    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
    else:
        plt.show()

def clean_posterior_emcee(sampler, params, n_burn):
    """
    Manipulate emcee chains to reject stray walkers and clean posterior
    Arguments:
        sampler - ensemble sampler from EMCEE
        params - set of best parameters from fit_EMCEE function
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
    # states = sampler.chain[w[0],burnin::thin,:].copy()  # with thinning
    new_states = sampler.chain[w[0],n_burn::,:].copy()  # no thinning
    gwlk, nthin, npars = np.shape(new_states)
    new_states = np.reshape(new_states,[gwlk*nthin, npars])
    n_rej, perc_rej = sampler.nwalkers-gwlk, 100*(1-gwlk/sampler.nwalkers)
    print(f'Obs: {n_rej} walkers ({round(perc_rej)}%) were rejected')
    
    # Finding median values and confidence intervals... FIT SKEWED GAUSSIANS!!!
    # To-Do... [...]
    test = params # [...]

    return new_states

def plot_fit(best, dataset, labels, lims, orig_data=[], best_50=[], pdf=""):

    fig = plt.figure(figsize=(7.5,5.5))
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :]) # or gs.new_subplotspec((0, 0), rowspan=2)
    if len(best) == 3:
        model = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]})
    elif len(best) == 5:
        model = mm.Model({'t_0_1': best[0], 'u_0_1': best[1], 't_0_2': best[2],
                          'u_0_2': best[3], 't_E': best[4]})
    event = mm.Event(model=model, datasets=[dataset])
    data_label = "Original data" if not orig_data else "Subtracted data"
    # breakpoint()
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
