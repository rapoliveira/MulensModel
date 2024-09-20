"""
Fits PSPL model with parallax using EMCEE sampler.

"""
import os
import sys
import numpy as np
try:
    import emcee
except ImportError as err:
    print(err)
    print("\nEMCEE could not be imported.")
    print("Get it from: http://dfm.io/emcee/current/user/install/")
    print("and re-run the script")
    sys.exit(1)
import matplotlib.pyplot as plt

import MulensModel as mm
# from example_11_raphael import fit_EMCEE


# Define likelihood functions
def ln_like(theta, event, parameters_to_fit):
    """ likelihood function """
    for (parameter, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)

    chi2 = event.get_chi2()
    if chi2 < ln_like.best[0]:
        ln_like.best = [chi2, theta]
    return -0.5 * chi2
ln_like.best = [np.inf]


def ln_prior(theta, parameters_to_fit):
    """priors - we only reject obviously wrong models"""
    if theta[parameters_to_fit.index("t_E")] < 0.:
        return -np.inf
    return 0.0


def ln_prob(theta, event, parameters_to_fit):
    """ combines likelihood and priors"""
    ln_prior_ = ln_prior(theta, parameters_to_fit)
    if not np.isfinite(ln_prior_):
        return -np.inf
    ln_like_ = ln_like(theta, event, parameters_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
        return -np.inf

    return ln_prior_ + ln_like_


def fit_EMCEE(parameters_to_fit, starting_params, sigmas, ln_prob, event,
              n_walkers=20, n_steps=3000, n_burn=1500):
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

    # Run emcee (this can take some time):
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_prob, args=(event, parameters_to_fit))
    sampler.run_mcmc(start, n_steps)

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
            event.fix_source_flux_ratio = {my_data: value}
        else:
            setattr(event.model.parameters, key, value)

    print("\nSmallest chi2 model:")
    print(*[repr(b) if isinstance(b, float) else b.value for b in best])
    print("chi2 = ", event.get_chi2())


# Read the data (ORIGINAL from example)
file_name = os.path.join(
   mm.DATA_PATH, "photometry_files", "OB05086",
   "starBLG234.6.I.218982.dat")
my_data = mm.MulensData(file_name=file_name, add_2450000=True)


# Raphael: I tried to copy the data from example_11 (binary sources) -- bad because it is in fluxes...

# t_0_1 = 6100.
# u_0_1 = 0.2
# t_0_2 = 6140.
# u_0_2 = 0.01
# t_E = 25.
# assumed_flux_1 = 100.
# assumed_flux_2 = 5.
# assumed_flux_blend = 10.
# n_a = 1000
# n_b = 600
# time_a = np.linspace(6000., 6300., n_a)
# time_b = np.linspace(6139., 6141., n_b)
# time = np.sort(np.concatenate((time_a, time_b)))
# model_1 = mm.Model({'t_0': t_0_1, 'u_0': u_0_1, 't_E': t_E})
# A_1 = model_1.get_magnification(time)
# model_2 = mm.Model({'t_0': t_0_2, 'u_0': u_0_2, 't_E': t_E})
# A_2 = model_2.get_magnification(time)
# flux = A_1 * assumed_flux_1 + A_2 * assumed_flux_2 + assumed_flux_blend
# flux_err = 6. + 0. * time
# flux += flux_err * np.random.normal(size=n_a+n_b)

# small modifications to the data (Raphael)
# mag = -2.5*np.log10(flux) + 22 # 21.5
# mag_err = (2.5/np.log(10)) * (flux_err/flux)
# time += 2450000
# print(mag, mag_err)

# my_data = mm.MulensData([time, flux, flux_err], phot_fmt='flux')
# my_data = mm.MulensData([time, mag, mag_err], phot_fmt='mag') # error...

# If you want to plot, then just uncomment:
# import matplotlib.pyplot as plt
# plt.plot(time, flux, 'ro')
# # plt.plot(time, mag, 'ro')
# plt.gca().invert_yaxis()
# plt.show()
# breakpoint()

coords = "18:04:45.71 -26:59:15.2"

# Starting parameters:
params = dict()
params['t_0'] = 2453628.3 # 2453628.3 or 6101?
params['t_0_par'] = 2453628. # 2453628.
params['u_0'] = 0.37 # 0.37 or 0.19?  # Change sign of u_0 to find the other solution.
params['t_E'] = 20. # 100.
params['pi_E_N'] = 0.
params['pi_E_E'] = 0.
my_model = mm.Model(params, coords=coords)
my_event = mm.Event(datasets=my_data, model=my_model)

# Which parameters we want to fit?
parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
# And remember to provide dispersions to draw starting set of points
sigmas = [0.01, 0.001, 0.1, 0.01, 0.01]

# Initializations for EMCEE
n_dim = len(parameters_to_fit)
n_walkers = 40
n_steps = 500
n_burn = 150
# Including the set of n_walkers starting points:
start_1 = [params[p] for p in parameters_to_fit]
start = [start_1 + np.random.randn(n_dim) * sigmas
         for i in range(n_walkers)]

# Run emcee (this can take some time):
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, ln_prob, args=(my_event, parameters_to_fit))
sampler.run_mcmc(start, n_steps)

# Remove burn-in samples and reshape:
samples = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))

# Results:
results = np.percentile(samples, [16, 50, 84], axis=0)
print("Fitted parameters:")
for i in range(n_dim):
    r = results[1, i]
    print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]))

print("\nSmallest chi2 model:")
print(*[b if isinstance(b, float) else b.value for b in ln_like.best[1]])
print(ln_like.best[0])

# Raphael: Trying to fit binary source models here... (from example11)
params2 = {'t_0_1': 2453628.3, 'u_0_1': 0.37, 't_0_2': 2453668.3, 'u_0_2': 0.10,
          't_E': 20.}
my_model2 = mm.Model(params2)
my_event2 = mm.Event(datasets=my_data, model=my_model2)
parameters_to_fit = ["t_0_1", "u_0_1", "t_0_2", "u_0_2", "t_E"]
sigmas2 = [0.1, 0.05, 1., 0.01, 1.]
print("\nFit for binary source event (Raphael). This can take some time...")
fit_EMCEE(parameters_to_fit, params2, sigmas2, ln_prob, my_event2)

# Now let's plot 3 models
plt.figure()
model_0 = mm.Model({'t_0': 2453628.29062, 'u_0': 0.37263, 't_E': 102.387105})
model_1 = mm.Model(
    {'t_0': 2453630.35507, 'u_0': 0.488817, 't_E': 93.611301,
     'pi_E_N': 0.2719, 'pi_E_E': 0.1025, 't_0_par': params['t_0_par']},
    coords=coords)
model_2 = mm.Model(
    {'t_0': 2453630.67778, 'u_0': -0.415677, 't_E': 110.120755,
     'pi_E_N': -0.2972, 'pi_E_E': 0.1103, 't_0_par': params['t_0_par']},
    coords=coords)
event_0 = mm.Event(model=model_0, datasets=[my_data])
event_1 = mm.Event(model=model_1, datasets=[my_data])
event_2 = mm.Event(model=model_2, datasets=[my_data])

t_1 = 2453200.
t_2 = 2453950.
plot_params = {'lw': 2.5, 'alpha': 0.3, 'subtract_2450000': True,
               't_start': t_1, 't_stop': t_2}

my_event.plot_data(subtract_2450000=True)
event_0.plot_model(label='no pi_E', **plot_params)
event_1.plot_model(label='pi_E, u_0>0', **plot_params)
event_2.plot_model(
    label='pi_E, u_0<0', color='black', ls='dashed', **plot_params)

# Raphael: plotting the binary source model
model_R = mm.Model(
    {'t_0_1': 2453619.32210, 'u_0_1': 4.46390, 't_0_2': 2453630.69323,
     'u_0_2': 0.57468, 't_E': 28.57504})
event_R = mm.Event(model=model_R, datasets=[my_data])
event_R.plot_model(
    label='binary_source (Raphael)', color='red', ls='dashed', **plot_params)

plt.xlim(t_1-2450000., t_2-2450000.)
plt.legend(loc='best')
plt.title('Data and 3 fitted models')
plt.tight_layout()
plt.show()

plt.figure()
event_R.plot_residuals(subtract_2450000=True)   # Raphael: anything missing here?
plt.xlim(t_1-2450000, t_2-2450000)
plt.tight_layout()
plt.show()

