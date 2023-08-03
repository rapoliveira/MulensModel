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

import fit_1L2S_3steps as fit_1L2S


def main():

    # Fix the seed for the random number generator so the behavior is reproducible.
    np.random.seed(12343)

    # nwlk, nstep, nburn = 20, 3000, 1500     
    n_emcee = {'nwalk': 20, 'nstep':3000, 'nburn': 1500} # 20, 10000, 5000
    # my_dataset, event_orig, t_0_1, t_0_2 = sim_data_ex11()
    my_dataset, event_orig = sim_data_ex11()

    pdf = PdfPages('1L2S_3steps_result.pdf')
    event, best, lims = fit_1L2S.make_all_fittings(my_dataset, n_emcee, pdf=pdf)
    print("chi2_2 = ", event.get_chi2())
    print("chi2 of model_orig = ", event_orig.get_chi2())
    pdf.close()

    pdf = PdfPages('1L2S_3steps_fit.pdf')
    labels = [f"t_0_1 = {best[0]:.2f}\nu_0_1 = {best[1]:.2f}\nt_0_2 = "+\
              f"{best[2]:.2f}\nu_0_2 = {best[3]:.2f}\nt_E = {best[4]:.2f}", ""]
    fit_1L2S.plot_fit(best, my_dataset, labels, pdf=pdf)
    pdf.close()

def sim_data_ex11():
    
    # Input parameters
    t_0_1 = 6100.
    u_0_1 = 0.2
    t_0_2 = 6140.   # I also tried 6120, good! Change time_b
    u_0_2 = 0.01
    t_E = 25.
    assumed_flux_1 = 100.
    assumed_flux_2 = 5.
    assumed_flux_blend = 10.
    n_a = 1000
    n_b = 600
    time_a = np.linspace(6000., 6300., n_a)
    time_b = np.linspace(6139., 6141., n_b)
    time = np.sort(np.concatenate((time_a, time_b)))
    
    # Simulating the data with the input parameters
    model_1 = mm.Model({'t_0': t_0_1, 'u_0': u_0_1, 't_E': t_E})
    A_1 = model_1.get_magnification(time)
    model_2 = mm.Model({'t_0': t_0_2, 'u_0': u_0_2, 't_E': t_E})
    A_2 = model_2.get_magnification(time)
    flux = A_1 * assumed_flux_1 + A_2 * assumed_flux_2 + assumed_flux_blend
    flux_err = 6. + 0. * time
    flux += flux_err * np.random.normal(size=n_a+n_b)

    # Setting and plotting data
    my_dataset = mm.MulensData([time, flux, flux_err], phot_fmt='flux')
    model_orig = mm.Model({'t_0_1': t_0_1, 'u_0_1': u_0_1, 't_0_2': t_0_2,
                        'u_0_2': u_0_2, 't_E': t_E})
    event_orig = mm.Event(datasets=my_dataset, model=model_orig)
    # plt.plot(time, flux, 'ro')
    # plt.show()

    # Transforming flux into magnitude (Raphael test)
    # mag, mag_err = mm.Utils.get_mag_and_err_from_flux(flux, flux_err)
    # mag2= -2.5*np.log10(flux) + 22 # 21.5
    # mag_err2 = (2.5/np.log(10)) * (flux_err/flux) # +/- the same

    return my_dataset, event_orig #, t_0_1, t_0_2

if __name__ == "__main__":
    main()