"""
Fits binary source model using EMCEE sampler.

The code simulates binary source light curve and fits the model twice:
with source flux ratio found via linear regression and
with source flux ratio as a chain parameter.
"""
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

import MulensModel as mm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

import corner
import os
from astropy.table import Table

import fit_1L2S_3steps as fit_1L2S


# Fix the seed for the random number generator so the behavior is reproducible.
np.random.seed(12343)
idx = -1

for dat_file in sorted(os.listdir('./W16_photometry/')):
    # breakpoint()
    if dat_file == "v":
        continue
    # idx += 1
    # if idx < 53: continue
    if dat_file != "PAR-09.dat": # "PAR-09.dat":
        continue
    print(f'\n\033[1m * Running fit for {dat_file}\033[0m')
    tab = Table.read(f'./W16_photometry/{dat_file}', format='ascii')
    my_dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
    my_dataset = mm.MulensData(my_dataset, phot_fmt='mag')
    
    # First look at the data
    # fig = plt.figure()
    # my_dataset.plot(phot_fmt='mag')
    # plt.tight_layout()
    # plt.show()

    # Fix the seed for the random number generator so the behavior is reproducible.
    np.random.seed(12343)

    # nwlk, nstep, nburn = 20, 3000, 1500 # 20, 10000, 5000    
    n_emcee = {'nwalk': 20, 'nstep': 3000, 'nburn': 1500}
    # my_dataset, event_orig, t_0_1, t_0_2 = sim_data_ex11()
    # t_0_1, t_0_2 = 2000, 3000

    pdf = PdfPages(f"W16_output/all_plots/{dat_file.split('.')[0]}_result.pdf")
    event, best, cplot = fit_1L2S.make_all_fittings(my_dataset, n_emcee, pdf=pdf)
    print("chi2_2 = ", event.get_chi2())
    # print("chi2 of model_orig = ", event_orig.get_chi2())
    pdf.close()
    print(f"Saved output: {dat_file.split('.')[0]}_result.pdf", end="")
    # breakpoint()

    pdf = PdfPages(f"W16_output/{dat_file.split('.')[0]}_cplot.pdf")
    pdf.savefig(cplot)
    pdf.close()

    pdf = PdfPages(f"W16_output/{dat_file.split('.')[0]}_fit.pdf")
    # try:
    if len(best) == 5 + 3:
        labels = [f"t_0_1 = {best[0]:.2f}\nu_0_1 = {best[1]:.2f}\nt_0_2 = "+
                  f"{best[2]:.2f}\nu_0_2 = {best[3]:.2f}\nt_E = {best[4]:.2f}",
                  ""]
        lims = sorted([best[0], best[2]])
        lims = [lims[0]-3*best[4], lims[1]+3*best[4]]
    # except Exception:
    else:
        labels = [f"t_0 = {best[0]:.2f}\nu_0 = {best[1]:.2f}\nt_E = "+
                  f"{best[2]:.2f}", ""]
        lims = [best[0] - 3*best[2], best[0] + 3*best[2]]
    fit_1L2S.plot_fit(best, my_dataset, labels, lims, pdf=pdf)
    # event = plot_fit(best, dataset, labels, lims, orig_data, pdf=pdf)
    pdf.close()
    print(f", {dat_file.split('.')[0]}_fit.pdf", end=' ')

    with open(f"W16_output/txt/{dat_file.split('.')[0]}.txt", 'w') as txt:
        txt.truncate()
        txt.write(f'Final chi2 = {event.get_chi2():.5f}\n')

    print(f"and {dat_file.split('.')[0]}.txt", end='\n\n')
    print("--------------------------------------------------")

    # breakpoint()
