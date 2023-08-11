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

def write_tables(event_id, result):

    best, samples, event = result
    path = os.path.dirname(os.path.realpath(__file__))
    chains_file = f"{path}/W16_output/chains/{event_id[:-4]}_chains.txt"
    results_file = f"{path}/results-1L2S.txt"

    # saving the states to file
    chains = Table(samples, names=list(best.keys())+['ln_prob'])
    chains.write(chains_file, format='ascii', overwrite=True)
    
    # saving the results to general table
    res_tab = Table.read(results_file, format='ascii')
    best = dict(item for item in list(best.items()) if 'flux' not in item[0])
    lst = list(best.values())+[0.,0.] if len(best)==3 else list(best.values())
    res_tab[int(event_id[4:6])-1] = [event_id[:-4]] + lst + [event.get_chi2()]
    res_tab.write(results_file, format='ascii', overwrite=True)

# Fix the seed for the random number generator so the behavior is reproducible.
np.random.seed(12343)
idx = -1

for dat_file in sorted(os.listdir('./W16_photometry/')):
    
    if dat_file == "v":
        continue
    # idx += 1
    # if idx < 58: continue
    if dat_file != "PAR-09.dat": # "PAR-45.dat":
        continue
    print(f'\n\033[1m * Running fit for {dat_file}\033[0m')
    tab = Table.read(f'./W16_photometry/{dat_file}', format='ascii')
    my_dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
    my_dataset = mm.MulensData(my_dataset, phot_fmt='mag')
    
    # First look at the data
    # fig = plt.figure(tight_layout=True)
    # my_dataset.plot(phot_fmt='mag')
    # plt.show()

    # nwlk, nstep, nburn = 20, 3000, 1500 # 20, 10000, 5000
    # ans should be 'max_prob' or '50th_perc'
    n_emcee = {'nwlk':20, 'nstep':3000, 'nburn':1500, 'ans':'max_prob',
               'clean_cplot': False, 'tqdm': True}

    pdf = PdfPages(f"W16_output/all_plots/{dat_file.split('.')[0]}_result.pdf")
    result, cplot = fit_1L2S.make_all_fittings(my_dataset, n_emcee, pdf=pdf)
    # print("chi2_2 = ", result[0].get_chi2())
    # print("chi2 of model_orig = ", event_orig.get_chi2())
    pdf.close()
    write_tables(dat_file, result)
    print(f"Saved output: {dat_file.split('.')[0]}_result.pdf,", end=" ")
    print(f"{dat_file.split('.')[0]}.txt and", end=' ')

    pdf = PdfPages(f"W16_output/{dat_file.split('.')[0]}_cplot.pdf")
    pdf.savefig(cplot)
    pdf.close()

    pdf = PdfPages(f"W16_output/{dat_file.split('.')[0]}_fit.pdf")
    xlim = fit_1L2S.get_xlim(result[0], my_dataset)
    fit_1L2S.plot_fit(result[0], my_dataset, n_emcee, xlim, pdf=pdf)
    pdf.close()
    print(f", {dat_file.split('.')[0]}_fit.pdf\n")
    print("--------------------------------------------------")
    