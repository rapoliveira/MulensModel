"""
Short script to combine the results from the 1L2S and 2L1S models.
The script reads the yaml files with the results from UltraNest and creates
a table with the parameters, chi2 and evidence of the best models.
"""
from astropy.table import Table
import numpy as np
import os
import sys
import yaml


def read_results_UN(res_1L2S, res_2L1S):
    """
    Read the UltraNest results for both 1L2S and 2L1S models.
    Dictionaries with the results and the event identifier are returned.

    NOTE: Still going to add separate function to get EMCEE results.
    """
    suffix = '_1L2S_all_results_UN.yaml'
    fname = os.path.join(dir_1, res_1L2S, res_1L2S + suffix)
    with open(fname, encoding='utf-8') as in_1L2S:
        res_1L2S_dict = yaml.safe_load(in_1L2S)

    suffix = '_2L1S_all_results_UN.yaml'
    fname = os.path.join(dir_2, res_2L1S, res_2L1S + suffix)
    with open(fname, encoding='utf-8') as in_2L1S:
        res_2L1S_dict = yaml.safe_load(in_2L1S)

    return res_1L2S_dict, res_2L1S_dict


def add_params_to_table(res_fit):
    """
    Add the parameters, chi2 and evidence of the best model to the table.
    This function is called twice, once for the 1L2S and once for the 2L1S
    model.
    """
    best = res_fit['Best model']
    fitted_pars = res_fit['Fitted parameters']
    fitted_bflux = res_fit['Fitted fluxes']['flux_b_1']
    if len(fitted_pars) == 5:
        round_params = (2, 5, 2, 5, 2)
    else:
        round_params = (2, 5, 2, 2, 5, 2)

    best_params = list(best['Parameters'].values())
    best_params = [round(p, r) for p, r in zip(best_params, round_params)]
    t_E_sig = np.mean([fitted_pars['t_E'][1], -fitted_pars['t_E'][2]])
    bflux = round(best['Fluxes']['flux_b_1'], 5)
    bflux_sig = np.mean([fitted_bflux[1], -fitted_bflux[2]])
    t_E_sig, bflux_sig = round(t_E_sig, 2), round(bflux_sig, 5)
    chi2, evid = round(best['chi2'], 2), round(best['ln_ev'][0], 2)

    line = [*best_params, t_E_sig, bflux, bflux_sig, chi2, evid]
    return line


def declare_and_format_table():
    """
    Initial setup of the table and formatting of the columns.
    """
    names = ['id', 't_0_1', 'u_0_1', 't_0_2', 'u_0_2', 't_E_1L2S', 'sig_t_E_1',
             'bflux_1L2S', 'sig_bflux_1', 'chi2_1L2S', 'ln_ev_1L2S',
             't_0', 'u_0', 't_E_2L1S', 's', 'q', 'alpha', 'sig_t_E_2',
             'bflux_2L1S', 'sig_bflux_2', 'chi2_2L1S', 'ln_ev_2L1S']
    tab = Table(names=names, dtype=['S16']+[np.float64]*21)

    n_digits = ('%10.2f', '%7.5f', '%10.2f', '%7.5f', '%8.2f', '%9.2f',
                '%10.5f', '%11.5f', '%9.2f', '%10.2f',
                '%10.2f', '%8.5f', '%8.2f', '%6.2f', '%7.5f', '%6.2f',
                '%9.2f', '%10.5f', '%11.5f', '%9.2f', '%10.2f')
    for (i, col) in enumerate(tab.colnames[1:]):
        tab[col].info.format = n_digits[i]

    return tab


def filling_table(results_1L2S, results_2L1S, tab):
    """
    Fill the table with the parameters, chi2 and evidence of the best models,
    for each event_id placed in the first column.
    """
    for (res_1L2S, res_2L1S) in zip(results_1L2S, results_2L1S):
        if res_1L2S[:16] == res_2L1S[:16]:
            event_id = '.'.join(res_1L2S.split('_')[:3])
        else:
            raise ValueError('The two files do not correspond to same event.')

        dict_1L2S, dict_2L1S = read_results_UN(res_1L2S, res_2L1S)
        line_1L2S = add_params_to_table(dict_1L2S)
        line_2L1S = add_params_to_table(dict_2L1S)
        tab.add_row([event_id.ljust(16), *line_1L2S, *line_2L1S])

    return tab


def final_setup_and_save_table(tab):
    """
    Final setup of the table and save it to a text file.
    The column sig_t_E_2 is moved to the 15th position, in order to be
    next to the t_E_2L1S column.
    A fixed width format is used, and the header is written manually.
    """
    column_to_move = tab['sig_t_E_2']
    tab.remove_column('sig_t_E_2')
    tab.add_column(column_to_move, index=14)
    tab.write(f'{path}/comp_1L2S_2L1S.txt', format='ascii.fixed_width',
              overwrite=True, delimiter='')

    new_header = "# id               t_0_1       u_0_1    t_0_2       " + \
                 "u_0_2    t_E_1L2S  sig_t_E_1  bflux_1L2S  sig_bflux_1" + \
                 "  chi2_1L2S  ln_ev_1L2S  t_0         u_0       t_E_2L1S" + \
                 "  sig_t_E_2  s       q        alpha   bflux_2L1S  " + \
                 "sig_bflux_2  chi2_2L1S  ln_ev_2L1S\n"
    with open(f'{path}/comp_1L2S_2L1S.txt', 'r+') as file:
        lines = file.readlines()
        if lines:
            lines[0] = new_header
        file.seek(0)
        file.writelines(lines)
        file.truncate()


if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(__file__))
    method = str(sys.argv[1]) if len(sys.argv) > 1 else "UltraNest"
    if method.lower() == "ultranest":
        dir_1 = f'{path}/ultranest_1L2S/'
    elif method.lower() == "emcee":
        dir_1 = f'{path}/results_1L2S/yaml_results/'
    dir_2 = dir_1.replace('1L2S', '2L1S')

    results_1L2S = [f.split('-1L2S')[0] for f in os.listdir(dir_1)
                    if f[0] != '.' and f.endswith(".yaml")]
    results_2L1S = [f.split('-2L1S')[0] for f in os.listdir(dir_2)
                    if f[0] != '.' and f.endswith(".yaml")]
    results_1L2S, results_2L1S = sorted(results_1L2S), sorted(results_2L1S)

    tab = declare_and_format_table()
    tab = filling_table(results_1L2S, results_2L1S, tab)
    final_setup_and_save_table(tab)
