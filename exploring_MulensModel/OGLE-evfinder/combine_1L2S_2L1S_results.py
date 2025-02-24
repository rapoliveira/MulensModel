"""
Short script to combine the results from the 1L2S and 2L1S models.
It reads the yaml files with results from EMCEE (specially to get the
uncertainties) and UltraNest. A table is created with parameters, chi2
and evidence of the best models.
"""
import argparse
import os

from astropy.table import Table
import numpy as np
import yaml


def parse_arguments():
    """
    Parses command-line arguments and returns them.
    """
    # defaults = ["../test_chips_oct24/runs_12_13_14", "BLG50X"]
    defaults = ["UltraNest", "obvious"]

    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument('method', type=str, nargs='?', default=defaults[0],
                        help="Method to get results: UltraNest or EMCEE")
    parser.add_argument('dataset', type=str, nargs='?', default=defaults[1],
                        help="Dataset to get results: obvious, BLG50X...")
    args = parser.parse_args()

    path = os.path.dirname(os.path.abspath(__file__))
    dirs = [f'{path}/ultranest_1L2S/{args.dataset}/',
            f'{path}/results_1L2S/{args.dataset}/yaml_results/']
    idx = 0 if args.method.lower() == "ultranest" else 1

    return args, path, dirs, idx


def declare_and_format_table(method):
    """
    Initial setup of the table and formatting of the columns.
    """
    names = ['id', 't_0_1', 'u_0_1', 't_0_2', 'u_0_2', 't_E_1L2S',
             'sig_t_E_1', 'bflux_1L2S', 'sig_bflux_1', 'chi2_1L2S',
             't_0', 'u_0', 't_E_2L1S', 's', 'q', 'alpha', 'sig_t_E_2',
             'bflux_2L1S', 'sig_bflux_2', 'chi2_2L1S']
    n_digits = ['%10.2f', '%7.5f', '%10.2f', '%7.5f', '%8.2f', '%9.2f',
                '%10.5f', '%11.5f', '%9.5f',
                '%10.2f', '%7.5f', '%8.2f', '%6.2f', '%7.5f', '%6.2f',
                '%9.2f', '%10.5f', '%11.5f', '%9.5f']

    if method.lower() == "ultranest":
        names.insert(10, 'ln_ev_1L2S')
        names.append('ln_ev_2L1S')
        n_digits.insert(10, '%10.2f')
        n_digits.append('%10.2f')
    tab = Table(names=names, dtype=['S16']+[np.float64]*len(names[1:]))

    for (i, col) in enumerate(tab.colnames[1:]):
        tab[col].info.format = n_digits[i]

    return tab


def read_results_EMCEE(dir_1L2S):
    """
    Read EMCEE results for both 1L2S and 2L1S models.
    The solution with lower chi2 is chosen for the 2L1S model.
    """
    # breakpoint()
    dir_1L2S += "_results.yaml"
    with open(dir_1L2S, encoding='utf-8') as in_1L2S:
        dict_1L2S = yaml.safe_load(in_1L2S)

    dir_2L1S = dir_1L2S.replace(f'1L2S/{args.dataset}/yaml_results/',
                                f'2L1S/{args.dataset}/')
    dir_2L1S = dir_2L1S.replace('_results', '_2L1S_all_results_between')
    with open(dir_2L1S, encoding='utf-8') as in_2L1S:
        res_2L1S_between = yaml.safe_load(in_2L1S)
        chi2_bet = res_2L1S_between['Best model']['chi2']
    dir_2L1S = dir_2L1S.replace('between', 'beyond')
    with open(dir_2L1S, encoding='utf-8') as in_2L1S:
        res_2L1S_beyond = yaml.safe_load(in_2L1S)
        chi2_bey = res_2L1S_beyond['Best model']['chi2']
    dict_2L1S = res_2L1S_between if chi2_bet < chi2_bey else res_2L1S_beyond

    return dict_1L2S, dict_2L1S


def get_EMCEE_sigmas(dict_1L2S, dict_2L1S):
    """
    Get the uncertainties from the EMCEE results, because the ones from
    UltraNest are underestimated.
    """
    sigmas_1L2S_t_E = dict_1L2S['Fitted parameters']['t_E']
    t_E_mean = np.mean([sigmas_1L2S_t_E[1], -sigmas_1L2S_t_E[2]])
    sigmas_1L2S_bflux = dict_1L2S['Fitted fluxes']['flux_b_1']
    bflux_mean = np.mean([sigmas_1L2S_bflux[1], -sigmas_1L2S_bflux[2]])
    sigmas_1L2S = [t_E_mean, bflux_mean]

    sigmas_2L1S_t_E = dict_2L1S['Fitted parameters']['t_E']
    t_E_mean = np.mean([sigmas_2L1S_t_E[1], -sigmas_2L1S_t_E[2]])
    sigmas_2L1S_bflux = dict_2L1S['Fitted fluxes']['flux_b_1']
    bflux_mean = np.mean([sigmas_2L1S_bflux[1], -sigmas_2L1S_bflux[2]])
    sigmas_2L1S = [t_E_mean, bflux_mean]

    return [sigmas_1L2S, sigmas_2L1S]


def read_results_UN(res_1L2S):
    """
    Read the UltraNest results for both 1L2S and 2L1S models.
    Dictionaries with both results are returned.
    No check of chi2 is needed, because a single UltraNest fit is done.
    """
    suffix = '_1L2S_all_results_UN.yaml'
    event_id = res_1L2S.split('/')[-1]
    fname = os.path.join(res_1L2S, event_id + suffix)
    with open(fname, encoding='utf-8') as in_1L2S:
        dict_1L2S = yaml.safe_load(in_1L2S)

    fname = fname.replace('1L2S', '2L1S')
    with open(fname, encoding='utf-8') as in_2L1S:
        dict_2L1S = yaml.safe_load(in_2L1S)

    return dict_1L2S, dict_2L1S


def change_sigmas_UN(dict_1L2S, dict_2L1S, sigmas):
    """
    Change the sigmas in the UltraNest results to the EMCEE.
    """
    for i, dict_name in enumerate([dict_1L2S, dict_2L1S]):
        dict_name['Fitted parameters']['t_E'][1] = sigmas[i][0]
        dict_name['Fitted parameters']['t_E'][2] = -sigmas[i][0]
        dict_name['Fitted fluxes']['flux_b_1'][1] = sigmas[i][1]
        dict_name['Fitted fluxes']['flux_b_1'][2] = -sigmas[i][1]


def add_params_to_table(method, res_fit, dof):
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

    best_pars = list(best['Parameters'].values())
    best_pars = [round(p, r) for p, r in zip(best_pars, round_params)]
    t_E_sig = np.mean([fitted_pars['t_E'][1], -fitted_pars['t_E'][2]])
    bflux = round(best['Fluxes']['flux_b_1'], 5)
    bflux_sig = np.mean([fitted_bflux[1], -fitted_bflux[2]])
    t_E_sig, bflux_sig = round(t_E_sig, 2), round(bflux_sig, 5)

    # line = [*best_pars, t_E_sig, bflux, bflux_sig, round(best['chi2'], 2)]
    line = [*best_pars, t_E_sig, bflux, bflux_sig, round(best['chi2']/dof, 5)]
    if method.lower() == "ultranest":
        line.append(round(best['ln_ev'][0], 2))

    return line


def fill_table(method, dirs, event_ids):
    """
    Fill the table with the parameters, chi2 and evidence of the best models,
    for each event_id placed in the first column.
    """
    tab = declare_and_format_table(method)

    for event_id in event_ids:
        dir_1L2S_emcee = os.path.join(dirs[1], event_id)
        dict_1L2S, dict_2L1S = read_results_EMCEE(dir_1L2S_emcee)

        if method.lower() == "ultranest":
            sigmas_1L2S_2L1S = get_EMCEE_sigmas(dict_1L2S, dict_2L1S)
            res_1L2S = os.path.join(dirs[0], event_id)
            dict_1L2S, dict_2L1S = read_results_UN(res_1L2S)
            change_sigmas_UN(dict_1L2S, dict_2L1S, sigmas_1L2S_2L1S)
        dof = dict_1L2S['Best model']['dof']
        line_1L2S = add_params_to_table(method, dict_1L2S, dof)
        line_2L1S = add_params_to_table(method, dict_2L1S, dof-1)

        event_id = event_id.split('_OGLE')[0]
        tab.add_row([event_id.ljust(16), *line_1L2S, *line_2L1S])

    return tab


def final_setup_and_save_table(path, tab, method):
    """
    Final setup of the table and save it to a text file.
    The column sig_t_E_2 is moved to the right of the t_E_2L1S column.
    A fixed width format is used, and the header is written manually.
    """
    column_to_move = tab['sig_t_E_2']
    tab.remove_column('sig_t_E_2')
    new_idx = 14 if method.lower() == "ultranest" else 13
    tab.add_column(column_to_move, index=new_idx)
    method_ = "_un" if method.lower() == "ultranest" else "_emcee"
    fname = os.path.join(path, f'comp_1L2S_2L1S{method_}.txt')
    tab.write(fname, format='ascii.fixed_width', overwrite=True, delimiter='')

    new_header = "# id               t_0_1       u_0_1    t_0_2       " + \
                 "u_0_2    t_E_1L2S  sig_t_E_1  bflux_1L2S  sig_bflux_1" + \
                 "  chi2_1L2S  ln_ev_1L2S  t_0         u_0      t_E_2L1S" + \
                 "  sig_t_E_2  s       q        alpha   bflux_2L1S  " + \
                 "sig_bflux_2  chi2_2L1S  ln_ev_2L1S\n"
    if method.lower() == "emcee":
        new_header = new_header.replace('ln_ev_1L2S  ', '')
        new_header = new_header.replace('ln_ev_2L1S', '')
    with open(fname, 'r+') as file:
        lines = file.readlines()
        if lines:
            lines[0] = new_header
        file.seek(0)
        file.writelines(lines)
        file.truncate()


if __name__ == '__main__':

    args, path, dirs, idx = parse_arguments()
    str_split = '-1L2S' if args.method.lower() == "ultranest" else '_results'
    event_ids = [f.split(str_split)[0] for f in os.listdir(dirs[idx])
                 if f[0] != '.' and f.endswith(".yaml")]
    tab = fill_table(args.method, dirs, sorted(event_ids))
    final_setup_and_save_table(path, tab, args.method)
