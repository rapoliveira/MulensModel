"""
Short script to combine the results from the 1L2S and 2L1S models.
It reads the yaml files with results from EMCEE (specially to get the
uncertainties) and UltraNest. A table is created with parameters, chi2
and evidence of the best models.
"""
import argparse
import os

from astropy.table import Table
import MulensModel as mm
import numpy as np
import yaml


def parse_arguments():
    """
    Parses command-line arguments and returns them.
    """
    defaults = ["UltraNest", "obvious"]
    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable verbose mode")
    parser.add_argument('-lc', '--lowcadence', action='store_true',
                        help="Enable low-cadence mode")
    parser.add_argument('method', type=str, nargs='?', default=defaults[0],
                        help="Method to get results: UltraNest or EMCEE")
    parser.add_argument('dataset', type=str, nargs='?', default=defaults[1],
                        help="Dataset: obvious, BLG50X1, lcmin3to5, lcall1...")
    args = parser.parse_args()

    path = os.path.dirname(os.path.abspath(__file__))
    dirs = [f'{path}/ultranest_1L2S/{args.dataset}/',
            f'{path}/results_1L2S/{args.dataset}/yaml_results/']
    args.method = args.method.lower()
    idx = 0 if args.method == "ultranest" else 1
    print()

    return args, path, dirs, idx


def remove_cvs_mroz20(event_ids):
    """
    Remove the CV candidates that are already in the Mroz+20 paper.

    Link to the paper:
    https://ui.adsabs.harvard.edu/abs/2015AcA....65..313M/abstract
    """
    cvs_table = Table.read('CVs_tab1_Mroz2015.dat', format='ascii')
    cvs_ids = [line['col4'] + '.' + line['col5'] for line in cvs_table]
    ids_temp = [id.split('_OGLE')[0].replace("_", '.') for id in event_ids]

    mask_cvs = [event_id in cvs_ids for event_id in ids_temp]
    new = [id_ for (id_, mask) in zip(event_ids, mask_cvs) if not mask]

    return new


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

    if method == "ultranest":
        names.insert(10, 'ln_ev_1L2S')
        names.append('ln_ev_2L1S')
        n_digits.insert(10, '%10.2f')
        n_digits.append('%10.2f')
    tab = Table(names=names, dtype=['S16']+[np.float64]*len(names[1:]))

    for (i, col) in enumerate(tab.colnames[1:]):
        tab[col].info.format = n_digits[i]

    return tab


def apply_1L2S_criteria(res_1L2S):
    """
    Apply criteria to the 1L2S results, before reading 2L1S results
    """
    chi2, dof, best_params, best_fluxes = res_1L2S['Best model'].values()
    chi2_dof = chi2 / dof
    t_E = best_params['t_E']
    max_u_0 = max(best_params['u_0_1'], best_params['u_0_2'])
    flux_s = min(best_fluxes['flux_s1_1'], best_fluxes['flux_s2_1'])

    event_id = res_1L2S['event_id'].split('_OGLE')[0]
    if event_id == "BLG508_17_126114":
        # obvious event that requires the only exception: good t_E_2L1S...
        return (chi2_dof < 2 and flux_s > 0)
    if event_id == "BLG506_23_129457":
        # obvious event twin with BLG506.23.129983, excluded by hand...
        return False

    return (chi2_dof < 2 and t_E < 200 and max_u_0 > 0.01 and flux_s > 0)


def check_n_datapoints(event_id):
    """
    Check if the number of datapoints is greater than 120.
    Minimum n_data in event_finder is 30, but here we need to increase.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    options = ["BLG50X1", "BLG50X2", "BLG50X3", "lcmin3to5", "lcall1",
               "lcall2", "lcall3"]
    for dataset in options:
        fname = os.path.join(path, f'phot/phot_{dataset}', f'{event_id}.dat')
        if os.path.exists(fname):
            break

    mm_data = mm.MulensData(file_name=fname)
    return mm_data.n_epochs > 120


def apply_more_criteria(args, res_1L2S, res_2L1S):
    """
    Apply further criteria to both 1L2S and 2L1S results.
    """
    chi2_1L2S, dof = [res_1L2S['Best model'][key] for key in ('chi2', 'dof')]
    chi2_2L1S = res_2L1S['Best model']['chi2']
    dict_ = res_1L2S if chi2_1L2S / dof < chi2_2L1S / (dof-1) else res_2L1S
    min_chi2 = min(chi2_1L2S / dof, chi2_2L1S / (dof-1))
    t_E_best = dict_['Best model']['Parameters']['t_E']

    params = res_1L2S['Best model']['Parameters']
    tau_1L2S = (params['t_0_1'] - params['t_0_2']) / params['t_E']
    max_u_0_i = max(params['u_0_1'], params['u_0_2'])
    sep_1L2S = np.sqrt(tau_1L2S**2 + max_u_0_i**2)
    s_2L1S = res_2L1S['Best model']['Parameters']['s']

    t_0_diff = abs(params['t_0_1'] - params['t_0_2'])
    t_0_thr = (t_0_diff >= 9 and t_0_diff < 2800)
    sflux_thr = res_2L1S['Best model']['Fluxes']['flux_s_1'] > 0
    bflux_1L2S = res_1L2S['Best model']['Fluxes']['flux_b_1']
    bflux_2L1S = res_2L1S['Best model']['Fluxes']['flux_b_1']
    bflux_thr = max(bflux_1L2S, bflux_2L1S) > -8.0
    n_data = check_n_datapoints(res_1L2S['event_id'])

    if not args.lowcadence:
        new_thrs = (sep_1L2S < 120 and s_2L1S < 250 and max_u_0_i < 1.6)
        return (new_thrs and sflux_thr and t_0_thr and bflux_thr and n_data)

    else:
        prev_thr = (min_chi2 <= 1.52 and t_E_best <= 120)
        new_thrs = (sep_1L2S < 70 and s_2L1S < 70 and max_u_0_i < 1.2)
        return (prev_thr and sflux_thr and t_0_thr and bflux_thr and new_thrs
                and n_data)


def read_results_EMCEE(args, dir_1L2S):
    """
    Read EMCEE results for 1L2S and 2L1S models and call criteria functions.
    For 2L1S model, between or beyond solution is chosen based on chi2.
    """
    dir_1L2S += "_results.yaml"
    with open(dir_1L2S, encoding='utf-8') as in_1L2S:
        dict_1L2S = yaml.safe_load(in_1L2S)
    if not apply_1L2S_criteria(dict_1L2S):
        return None, None

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
    if not apply_more_criteria(args, dict_1L2S, dict_2L1S):
        return None, None

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
    s_flux = [val for key, val in best['Fluxes'].items() if 'flux_s' in key]
    if any(np.array(s_flux) < 0):
        name = res_fit.get('event_id', best['Parameters'].get('t_0'))
        print(name, 'has negative s_flux.')

    best_pars = list(best['Parameters'].values())
    best_pars = [round(p, r) for p, r in zip(best_pars, round_params)]
    t_E_sig = np.mean([fitted_pars['t_E'][1], -fitted_pars['t_E'][2]])
    bflux = round(best['Fluxes']['flux_b_1'], 5)
    bflux_sig = np.mean([fitted_bflux[1], -fitted_bflux[2]])
    t_E_sig, bflux_sig = round(t_E_sig, 2), round(bflux_sig, 5)

    # line = [*best_pars, t_E_sig, bflux, bflux_sig, round(best['chi2'], 2)]
    line = [*best_pars, t_E_sig, bflux, bflux_sig, round(best['chi2']/dof, 5)]
    if method == "ultranest":
        line.append(round(best['ln_ev'][0], 2))

    return line


def fill_table(args, dirs, event_ids):
    """
    Fill the table with the parameters, chi2 and evidence of the best models,
    for each event_id placed in the first column.
    """
    tab = declare_and_format_table(args.method)

    for event_id in event_ids:
        print("Starting:", event_id)
        dir_1L2S_emcee = os.path.join(dirs[1], event_id)
        dict_1L2S, dict_2L1S = read_results_EMCEE(args, dir_1L2S_emcee)
        if dict_1L2S is None:
            if args.verbose:
                print(event_id, "does not meet criteria!")
            continue

        # review the method with Ultranest at some point
        if args.method == "ultranest":
            sigmas_1L2S_2L1S = get_EMCEE_sigmas(dict_1L2S, dict_2L1S)
            res_1L2S = os.path.join(dirs[0], event_id)
            dict_1L2S, dict_2L1S = read_results_UN(res_1L2S)
            change_sigmas_UN(dict_1L2S, dict_2L1S, sigmas_1L2S_2L1S)
        dof = dict_1L2S['Best model']['dof']
        line_1L2S = add_params_to_table(args.method, dict_1L2S, dof)
        line_2L1S = add_params_to_table(args.method, dict_2L1S, dof-1)

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
    new_idx = 14 if method == "ultranest" else 13
    tab.add_column(column_to_move, index=new_idx)
    method_tmp = "_un" if method == "ultranest" else "_emcee"
    fname = os.path.join(path, f'comp_1L2S_2L1S{method_tmp}.txt')
    tab.write(fname, format='ascii.fixed_width', overwrite=True, delimiter='')

    new_header = "# id               t_0_1       u_0_1    t_0_2       " + \
                 "u_0_2    t_E_1L2S  sig_t_E_1  bflux_1L2S  sig_bflux_1" + \
                 "  chi2_1L2S  ln_ev_1L2S  t_0         u_0      t_E_2L1S" + \
                 "  sig_t_E_2  s       q        alpha   bflux_2L1S  " + \
                 "sig_bflux_2  chi2_2L1S  ln_ev_2L1S\n"
    if method == "emcee":
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
    str_split = '-1L2S' if args.method == "ultranest" else '_results'
    event_ids = [f.split(str_split)[0] for f in os.listdir(dirs[idx])
                 if f[0] != '.' and f.endswith(".yaml")]
    event_ids = remove_cvs_mroz20(event_ids)
    tab = fill_table(args, dirs, sorted(event_ids))
    if args.verbose:
        print()
        print(tab, '\n')
    final_setup_and_save_table(path, tab, args.method)
