"""
Write documentation later...
"""
from astropy.table import Table
import numpy as np
import os
import yaml


def read_results(res_1L2S, res_2L1S):
    """_summary_

    Args:
        res_1L2S (_type_): _description_
        res_2L1S (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if res_1L2S[:16] == res_2L1S[:16]:
        event_id = '.'.join(res_1L2S.split('_')[:3])
    else:
        raise ValueError('The two files do not correspond to the same event.')

    # Previous case: 1L2S result from EMCEE (between and beyond)
    # with open(f'{dir_1}{res_1L2S}', encoding='utf-8') as in_1L2S:
    #     res_1L2S = yaml.safe_load(in_1L2S)

    # New code: directly UltraNest results (best, chi2 and evidence)
    suffix = '_1L2S_all_results_UN.yaml'
    fname = os.path.join(dir_1, res_1L2S, res_1L2S + suffix)
    with open(fname, encoding='utf-8') as in_1L2S:
        res_1L2S = yaml.safe_load(in_1L2S)

    # Previous case: 2L1S result from EMCEE (between and beyond)
    # with open(f'{dir_2}{res_2L1S}', encoding='utf-8') as in_2L1S:
    #     res_2L1S_between = yaml.safe_load(in_2L1S)
    #     chi2_bet = res_2L1S_between['Best model']['chi2']
    # res_2L1S = res_2L1S.replace('between', 'beyond')
    # with open(f'{dir_2}{res_2L1S}', encoding='utf-8') as in_2L1S:
    #     res_2L1S_beyond = yaml.safe_load(in_2L1S)
    #     chi2_bey = res_2L1S_beyond['Best model']['chi2']
    # res_2L1S = res_2L1S_between if chi2_bet < chi2_bey else res_2L1S_beyond

    # New code: directly UltraNest results (best, chi2 and evidence)
    suffix = '_2L1S_all_results_UN.yaml'
    fname = os.path.join(dir_2, res_2L1S, res_2L1S + suffix)
    with open(fname, encoding='utf-8') as in_2L1S:
        res_2L1S = yaml.safe_load(in_2L1S)

    return event_id, res_1L2S, res_2L1S


def add_params_to_table(res_fit):
    """_summary_

    Args:
        res_1L2S (_type_): _description_
        res_2L1S (_type_): _description_
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
    chi2, evid = round(best['chi2'], 2), round(best['ln_ev'][0], 2)

    line = [*best_params, round(t_E_sig, 2), bflux, round(bflux_sig, 2), chi2,
            evid]
    return line


if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(__file__))
    breakpoint()

    # dir_1 = f'{path}/results_1L2S/yaml_results/'
    dir_1 = f'{path}/ultranest_1L2S/'
    # dir_2 = f'{path}/results_2L1S/'
    dir_2 = f'{path}/ultranest_2L1S/'
    # results_1L2S = sorted([f for f in os.listdir(dir_1) if f[0] != '.'])
    results_1L2S = [f for f in os.listdir(dir_1) if f[0] != '.' and
                    not f.endswith(".yaml")]
    # results_2L1S = [f for f in os.listdir(dir_2) if f[0] != '.' and
    #                 f.endswith('between.yaml')]
    results_2L1S = [f for f in os.listdir(dir_2) if f[0] != '.' and
                    not f.endswith(".yaml")]
    results_1L2S, results_2L1S = sorted(results_1L2S), sorted(results_2L1S)
    names = ['id', 't_0_1', 'u_0_1', 't_0_2', 'u_0_2', 't_E_1L2S', 'sig_t_E_1',
             'bflux_1L2S', 'sig_bflux_1', 'chi2_1L2S', 'ln_ev_1L2S',
             't_0', 'u_0', 't_E_2L1S', 's', 'q', 'alpha', 'sig_t_E_2',
             'bflux_2L1S', 'sig_bflux_2', 'chi2_2L1S', 'ln_ev_2L1S']
    tab = Table(names=names, dtype=['str']+[np.float64]*21)

    for (res_1L2S, res_2L1S) in zip(results_1L2S, sorted(results_2L1S)):
        event_id, res_1L2S, res_2L1S = read_results(res_1L2S, res_2L1S)
        line_1L2S = add_params_to_table(res_1L2S)
        line_2L1S = add_params_to_table(res_2L1S)
        tab.add_row([event_id, *line_1L2S, *line_2L1S])
    # tab.round(5)
    tab.write(f'{path}/comp_1L2S_2L1S.txt', format='ascii.commented_header',
              overwrite=True)
    breakpoint()
