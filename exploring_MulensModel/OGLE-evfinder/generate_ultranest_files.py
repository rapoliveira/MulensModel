"""
Short script to generate the yaml input files for the UltraNest fitting,
using the EMCEE results as input.
The script is called with path to the directory containing the photometry
files and the number of sigma to use for the limits of the parameters.
"""
import os
import sys
import yaml

import numpy as np


def get_1L2S_params(path, diff_path, obj_id, n_sigma):
    """
    Get the parameters from the fitted 1L2S model.
    A list is returned with the path to the photometry file, the limits for
    each parameter, the path to save the UltraNest results, and the event
    identifier.
    """
    results_file = 'results_1L2S/yaml_results/{:}_results.yaml'
    results_file = os.path.join(path, results_file.format(obj_id))
    with open(results_file, 'r', encoding='utf-8') as data:
        results_1L2S = yaml.safe_load(data)

    list_1L2S = [os.path.join(diff_path, sys.argv[1], obj_id + '.dat')]
    best_vals = results_1L2S['Best model']['Parameters'].values()
    perc_vals = results_1L2S['Fitted parameters'].values()
    for (best, perc) in zip(best_vals, perc_vals):
        mean_std = np.mean([perc[1], -perc[2]])
        lower_limit = max(0, round(best - n_sigma*mean_std, 5))
        upper_limit = round(best + n_sigma*mean_std, 5)
        list_1L2S.append([lower_limit, upper_limit])

    path_yaml = os.path.join(diff_path, 'ultranest_1L2S')
    list_1L2S += [path_yaml, obj_id, '']

    return list_1L2S


def get_2L1S_params(path, diff_path, obj_id, n_sigma):
    """
    Get the parameters from the 2L1S model, selecting the smaller chi2
    between the trajectory `between` and `beyond` the lenses.
    A similar list to get_1L2S_params() is returned.
    """
    results_file = 'results_2L1S/{:}_2L1S_all_results_{:}.yaml'
    fname_1 = os.path.join(path, results_file.format(obj_id, 'between'))
    fname_2 = os.path.join(path, results_file.format(obj_id, 'beyond'))
    with open(fname_1, 'r', encoding='utf-8') as data:
        results_2L1S = yaml.safe_load(data)
    with open(fname_2, 'r', encoding='utf-8') as data:
        temp = yaml.safe_load(data)
        if temp['Best model']['chi2'] < results_2L1S['Best model']['chi2']:
            results_2L1S = temp

    list_2L1S = [os.path.join(diff_path, sys.argv[1], obj_id + '.dat')]
    best_vals = results_2L1S['Best model']['Parameters'].values()
    perc_vals = results_2L1S['Fitted parameters'].values()
    for (best, perc) in zip(best_vals, perc_vals):
        mean_std = np.mean([perc[1], -perc[2]])
        lower_limit = max(0, round(best - n_sigma*mean_std, 5))
        upper_limit = round(best + n_sigma*mean_std, 5)
        list_2L1S.append([lower_limit, upper_limit])

    path_yaml = os.path.join(diff_path, 'ultranest_2L1S')
    list_2L1S += ['', path_yaml, obj_id, '']

    return list_2L1S


def get_xlim(path, obj_id):
    """
    Get the xlim from the 2L1S input yaml file.
    """
    yaml_input = obj_id + '-2L1S_traj_beyond.yaml'
    yaml_input = os.path.join(path, 'yaml_files_2L1S', yaml_input)
    with open(yaml_input, 'r', encoding='utf-8') as data:
        input_2L1S = yaml.safe_load(data)

    model_methods = input_2L1S['model']['methods']
    xlim = input_2L1S['plots']['best model']['time range']

    return model_methods, xlim


def save_yaml_inputs(path, obj_id, list_1L2S, list_2L1S):
    """
    Save yaml inputs for the 1L2S and 2L1S model, to apply UltraNest.
    """
    with open('template_1L2S_UltraNest.yaml', 'r', encoding='utf-8') as t_file:
        temp_UN = t_file.read()
    yaml_file = obj_id + '-1L2S_UltraNest.yaml'
    yaml_path = os.path.join(path, list_1L2S[6].split('/')[-1], yaml_file)
    with open(yaml_path, 'w') as yaml_input:
        yaml_input.write(temp_UN.format(*list_1L2S))

    with open('template_2L1S_UltraNest.yaml', 'r', encoding='utf-8') as t_file:
        temp_UN = t_file.read()
    yaml_path = yaml_path.replace('1L2S', '2L1S')
    with open(yaml_path, 'w') as yaml_input:
        yaml_input.write(temp_UN.format(*list_2L1S))


def create_results_dir(path, obj_id):
    """
    Create a directory to save the results of the UltraNest fitting.
    """
    results_dir = os.path.join(path, 'ultranest_1L2S', obj_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir = os.path.join(path, 'ultranest_2L1S', obj_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(__file__))
    path_point = os.path.normpath(path + os.sep + os.pardir)
    diff_path = path.replace(path_point, '.')

    if os.path.isdir(sys.argv[1]):
        obj_list = [f for f in os.listdir(sys.argv[1]) if f[0] != '.']
        obj_list = [item.split('.')[0] for item in sorted(obj_list)]
    elif os.path.isfile(sys.argv[1]):
        obj_list = [os.path.basename(sys.argv[1])]
    else:
        raise ValueError('Input is neither a file nor a directory.')

    n_sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    for obj_id in obj_list:
        # for obj_id in obj_list[13:14]:  # [:1], [12:14]
        list_1L2S = get_1L2S_params(path, diff_path, obj_id, n_sigma)
        list_2L1S = get_2L1S_params(path, diff_path, obj_id, n_sigma)
        model_methods, xlim = get_xlim(path, obj_id)
        list_1L2S[-1] = xlim
        list_2L1S[7], list_2L1S[-1] = model_methods, xlim
        save_yaml_inputs(path, obj_id, list_1L2S, list_2L1S)
        create_results_dir(path, obj_id)
        print("Done for", obj_id)
