"""
Short script to generate the yaml input files for the UltraNest fitting,
using the EMCEE results as input.
The script is called with path to the directory containing the photometry
files and the number of sigma to use for the limits of the parameters.
"""
import os
import sys
import yaml

from astropy.table import Table
import numpy as np


def get_paths(files_dir, dataset):
    """
    Get the paths to the results files and the directory to save the
    UltraNest results.
    STILL NEED TO MAKE MEANINGFUL NAMES...
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path_point = os.path.normpath(path + os.sep + os.pardir)
    diff_path = path.replace(path_point, '.')

    files_dir = os.path.join(path, files_dir)
    yaml_in_2L1S = '{:}-2L1S_traj_beyond.yaml'
    yaml_in_2L1S = os.path.join(path, 'yaml_files_2L1S', dataset, yaml_in_2L1S)
    file_1L2S = os.path.join(path, 'results_1L2S', dataset, 'yaml_results',
                             '{:}_results.yaml')
    file_2L1S = os.path.join(path, 'results_2L1S', dataset,
                             '{:}_2L1S_all_results_{:}.yaml')

    out_1L2S = os.path.join(path, '{:}', dataset, '{:}-1L2S_UltraNest.yaml')
    out_2L1S = os.path.join(path, '{:}', dataset, '{:}-2L1S_UltraNest.yaml')
    out_mkdir = os.path.join(path, 'ultranest_1L2S', dataset, '{:}')

    return (files_dir, yaml_in_2L1S, [file_1L2S, diff_path],
            [file_2L1S, diff_path], [out_1L2S, out_2L1S], out_mkdir)


def get_obj_list(files_dir):
    """
    Get the list of objects to process, from a directory or a single file.
    """
    if os.path.isdir(files_dir):
        obj_list = [f for f in os.listdir(files_dir) if f[0] != '.']
        obj_list = [item.split('.')[0] for item in sorted(obj_list)]
        return files_dir, obj_list

    elif os.path.isfile(files_dir):
        if files_dir.endswith('_OGLE.dat'):
            obj_list = [os.path.splitext(os.path.basename(files_dir))[0]]
            files_dir = os.path.dirname(files_dir)
        elif 'comp_' in files_dir and files_dir.endswith('.txt'):
            tab = Table.read(files_dir, format='ascii')
            obj_list = [line['id'] + '_OGLE' for line in tab]
            files_dir = '/Users/rapoliveira/postdoc/MulensModel_raphael/' + \
                        'exploring_MulensModel/OGLE-evfinder/phot/phot_' + \
                        sys.argv[2]

        return files_dir, obj_list

    raise ValueError('Input is neither a file nor a directory.')


def get_xlim(path, obj_id):
    """
    Get the xlim from the 2L1S input yaml file.
    """
    yaml_input = path.format(obj_id)
    with open(yaml_input, 'r', encoding='utf-8') as data:
        input_2L1S = yaml.safe_load(data)

    model_methods = input_2L1S['model']['methods']
    xlim = input_2L1S['plots']['best model']['time range']

    return model_methods, xlim


def get_1L2S_params(path, diff_path, obj_id, files_dir):
    """
    Get the parameters from the fitted 1L2S model.
    A list is returned with the path to the photometry file, the limits for
    each parameter, the path to save the UltraNest results, and the event
    identifier.
    """
    dataset = sys.argv[2]
    n_sigma = float(sys.argv[3])

    with open(path.format(obj_id), 'r', encoding='utf-8') as data:
        results_1L2S = yaml.safe_load(data)
    best_vals = results_1L2S['Best model']['Parameters'].values()
    perc_vals = results_1L2S['Fitted parameters'].values()

    list_1L2S = [os.path.join(diff_path, files_dir, obj_id + '.dat')]
    for (best, perc) in zip(best_vals, perc_vals):
        mean_std = np.mean([perc[1], -perc[2]])
        lower_limit = max(0, round(best - n_sigma*mean_std, 5))
        upper_limit = round(best + n_sigma*mean_std, 5)
        list_1L2S.append([lower_limit, upper_limit])

    path_yaml = os.path.join(diff_path, 'ultranest_1L2S', dataset)
    list_1L2S += [path_yaml, obj_id, '']

    return list_1L2S


def get_2L1S_params(path, diff_path, obj_id, files_dir):
    """
    Get the parameters from the 2L1S model, selecting the smaller chi2
    between the trajectory `between` and `beyond` the lenses.
    A similar list to get_1L2S_params() is returned.
    """
    n_sigma = float(sys.argv[3])
    with open(path.format(obj_id, 'between'), 'r', encoding='utf-8') as data:
        results_2L1S = yaml.safe_load(data)
    with open(path.format(obj_id, 'beyond'), 'r', encoding='utf-8') as data:
        temp = yaml.safe_load(data)
        if temp['Best model']['chi2'] < results_2L1S['Best model']['chi2']:
            results_2L1S = temp

    list_2L1S = [os.path.join(diff_path, files_dir, obj_id + '.dat')]
    best_vals = results_2L1S['Best model']['Parameters'].values()
    perc_vals = results_2L1S['Fitted parameters'].values()
    for (best, perc) in zip(best_vals, perc_vals):
        mean_std = np.mean([perc[1], -perc[2]])
        lower_limit = max(0, round(best - n_sigma*mean_std, 5))
        upper_limit = round(best + n_sigma*mean_std, 5)
        list_2L1S.append([lower_limit, upper_limit])

    path_yaml = os.path.join(diff_path, 'ultranest_2L1S', dataset)
    list_2L1S += ['', path_yaml, obj_id, '']

    return list_2L1S


def save_yaml_inputs(path, obj_id, list_1L2S=None, list_2L1S=None):
    """
    Save yaml inputs for the 1L2S and 2L1S model, to apply UltraNest.
    """
    if list_1L2S is not None:
        fname = 'template_1L2S_UltraNest.yaml'
        with open(fname, 'r', encoding='utf-8') as t_file:
            temp_UN = t_file.read()
        yaml_path = path[0].format(list_1L2S[6].split('/')[-2], obj_id)
        with open(yaml_path, 'w') as yaml_input:
            yaml_input.write(temp_UN.format(*list_1L2S))

    if list_2L1S is not None:
        fname = 'template_2L1S_UltraNest.yaml'
        with open(fname, 'r', encoding='utf-8') as t_file:
            temp_UN = t_file.read()
        yaml_path = path[1].format(list_2L1S[8].split('/')[-2], obj_id)
        with open(yaml_path, 'w') as yaml_input:
            yaml_input.write(temp_UN.format(*list_2L1S))


def create_results_dir(path, obj_id):
    """
    Create a directory to save the results of the UltraNest fitting.
    """
    results_dir = path.format(obj_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir = results_dir.replace('1L2S', '2L1S')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


if __name__ == '__main__':

    # still add function to parse_args and organize paths name!!!
    files_dir = sys.argv[1]
    dataset = sys.argv[2]
    which_models = "both" if len(sys.argv) < 5 else sys.argv[4].lower()
    list_1L2S, list_2L1S = None, None

    all_paths = get_paths(files_dir, dataset)
    files_dir, obj_list = get_obj_list(all_paths[0])
    for obj_id in obj_list:
        model_methods, xlim = get_xlim(all_paths[1], obj_id)
        if which_models in ["both", "1l2s"]:
            list_1L2S = get_1L2S_params(*all_paths[2], obj_id, files_dir)
            list_1L2S[-1] = xlim
        if which_models in ["both", "2l1s"]:
            list_2L1S = get_2L1S_params(*all_paths[3], obj_id, files_dir)
            list_2L1S[7], list_2L1S[-1] = model_methods, xlim
        save_yaml_inputs(all_paths[4], obj_id, list_1L2S, list_2L1S)
        create_results_dir(all_paths[5], obj_id)
        print("Done for", obj_id)
