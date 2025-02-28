"""
Short script to generate the yaml input files for the UltraNest fitting,
using the EMCEE results as input.
The script is called with path to the directory containing the photometry
files and the number of sigma to use for the limits of the parameters.
"""
import argparse
import os
import yaml

from astropy.table import Table
import numpy as np


def parse_arguments():
    """
    Parses command-line arguments and returns them.
    """
    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument('files_dir', type=str,
                        help="Directory or file with the photometry data")
    parser.add_argument('dataset', type=str,
                        help="Dataset to get results: obvious, BLG50X1...")
    lst = ["1l2s", "1L2S", "2l1s", "2L1S", "both"]
    parser.add_argument('which_models', choices=lst, nargs='?', default="both",
                        help="Which models to use: 1L2S, 2L1S or both")
    n_sigma_def = 3 if lst in ["2l1s", "2L1S"] else 5
    parser.add_argument('n_sigma', type=float, nargs='?', default=n_sigma_def,
                        help="Number of sigma to use around EMCEE solution")
    args = parser.parse_args()

    if not os.path.exists(args.files_dir):
        raise ValueError('First argument is not a valid directory or file.')
    if not os.path.isabs(args.files_dir):
        path = os.path.dirname(os.path.abspath(__file__))
        args.files_dir = os.path.join(path, args.files_dir)
    args.which_models = args.which_models.lower()

    return args


def get_paths(dataset):
    """
    Get the paths to the results files and the directory to save the
    UltraNest results.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path_point = os.path.normpath(path + os.sep + os.pardir)
    diff_path = path.replace(path_point, '.')

    yaml_in_2L1S = '{:}-2L1S_traj_beyond.yaml'
    yaml_in_2L1S = os.path.join(path, 'yaml_files_2L1S', dataset, yaml_in_2L1S)
    file_1L2S = os.path.join(path, 'results_1L2S', dataset, 'yaml_results',
                             '{:}_results.yaml')
    file_2L1S = os.path.join(path, 'results_2L1S', dataset,
                             '{:}_2L1S_all_results_{:}.yaml')
    res_1L2S, res_2L1S = [file_1L2S, diff_path], [file_2L1S, diff_path]

    out_1L2S = os.path.join(path, '{:}', dataset, '{:}-1L2S_UltraNest.yaml')
    out_2L1S = os.path.join(path, '{:}', dataset, '{:}-2L1S_UltraNest.yaml')
    out_mkdir = os.path.join(path, 'ultranest_1L2S', dataset, '{:}')

    return (yaml_in_2L1S, res_1L2S, res_2L1S, [out_1L2S, out_2L1S], out_mkdir)


def get_obj_list(files_dir, dataset):
    """
    Retrieve a list of objects to process from a directory or a single file.

    - If the input is a directory, returns a list of photometry files.
    - If it is a photometry file, returns a list with single object ID.
    - If it is a results table, updates the photometry file directory and
    returns the list of object IDs in the table.
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
            files_dir = 'phot/phot_' + dataset

        return files_dir, obj_list

    raise ValueError('Input is neither a file nor a directory.')


def get_xlim(path, obj_id):
    """
    Get the x-axis range from the 2L1S input yaml file.
    """
    yaml_input = path.format(obj_id)
    with open(yaml_input, 'r', encoding='utf-8') as data:
        input_2L1S = yaml.safe_load(data)

    model_methods = input_2L1S['model']['methods']
    xlim = input_2L1S['plots']['best model']['time range']

    return model_methods, xlim


def get_1L2S_params(args, path, diff_path, obj_id):
    """
    Get the parameters from the 1L2S model fitted with EMCEE.
    A list is returned with the path to the photometry file, the limits for
    each parameter, the path to save the UltraNest results, and the event
    identifier.
    """
    with open(path.format(obj_id), 'r', encoding='utf-8') as data:
        results_1L2S = yaml.safe_load(data)
    best_vals = results_1L2S['Best model']['Parameters'].values()
    perc_vals = results_1L2S['Fitted parameters'].values()

    list_1L2S = [os.path.join(diff_path, args.files_dir, obj_id + '.dat')]
    for (best, perc) in zip(best_vals, perc_vals):
        mean_std = np.mean([perc[1], -perc[2]])
        lower_limit = max(0, round(best - args.n_sigma*mean_std, 5))
        upper_limit = round(best + args.n_sigma*mean_std, 5)
        list_1L2S.append([lower_limit, upper_limit])

    path_yaml = os.path.join(diff_path, 'ultranest_1L2S', args.dataset)
    list_1L2S += [path_yaml, obj_id, '']

    return list_1L2S


def get_2L1S_params(args, path, diff_path, obj_id):
    """
    Get the parameters from the 2L1S model, selecting the smaller chi2
    between the trajectory `between` and `beyond` the lenses.
    A similar list to get_1L2S_params() is returned.
    """
    with open(path.format(obj_id, 'between'), 'r', encoding='utf-8') as data:
        results_2L1S = yaml.safe_load(data)
    with open(path.format(obj_id, 'beyond'), 'r', encoding='utf-8') as data:
        temp = yaml.safe_load(data)
        if temp['Best model']['chi2'] < results_2L1S['Best model']['chi2']:
            results_2L1S = temp

    list_2L1S = [os.path.join(diff_path, args.files_dir, obj_id + '.dat')]
    best_vals = results_2L1S['Best model']['Parameters'].values()
    perc_vals = results_2L1S['Fitted parameters'].values()
    for (best, perc) in zip(best_vals, perc_vals):
        mean_std = np.mean([perc[1], -perc[2]])
        lower_limit = max(0, round(best - args.n_sigma*mean_std, 5))
        upper_limit = round(best + args.n_sigma*mean_std, 5)
        list_2L1S.append([lower_limit, upper_limit])

    path_yaml = os.path.join(diff_path, 'ultranest_2L1S', args.dataset)
    list_2L1S += ['', path_yaml, obj_id, '']
    list_2L1S.append(results_2L1S)

    return list_2L1S


def update_mag_methods(mag_methods, list_1L2S):
    """
    Update the methods to calculate the magnification in the 2L1S model,
    if the abs(t_0_1-t_0_2) > 1000 and t_E < 30 days.
    """
    t_0_1 = np.mean(list_1L2S[1])
    t_0_2 = np.mean(list_1L2S[3])
    t_E = np.mean(list_1L2S[5])

    if abs(t_0_1 - t_0_2) > 1000 and t_E < 30:
        t_E_min = max(5*t_E, 50)
        mag_methods = mag_methods.split()
        mag_methods.insert(2, '%.1f' % (min([t_0_1, t_0_2]) + t_E_min))
        mag_methods.insert(3, 'point_source_point_lens')
        mag_methods.insert(4, '%.1f' % (max([t_0_1, t_0_2]) - t_E_min))
        mag_methods.insert(5, 'point_source')
        mag_methods = ' '.join(mag_methods)

    return mag_methods


def update_large_sigmas(list_2L1S):
    """
    Update the prior limits in cases with large uncertainties (s, q, alpha).
    """
    results_2L1S = list_2L1S.pop(-1)
    perc = results_2L1S['Fitted parameters']
    sig_alpha = np.mean([perc['alpha'][1], -perc['alpha'][2]])
    sig_q = np.mean([perc['q'][1], -perc['q'][2]]) / perc['q'][0]
    sig_s = np.mean([perc['s'][1], -perc['s'][2]]) / perc['s'][0]

    if sig_alpha > 3 and (sig_q > 0.5 or sig_s > 2):
        new_n_sigma = 0.1
        best = results_2L1S['Best model']['Parameters'].values()

        for i, (best, perc) in enumerate(zip(best, perc.values())):
            mean_std = np.mean([perc[1], -perc[2]])
            lower_limit = max(0, round(best - new_n_sigma*mean_std, 5))
            upper_limit = round(best + new_n_sigma*mean_std, 5)
            list_2L1S[i+1] = [lower_limit, upper_limit]
        return (list_2L1S, new_n_sigma)

    return (list_2L1S, None)


def save_yaml_UN_inputs(path, obj_id, list_1L2S=None, list_2L1S=None,
                        new_sigma=None):
    """
    Save yaml inputs for the 1L2S and 2L1S model, to apply UltraNest.
    """
    n_sigma = new_sigma if new_sigma is not None else args.n_sigma
    n_sigma = str(n_sigma).replace('.', 'p')
    if list_1L2S is not None:
        fname = 'template_1L2S_UltraNest.yaml'
        with open(fname, 'r', encoding='utf-8') as t_file:
            temp_UN = t_file.read()
        yaml_path = path[0].format(list_1L2S[6].split('/')[-2], obj_id)
        yaml_path = yaml_path.replace('.yaml', f'_{n_sigma}sig.yaml')
        with open(yaml_path, 'w') as yaml_input:
            yaml_input.write(temp_UN.format(*list_1L2S))

    if list_2L1S is not None:
        fname = 'template_2L1S_UltraNest.yaml'
        with open(fname, 'r', encoding='utf-8') as t_file:
            temp_UN = t_file.read()
        yaml_path = path[1].format(list_2L1S[8].split('/')[-2], obj_id)
        yaml_path = yaml_path.replace('.yaml', f'_{n_sigma}sig.yaml')
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

    args = parse_arguments()
    all_paths = get_paths(args.dataset)
    args.files_dir, obj_list = get_obj_list(args.files_dir, args.dataset)
    list_1L2S, list_2L1S, sig = None, None, None

    for obj_id in obj_list:
        mag_method, xlim = get_xlim(all_paths[0], obj_id)
        list_1L2S = get_1L2S_params(args, *all_paths[1], obj_id)
        list_1L2S[-1] = xlim

        if args.which_models in ["both", "2l1s"]:
            list_2L1S = get_2L1S_params(args, *all_paths[2], obj_id)
            new_methods = update_mag_methods(mag_method, list_1L2S)
            list_2L1S, sig = update_large_sigmas(list_2L1S)
            list_2L1S[7], list_2L1S[-1] = new_methods, xlim
        if args.which_models == "2l1s":
            list_1L2S = None
        save_yaml_UN_inputs(all_paths[3], obj_id, list_1L2S, list_2L1S, sig)
        create_results_dir(all_paths[4], obj_id)
        print("Done for", obj_id)
