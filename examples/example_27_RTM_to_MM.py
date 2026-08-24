"""
This example takes the results from RTModel in form of txt files (either final
or all models) and makes YAML files compliant with the MulensModel standard.

The standard example creates a short YAML file with the parameters and values,
while the extended version (--extended) creates a complete YAML file to be used
in example_16, with model methods, starting parameters, ranges and paths to the
plots. In this case, an additional option allows to plot the RTModel result to
a file (-p) or to also display it on screen (-p2).
"""

import argparse
import copy
import glob
import numpy as np
import os
import warnings
import yaml

from example_16.ulens_model_fit import UlensModelFit


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate MulensModel parameters from RTModel results.")
    parser.add_argument("event", help="Directory containing the RTModel results.")
    parser.add_argument("models", choices=["all", "final"],
                        help="Select models from either all Models or FinalModels folders.")
    msg = "Include additional fitting information in the YAML file (model methods, starting parameters and ranges)."
    parser.add_argument("--extended", action="store_true", help=msg)

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("-p", "--plot", action="store_true", help="Plot the RTModel result to file.")
    msg = "Plot the RTModel result to file and display it on screen."
    plot_group.add_argument("-p2", "--plot-screen", action="store_true", help=msg)

    args = parser.parse_args()
    model_folder = "FinalModels" if args.models == "final" else "Models"
    models = glob.glob(os.path.join(args.event, model_folder, "*"))
    if (args.plot or args.plot_screen) and not args.extended:
        parser.error("-p/-p2 can only be used together with --extended")
    plot = "both" if args.plot_screen else "file" if args.plot else None

    return args.event, models, args.extended, plot


def get_yaml_name(event, model):
    """
    Creates yaml file names to save the new parameters
    """
    nazwa_yaml = str(event) + '_' + str(model) + '.yaml'
    return nazwa_yaml


def get_ids(list_of_models):
    """
    Extracts the model code from the file name
    """
    ids = []
    for model in list_of_models:
        ids.append(model[-12:-4])
    return ids


def get_parameters_names(ID):
    """
    Based on the shortened model code, gives the parameters names
    """
    model_id_short = ID[0:2]
    if model_id_short in ['BO', 'LK', 'TS', 'TX', 'TO']:
        warnings.warn("Transformation for {:} model is not accurate!".format(model_id_short))
        print("This model category: {:}, includes one of the following: xallarap, triple lens or elliptical orbital "
              "motion.\nThese effects are not yet included in this example.".format(model_id_short))

    model_and_parnames = {
        'PS': ['u_0', 't_E', 't_0', 'rho'],
        'PX': ['u_0', 't_E', 't_0', 'rho', 'pi_E_N', 'pi_E_E'],
        'BS': ['t_E', 'FR', 'u_0_1', 'u_0_2', 't_0_1', 't_0_2', 'rho_1'],
        'BO': ['t_E', 'FR', 'u_0_1', 'u_0_2', 't_0_1', 't_0_2', 'rho_1', 'pi_E_N', 'pi_E_E', 'gamma1', 'gamma2',
               'gammaz'],
        'LS': ['s', 'q', 'u_0', 'alpha', 'rho', 't_E', 't_0'],
        'LX': ['s', 'q', 'u_0', 'alpha', 'rho', 't_E', 't_0', 'pi_E_N', 'pi_E_E'],
        'LO': ['s', 'q', 'u_0', 'alpha', 'rho', 't_E', 't_0', 'pi_E_N', 'pi_E_E', 'gamma1', 'gamma2', 'gammaz'],
        'LK': ['s', 'q', 'u_0', 'alpha', 'rho', 't_E', 't_0', 'pi_E_N', 'pi_E_E', 'gamma1', 'gamma2', 'gammaz',
               'sz_s', 'a_s3d'],
        'TS': ['s', 'q', 'u_0', 'alpha', 'rho', 't_E', 't_0', 's2', 'q2', 'beta'],
        'TX': ['s', 'q', 'u_0', 'alpha', 'rho', 't_E', 't_0', 's2', 'q2', 'beta', 'pi_E_N', 'pi_E_E']}

    return model_and_parnames[model_id_short]


def get_MMformat_for_orbital_motion(params):
    """
    Changes the units of the orbital motion parameters
    """
    ds_dt = params["s"] * params["gamma1"] * 365.25
    ds_z_dt = params["gammaz"] * params["s"] * 365.25
    dalpha_dt = params["gamma2"] * 365.25 * 180/np.pi
    params.update({"ds_dt": ds_dt})
    params.update({"ds_z_dt": ds_z_dt})
    params.update({"dalpha_dt": -dalpha_dt})
    new_dict = params.copy()
    new_dict.pop("gamma1")
    new_dict.pop("gamma2")
    new_dict.pop("gammaz")
    return new_dict


def check_t0_format(t0):
    """
    Checks if t0 is in the MM format
    """
    if t0 < 2450000:
        t0 = t0 + 2450000
    return t0


def get_to_MM_format(params, model):
    """
    Changes the parameters to MM format, adds t_0_par if needed
    """
    if model[0] == "B":
        t01 = params["t_0_1"]
        t01 = check_t0_format(t01)
        params.update({"t_0_1": t01})
        t02 = params["t_0_2"]
        t02 = check_t0_format(t02)
        params.update({"t_0_2": t02})
    else:
        t0 = params["t_0"]
        t0 = check_t0_format(t0)
        params.update({"t_0": t0})
    if model[0] == "L":
        alpha = params["alpha"] * 180/np.pi
        params.update({"alpha": alpha})
    if model[0:2] == "LO" or model[0:2] == "LK":
        params = get_MMformat_for_orbital_motion(params)
    if model[1] in ["X", "O", "K"]:
        if model[0] == "B":
            t_0_par = params["t_0_1"]
        else:
            t_0_par = params["t_0"]
        params.update({"t_0_par": t_0_par})
    return params


def transform_RTM_to_MM(models_files):
    """
    Read RTModel format files and transform models to MulensModel format.
    Parameters :
        models_files: list of str
            paths to files with RTModel results
    """
    models_ids = get_ids(models_files)
    out = []
    for (model, model_id) in zip(models_files, models_ids):
        parnames = get_parameters_names(model_id)
        with open(model) as f:
            line = f.readlines()[0]

        chunks = line.split(' ')
        values = [float(v) for v in chunks]
        par_dict = dict(zip(parnames, values))
        par_dict = get_to_MM_format(par_dict, model_id)
        out.append(par_dict)

    return out


def get_data_dir_and_check(folder_name, checks=True):
    """
    Get the directory path for the data file (OGLE format: HJD, mag, err) and
    check if it exists.
    If the file does not exist, issue a warning and create it from the RTModel
    data file (TelescopeA.dat, with columns Mag, err, HJD).

    Note: All produced paths are relative to example_16's folder, which is one
    level up from the current working directory.
    """
    event_name = os.path.basename(folder_name)
    data_dir = os.path.join("..", folder_name, "Data", f"{event_name}_OGLE.txt")

    if checks:
        temp_dir = data_dir.removeprefix("../")
        if not os.path.isfile(temp_dir):
            warnings.warn(f"Data file not found: {temp_dir}. Attempting to create it from RTModel data.")
            rtm_data_file = os.path.join(folder_name, "Data", "TelescopeA.dat")
            rtm_data = np.loadtxt(rtm_data_file)
            np.savetxt(temp_dir, rtm_data[:, [2, 0, 1]], fmt=["%12.6f", "%9.6f", "%9.6f"])

        elif np.loadtxt(temp_dir, max_rows=1)[0] > 2450000:
            raise ValueError(f"Times in {temp_dir} are not subtracted by 2450000. Please check the file content.")

    return f"    - {{file_name: {data_dir}, add_2450000: True, phot_fmt: mag}}\n"


def get_model_methods(model_type, par_dict):
    """
    Write later, based on the notes of each model type below...

    # Example for LS, LX, LO:
    # methods: 2455895. point_source 2456013. VBBL 2456036. point_source 2456150.
    # default method: point_source_point_lens
    # parameters: s q u_0 alpha rho t_E t_0 pi_E_N pi_E_E ds_dt ds_z_dt dalpha_dt

    # Example for BS, BO:
    # methods source 1:
    #     2457913. finite_source_uniform_Gould94 2458062.
    # default method: point_source
    # parameters: t_E u_0_1 u_0_2 t_0_1 t_0_2 rho_1 (pi_E_N pi_E_E gamma1 gamma2 gammaz)

    # Example for PS, PX:
    # methods: 2458499. finite_source_uniform_Gould94 2458594.
    # default method: point_source
    # parameters: u_0 t_E t_0 rho (pi_E_N pi_E_E)
    """
    t_0_name = "t_0" if model_type not in ['BS', 'BO'] else "t_0_1"
    rho_value = par_dict['rho'] if 'rho' in par_dict else par_dict['rho_1']
    delta = 3*par_dict['t_E'] if model_type in ['LS', 'LX', 'LO'] else max(0.1, 5*rho_value*par_dict['t_E'])
    left_limit = "%.1f" % (par_dict[t_0_name] - delta)
    right_limit = "%.1f" % (par_dict[t_0_name] + delta)

    if model_type in ['LS', 'LX', 'LO']:
        default_method = "point_source_point_lens"
        methods = " ".join([left_limit, "VBBL", right_limit])
    else:
        default_method = "point_source"
        methods = " ".join([left_limit, "finite_source_uniform_Gould94", right_limit])

    pre_methods = "methods:" if model_type not in ["BS", "BO"] else "methods source 1:"
    default_method = "\n    default method: " + default_method
    full_str = "    " + " ".join([pre_methods, methods]) + default_method

    return full_str


def get_coords(event):
    """
    Get the coordinates of the event from the RTModel results.
    """
    coords_file = event + "/Data/event.coordinates"
    if not glob.glob(coords_file):
        raise FileNotFoundError(f"Coordinates file not found: {coords_file}")
    with open(coords_file) as f:
        ra_dec_str = f.readlines()[0].strip()

    return ra_dec_str


def get_params_and_values(par_dict, chi2=0.):
    """
    Format parameters and values for the yaml file, excluding FR and t_0_par.
    """
    parameters_lst, values_lst = list(par_dict.keys()), list(par_dict.values())
    for i, param in enumerate(parameters_lst):
        if param in ["FR", "t_0_par"]:
            parameters_lst = parameters_lst[:i] + parameters_lst[i+1:]
            values_lst = values_lst[:i] + values_lst[i+1:]

    params_str = "    parameters: " + " ".join(parameters_lst) + "\n"
    params_str += f"    # From RTModel, chi2={chi2}:\n"
    params_str += "    values: " + " ".join([str(v) for v in values_lst]) + "\n"

    return params_str


def get_starting_params_and_sigmas(par_dict):
    """
    Format starting parameters and sigmas for the yaml file.
    A sigma of 1e-7 is used for rho and q, and 1e-3 for the other parameters.
    """
    starting_params_str = "starting_parameters:\n"
    for param, value in par_dict.items():
        if param in ["FR", "t_0_par"]:
            continue
        if param.startswith("rho") or param == "q":
            starting_params_str += f"    {param}: gauss {value:.7f} 0.00001\n"
        else:
            starting_params_str += f"    {param}: gauss {value:.5f} 0.001\n"

    return starting_params_str


def get_min_max_values(par_dict):
    """
    Format min and max values for the yaml file.
    To-Do: Mention pi_E values, also rho and s...
    """
    min_values_str = "min_values:\n"
    max_values_str = "max_values:\n"
    for param, value in par_dict.items():
        if param in ["FR", "t_0_par"]:
            continue
        if param.startswith("t_0"):
            min_values_str += f"    {param}: 2455000.\n"
            max_values_str += f"    {param}: 2459000.\n"
        if param.startswith("u_0"):
            if value > 0:
                min_values_str += f"    {param}: 0.\n"
            else:
                max_values_str += f"    {param}: 0.\n"
        if param == "t_E":
            min_values_str += f"    {param}: 0.\n"
        if param == "s":
            min_values_str += f"    {param}: 0.\n"
            max_value = 2.0 if value < 1.5 else round(2*value, 1)
            max_values_str += f"    {param}: {max_value}\n"
        if param.startswith("rho"):
            min_values_str += f"    {param}: 0.\n"
            max_value = 0.1 if value < 0.08 else round(2*value, 5)
            max_values_str += f"    {param}: {max_value}\n"
        if param == "q":
            min_values_str += f"    {param}: 0.\n"
            max_values_str += f"    {param}: 1.\n"
        if param == "alpha":
            min_values_str += f"    {param}: 0.\n"
            max_values_str += f"    {param}: 360.\n"
        if param.startswith("pi_E"):
            min_values_str += f"    {param}: {value-0.5:.1f}\n"
            max_values_str += f"    {param}: {value+0.5:.1f}\n"

    return min_values_str + max_values_str


def get_plots_directory(folder_name, par_dict, model_id):
    """
    Create a directory named figs for the plots (the same as RTModel) and save
    the paths for best model, triangle, and trace plots in the yaml file.
    Improve docstrings...
    """
    figs_folder = os.path.join(folder_name, "figs")
    os.makedirs(figs_folder, exist_ok=True)
    event_name = folder_name.split("/")[-1]
    figs_folder = os.path.join("..", figs_folder)
    best_model_plot = os.path.join(figs_folder, f"{event_name}_{model_id}_model.png")

    t_0_left = par_dict['t_0'] if "t_0_2" not in par_dict else min(par_dict['t_0_1'], par_dict['t_0_2'])
    left_limit = t_0_left - min(3*par_dict['t_E'], 250)
    t_0_right = par_dict['t_0'] if "t_0_2" not in par_dict else max(par_dict['t_0_1'], par_dict['t_0_2'])
    right_limit = t_0_right + min(3*par_dict['t_E'], 250)
    t_range = f"{left_limit:.1f} {right_limit:.1f}"

    triangle_plot = os.path.join(figs_folder, f"{event_name}_{model_id}_triangle.png")
    trace_plot = os.path.join(figs_folder, f"{event_name}_{model_id}_trace.png")

    plots_str = f"\nplots:\n    best model:\n        file: {best_model_plot}\n        time range: {t_range}"
    plots_str += f"\n    triangle:\n        file: {triangle_plot}\n        shift t_0: False"
    plots_str += f"\n    trace:\n        file: {trace_plot}\n"

    return plots_str


def call_example_16(yaml_str, plot=None):
    """
    Call example_16.ulens_model_fit with the given yaml_dict to get chi2 and
    plot the best model if the -p flag is set. [...]
    Improve docstrings...
    """
    yaml_dict = yaml.safe_load(yaml_str)
    phot_files_dict = yaml_dict["photometry_files"][0]
    phot_files_dict["file_name"] = phot_files_dict["file_name"].removeprefix("../")
    keys = ["starting_parameters", "min_values", "max_values", "fitting_parameters"]
    for key in keys:
        yaml_dict[key] = None
    if "plots" in yaml_dict:
        yaml_dict["plots"].pop("triangle", None)
        yaml_dict["plots"].pop("trace", None)
    orig_dict = copy.deepcopy(yaml_dict)

    new_name = yaml_dict["plots"]["best model"]['file'].replace("_model.png", "_RTM.png")
    yaml_dict["plots"]["best model"]['file'] = new_name.removeprefix("../")
    ulens_model_fit = UlensModelFit(**yaml_dict)
    ulens_model_fit.plot_best_model()
    chi2 = ulens_model_fit._event.get_chi2()

    if plot not in ["file", "both"]:
        os.remove(new_name)
    if plot in ["screen", "both"]:
        orig_dict["plots"]["best model"].pop("file")
        ulens_model_fit = UlensModelFit(**orig_dict)
        ulens_model_fit.plot_best_model()

    return chi2


def write_short_yaml(yaml_name, par_dict):
    """
    Write a short yaml file with the parameters, as in the original code.
    """
    with open(yaml_name, 'x') as file_out:
        print(yaml_name)
        file_out.writelines("Best model:\n")
        file_out.writelines("  Parameters:\n")
        for x, y in par_dict.items():
            file_out.writelines("    {:}: {:}\n".format(x, y))


def write_extended_yaml(yaml_name, par_dict, model_id, plot):
    """
    Write an extended yaml file with the parameters and additional information.
    Improve docstrings...
    """
    yaml_str = "photometry_files:\n"
    folder_name = yaml_name.split(model_id)[0][:-1]
    yaml_str += get_data_dir_and_check(folder_name, checks=True)
    yaml_str += "fit_method: EMCEE\nmodel:\n"
    yaml_str += f"{get_model_methods(model_id[:2], par_dict)}\n"
    yaml_str += f"    coords: {get_coords(event)}\n"
    yaml_str += get_params_and_values(par_dict)

    if "t_0_par" in par_dict:
        yaml_str += "\nfixed_parameters:\n    t_0_par: {0}\n".format(par_dict["t_0_par"])

    yaml_str += "\n" + get_starting_params_and_sigmas(par_dict)
    yaml_str += get_min_max_values(par_dict)
    yaml_str += "fitting_parameters:\n    n_walkers: 100\n    n_steps: 10000\n    n_burn: 5000\n    progress: True\n"
    yaml_str += get_plots_directory(folder_name, par_dict, model_id)

    best_chi2 = call_example_16(yaml_str, plot)
    yaml_str = yaml_str.replace("    parameters: ", "    # parameters: ")
    yaml_str = yaml_str.replace("chi2=0.0", f"chi2={best_chi2:.4f}")
    yaml_str = yaml_str.replace("    values: ", "    # values: ")

    with open(yaml_name, 'w') as file_out:
        print(yaml_name)
        file_out.writelines(yaml_str)
        print("chi2_RTModel =", best_chi2)


if __name__ == '__main__':

    event, models, extended, plot = parse_args()
    MM_dicts = transform_RTM_to_MM(models)

    for (par_dict, model_id) in zip(MM_dicts, get_ids(models)):
        yaml_name = get_yaml_name(event, model_id)
        if not extended:
            write_short_yaml(yaml_name, par_dict)
        else:
            write_extended_yaml(yaml_name, par_dict, model_id, plot)
