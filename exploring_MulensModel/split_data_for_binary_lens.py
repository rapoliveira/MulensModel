"""
Auxiliary script to get initial 2L1S parameters from split 1L2S data.

The auxiliary code uses the binary source results as input, finds the minimum
of the light curve between t_0_1 and t_0_2, splits the data into t < t_0_1 and
t > t_0_2, and fits PSPL to each of them. These results generate the initial
parameters for the binary lens fitting (ulens_fit, yaml file).
"""
import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np

import MulensModel as mm
from ulens_model_fit import UlensModelFit
from fit_1L2S_3steps import read_data, fit_EMCEE, ln_prob

def find_minimum_and_split(event_1L2S, result):
    """
    Find the minimum between t_0_1 and t_0_2 (1L2S) and split data into two.

    Args:
        event_1L2S (mm.Event): derived binary source event
        result (tuple): all the results derived in binary source fit

    Returns:
        tuple: two mm.Data instances, to the left and right of the minimum
    """

    t_0_1, t_0_2 = result[0]['t_0_1'], result[0]['t_0_2']
    t_0_left, t_0_right = sorted([t_0_1, t_0_2])
    mm_data = event_1L2S.datasets[0]
    flux_model = event_1L2S.fits[0].get_model_fluxes()
    model_data = np.c_[mm_data.time, flux_model, mm_data.err_flux]
    between_peaks = (t_0_left < mm_data.time) & (mm_data.time < t_0_right)
    model_data_between_peaks = model_data[between_peaks]

    # Detect the minimum flux in model_data_between_peaks
    idx_min_flux = np.argmin(model_data_between_peaks[:,1])
    time_min_flux = model_data_between_peaks[:,0][idx_min_flux]
    flag = mm_data.time <= time_min_flux
    
    mm_data = np.c_[mm_data.time, mm_data.mag, mm_data.err_mag]
    data_left = mm.MulensData(mm_data[flag].T, phot_fmt='mag')
    data_right = mm.MulensData(mm_data[~flag].T, phot_fmt='mag')
    if min(data_left.mag) < min(data_right.mag):
        return (data_left, data_right)
    return (data_right, data_left)
    # if min(mm_data[flag][:,0]) < min(mm_data[~flag][:,0]):
    #     return mm.MulensData(mm_data[flag].T, phot_fmt='mag')
    # return mm.MulensData(mm_data[~flag].T, phot_fmt='mag')

def fit_PSPL_twice(result, data_left_right, settings):
    """
    Fit PSPL to data_left and another PSPL to the right subtracted data.

    Args:
        result (tuple): all the results derived in binary source fit
        data_left_right (tuple): two mm.Data instances (left and right)
        settings (dict): all settings from yaml file

    Returns:
        tuple: two PSPL dictionaries with the result parameters
    """

    # 1st PSPL (data_left or brighter)
    data_1, data_2 = data_left_right
    t_0_left, t_0_right = sorted([result[0]['t_0_1'], result[0]['t_0_2']])
    settings['123_fits'] = '1st fit after result'
    start = {'t_0': round(t_0_left, 2), 'u_0': 0.1, 't_E': 25}
    n_emcee = settings['fitting_parameters']
    fix_left = {data_1: n_emcee['fix_blend']}
    ev_st = mm.Event(data_1, model=mm.Model(start), fix_blend_flux=fix_left)
    n_emcee['sigmas'][0] = [0.01, 0.05, 1.0]
    output = fit_EMCEE(start, n_emcee['sigmas'][0], ln_prob, ev_st, settings)
    model = mm.Model(dict(list(output[0].items())[:3]))

    # Subtract the data_2 from first fit
    aux_event = mm.Event(data_2, model=model,
                         fix_blend_flux={data_2: n_emcee['fix_blend']})
    (flux, blend) = aux_event.get_flux_for_dataset(0)
    fsub = data_2.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
    subt_right = np.c_[data_2.time, fsub, data_2.err_flux][fsub > 0]
    subt_right = mm.MulensData(subt_right.T, phot_fmt='flux')

    # 2nd PSPL (not to original data_2, but to subt_right)
    settings['123_fits'] = '2nd fit after result'
    start = {'t_0': round(t_0_right, 2), 'u_0': 0.1, 't_E': 25}
    fix_right = {subt_right: n_emcee['fix_blend']}
    ev_st = mm.Event(subt_right, model=mm.Model(start), fix_blend_flux=fix_right)
    output_1 = fit_EMCEE(start, n_emcee['sigmas'][0], ln_prob, ev_st, settings)

    # Quick plot to check fits
    # plt.figure(figsize=(7.5,4.8))
    # subt_right.plot(phot_fmt='mag', label='right_subt')
    # orig_data = result[4].datasets[0]
    # plt.scatter(orig_data.time, orig_data.mag, color="#CECECE", label='orig')
    # plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': False,
    #                'color': 'black', 't_start': settings['xlim'][0],
    #                't_stop': settings['xlim'][1], 'zorder': 10}
    # event_left = mm.Event(data_1, model=model, fix_blend_flux=fix_left)
    # event_left.plot_model(label='model_left', **plot_params)
    # model_1 = mm.Model(dict(list(output_1[0].items())[:3]))
    # event_right = mm.Event(subt_right, model=model_1, fix_blend_flux=fix_right)
    # event_right.plot_model(label='model_right', **plot_params)
    # model_1L2S = mm.Model(dict(list(result[0].items())[:5]))
    # event_1L2S = mm.Event(orig_data, model=model_1L2S)
    # event_1L2S.plot_model(label='model_1L2S', ls='--', **plot_params)
    # plt.xlim(settings['xlim'])
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return output[0], output_1[0]

def generate_2L1S_yaml_files(path, two_pspl, name, settings):
    """
    Generate two yaml files with initial parameters for the 2L1S fitting.

    Args:
        path (str): directory of the Python script and catalogues
        two_pspl (tuple of dict): results from the two PSPL fits 
        name (str): name of the photometry file
        settings (dict): all settings from yaml file
    """

    yaml_dir = settings['other_output']['yaml_files_2L1S']['yaml_dir_name']
    yaml_dir = yaml_dir.format(name.split('.')[0])
    yaml_file_1 = yaml_dir.replace('.yaml', '_traj_between.yaml')
    yaml_file_2 = yaml_dir.replace('.yaml', '_traj_beyond.yaml')
    pspl_1, pspl_2 = two_pspl[0].copy(), two_pspl[1].copy()
    xlim_str = [str(2450000 + int(item))+'.' for item in settings['xlim']]

    # equations for trajectory between the lenses
    pspl_1['t_0'] += 2450000  # temporary line (convert code to 2450000)
    pspl_2['t_0'] += 2450000  # temporary line (convert code to 2450000)
    if (pspl_2['t_E'] / pspl_1['t_E']) ** 2 > 1.:
        pspl_1, pspl_2 = pspl_2, pspl_1
    q_2L1S = (pspl_2['t_E'] / pspl_1['t_E']) ** 2
    t_0_2L1S = (q_2L1S*pspl_2['t_0'] + pspl_1['t_0']) / (1 + q_2L1S)
    u_0_2L1S = (q_2L1S*pspl_2['u_0'] - pspl_1['u_0']) / (1 + q_2L1S) # negative!!!
    t_E_2L1S = np.sqrt(pspl_1['t_E']**2 + pspl_2['t_E']**2)
    t_a = (pspl_1['u_0']+pspl_2['u_0'])*t_E_2L1S / (pspl_2['t_0']-pspl_1['t_0'])
    alpha_2L1S = np.degrees(np.arctan(t_a))
    alpha_2L1S = 180. + alpha_2L1S if alpha_2L1S < 0. else alpha_2L1S
    s_prime = np.sqrt(((pspl_2['t_0']-pspl_1['t_0'])/t_E_2L1S)**2 +
                      (pspl_1['u_0']+pspl_2['u_0'])**2)
    factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
    s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.

    # plot best 2L1S model (between)
    with open('2L1S_plot_template.yaml', 'r') as data:
        template_plot_2L1S = data.read()
    init_2L1S = [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S, alpha_2L1S]
    plot_list = [name.split('.')[0]] + init_2L1S + ['between'] + xlim_str
    plot_2L1S = yaml.safe_load(template_plot_2L1S.format(*plot_list))
    ulens_model_fit = UlensModelFit(**plot_2L1S)
    ulens_model_fit.plot_best_model()

    # writing traj_between yaml file
    init_2L1S = [round(param, 3) for param in init_2L1S]
    init_2L1S[0], init_2L1S[2] = round(init_2L1S[0], 2), round(init_2L1S[2], 2)
    diff_path = path.replace(os.getcwd(), '.')
    init_2L1S.insert(0, diff_path)
    init_2L1S.insert(1, name.split('.')[0])
    init_2L1S += xlim_str + [round(max(3, t_E_2L1S/5.), 3), 'between']
    f_template = settings['other_output']['yaml_files_2L1S']['yaml_template']
    with open(f'{path}/{f_template}') as template_file_:
        template = template_file_.read()
    with open(f'{path}/{yaml_file_1}', 'w') as out_file_1:
        out_file_1.write(template.format(*init_2L1S))
    
    # equations for trajectory beyond the lenses
    u_0_2L1S = -(pspl_1['u_0'] + q_2L1S*pspl_2['u_0']) / (1 + q_2L1S) # negative!!!
    t_a = abs(pspl_1['u_0']-pspl_2['u_0'])*t_E_2L1S / (pspl_2['t_0']-pspl_1['t_0'])
    alpha_2L1S = np.degrees(np.arctan(t_a))
    alpha_2L1S = 180. + alpha_2L1S if alpha_2L1S < 0. else alpha_2L1S
    s_prime = np.sqrt(((pspl_2['t_0']-pspl_1['t_0'])/t_E_2L1S)**2 +
                      (pspl_1['u_0']-pspl_2['u_0'])**2)
    factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
    s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.
    init_2L1S[-1], init_2L1S[3] = 'beyond', round(u_0_2L1S, 3)
    init_2L1S[5], init_2L1S[7] = round(s_2L1S, 3), round(alpha_2L1S, 3)
    with open(f'{path}/{yaml_file_2}', 'w') as out_file_2:
        out_file_2.write(template.format(*init_2L1S))

    # plot best 2L1S model (beyond)
    init_2L1S = [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S, alpha_2L1S]
    plot_list = [name.split('.')[0]] + init_2L1S + ['beyond'] + xlim_str
    plot_2L1S = yaml.safe_load(template_plot_2L1S.format(*plot_list))
    ulens_model_fit = UlensModelFit(**plot_2L1S)
    ulens_model_fit.plot_best_model()

if __name__ == '__main__':

    np.random.seed(12343)
    path = os.path.dirname(os.path.realpath(sys.argv[1]))
    with open(sys.argv[1], 'r') as in_data:
        settings = yaml.safe_load(in_data)

    # Do it later: calling code as main will require a yaml file with 1L2S
    data_list, filenames = read_data(path, settings['phot_settings'])
    print('Still working on it...')
    # for data, name in zip(data_list, filenames):
    # [...]

    # for data, name in zip(data_list, filenames):
    #     print(f'\n\033[1m * Running fit for {name}\033[0m')
    #     # breakpoint()
    #     pdf_dir = settings['plots']['all_plots']['file_dir']
    #     pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_result.pdf")
    #     result, cplot, xlim = make_all_fittings(data, name, settings, pdf=pdf)
    #     pdf.close()
    #     write_tables(path, settings, name, result)

    #     pdf_dir = settings['plots']['triangle']['file_dir']
    #     pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_cplot.pdf")
    #     pdf.savefig(cplot)
    #     pdf.close()

    #     pdf_dir = settings['plots']['best model']['file_dir']
    #     pdf = PdfPages(f"{path}/{pdf_dir}/{name.split('.')[0]}_fit.pdf")
    #     plot_fit(result[0], data, settings['fitting_parameters'], xlim, pdf=pdf)
    #     pdf.close()
    #     print("\n--------------------------------------------------")
    # breakpoint()
