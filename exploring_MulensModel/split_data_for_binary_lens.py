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

# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

import MulensModel as mm
from ulens_model_fit import UlensModelFit
from fit_1L2S_3steps_bkpi import fit_emcee, ln_prob, get_initial_t0_u0
import fit_1L2S_3steps_bkpi as fit


def split_before_result(mm_data, f_base, f_base_sigma):
    """
    WRITE LATER...

    Args:
        mm_data (_type_): _description_
        f_base (_type_): _description_
        f_base_sigma (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Testing gradient stuff to check for ...
    flag_3sigma = mm_data.flux >= (f_base + 3*f_base_sigma)
    flux_above, t_above = mm_data.flux[flag_3sigma], mm_data.time[flag_3sigma]
    min_ids = argrelextrema(flux_above, np.less_equal, order=20)[0]
    max_ids = argrelextrema(flux_above, np.greater_equal, order=20)[0]
    if len(max_ids) > 2:
        t_brightest = np.mean(mm_data.time[np.argsort(mm_data.mag)][:10])
        diff = t_above[max_ids] - t_brightest
        max_ids = np.sort([x for _, x in sorted(zip(abs(diff), max_ids))])[:2]
    t_mins, t_maxs = t_above[min_ids], t_above[max_ids]
    flag_between = (t_mins > t_maxs[0]) & (t_mins < t_maxs[1])
    time_min_flux = t_mins[flag_between]

    flag = mm_data.time <= time_min_flux
    mm_data = np.c_[mm_data.time, mm_data.mag, mm_data.err_mag]
    data_left = mm.MulensData(mm_data[flag].T, phot_fmt='mag')
    data_right = mm.MulensData(mm_data[~flag].T, phot_fmt='mag')
    if min(data_left.mag) < min(data_right.mag):
        return (data_left, data_right)
    return (data_right, data_left)


def split_after_result(event_1L2S, result, settings):
    """
    Find the minimum between t_0_1 and t_0_2 (1L2S) and split data into two.

    Args:
        event_1L2S (mm.Event): derived binary source event
        result (tuple): all the results derived in binary source fit

    Returns:
        tuple: two mm.Data instances, to the left and right of the minimum
    """

    if np.all(settings['starting_parameters']['t_peaks_list_orig'] != 0):
        t_peaks = settings['starting_parameters']['t_peaks_list_orig'] + 2.45e6
    else:
        t_peaks = [result[0]['t_0_1'], result[0]['t_0_2']]
    t_0_left, t_0_right = sorted(t_peaks)
    mm_data = event_1L2S.datasets[0]
    flux_model = event_1L2S.fits[0].get_model_fluxes()
    model_data = np.c_[mm_data.time, flux_model, mm_data.err_flux]
    between_peaks = (t_0_left <= mm_data.time) & (mm_data.time <= t_0_right)
    model_data_between_peaks = model_data[between_peaks]

    # Exclude cases where there is no data between peaks or no minimum
    if len(model_data_between_peaks) == 0:
        return [], []

    # Detect the minimum flux in model_data_between_peaks
    idx_min_flux = np.argmin(model_data_between_peaks[:, 1])
    time_min_flux = model_data_between_peaks[:, 0][idx_min_flux]
    # if time_min_flux - 0.1 < t_0_left or time_min_flux + 0.1 > t_0_right:
    #     # return [], []
    # elif model_data_between_peaks[:,1].min() in [t_0_left, t_0_right] or \
    # [...] Still to think about it... Cases where the minimum flux is
    chi2_lr, chi2_dof_lr = [], []
    if time_min_flux - 0.1 < t_0_left or time_min_flux + 0.1 > t_0_right or \
            min(idx_min_flux, len(model_data_between_peaks)-idx_min_flux) < 3:
        for item in model_data_between_peaks[::10]:
            flag = mm_data.time <= item[0]
            data_np = np.c_[mm_data.time, mm_data.mag, mm_data.err_mag]
            data_left = mm.MulensData(data_np[flag].T, phot_fmt='mag')
            data_right = mm.MulensData(data_np[~flag].T, phot_fmt='mag')
            model_1 = fit.fit_utils('scipy_minimize', data_left, settings)
            data_r_subt = fit.fit_utils('subt_data', data_right, settings, model_1)
            model_2 = fit.fit_utils('scipy_minimize', data_r_subt, settings)
            data_l_subt = fit.fit_utils('subt_data', data_left, settings, model_2)
            model_1 = fit.fit_utils('scipy_minimize', data_l_subt, settings)
            ev_1 = mm.Event(data_l_subt, model=mm.Model(model_1))
            ev_2 = mm.Event(data_r_subt, model=mm.Model(model_2))
            chi2_lr.append([ev_1.get_chi2(), ev_2.get_chi2()])
            chi2_dof_lr.append([ev_1.get_chi2() / (data_left.n_epochs-3),
                                ev_2.get_chi2() / (data_right.n_epochs-3)])
        ### Add fine solution later, getting the 10 or 100 central...
        # Temporary solution for BLG505.31.30585
        chi2_dof_l, chi2_dof_r = np.array(chi2_dof_lr).T
        idx_min = int(np.mean([np.argmin(chi2_dof_l[:-1]), np.argmin(chi2_dof_r[:-1])]))
        time_min_flux = model_data_between_peaks[idx_min*10][0]

    flag = mm_data.time <= time_min_flux
    mm_data = np.c_[mm_data.time, mm_data.mag, mm_data.err_mag]
    data_left = mm.MulensData(mm_data[flag].T, phot_fmt='mag')
    data_right = mm.MulensData(mm_data[~flag].T, phot_fmt='mag')
    if min(data_left.mag) < min(data_right.mag):
        return (data_left, data_right), time_min_flux
    return (data_right, data_left), time_min_flux
    # if min(mm_data[flag][:,0]) < min(mm_data[~flag][:,0]):
    #     return mm.MulensData(mm_data[flag].T, phot_fmt='mag')
    # return mm.MulensData(mm_data[~flag].T, phot_fmt='mag')


def fit_PSPL_twice(data_left_right, settings, result=[], start={}):
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
    n_emcee = settings['fitting_parameters']
    settings['123_fits'] = '1st fit'
    # if start == {}:
    if not isinstance(result, list):
        settings['123_fits'] += ' after 1L2S result'
        t_brightest = round(result[0]['t_0_1'], 2)
        start = get_initial_t0_u0(data_1, settings, t_brightest=t_brightest)[0]
    if not data_1.time.min() <= start['t_0'] <= data_1.time.max():
        data_1, data_2 = data_2, data_1  # BLG611.09.12112 only?
    fix_1 = None if n_emcee['fix_blend'] is False else {data_1:
                                                        n_emcee['fix_blend']}
    ev_st = mm.Event(data_1, model=mm.Model(start), fix_blend_flux=fix_1)
    n_emcee['sigmas'][0] = [0.01, 0.05, 1.0]
    output_1 = fit_emcee(start, n_emcee['sigmas'][0], ln_prob, ev_st, settings)
    model_1 = mm.Model(dict(list(output_1[0].items())[:3]))

    # Subtract data_2 from the first fit
    aux_event = mm.Event(data_2, model=model_1,
                         fix_blend_flux={data_2: n_emcee['fix_blend']})
    (flux, blend) = aux_event.get_flux_for_dataset(0)
    fsub = data_2.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
    data_2_subt = np.c_[data_2.time, fsub, data_2.err_flux][fsub > 0]
    data_2_subt = mm.MulensData(data_2_subt.T, phot_fmt='flux')

    # HERE ::: ADD STEP OF CHECKING IF THERE IS DATA > 3sigma above f_base...

    # 2nd PSPL (not to original data_2, but to data_2_subt)
    settings['123_fits'] = settings['123_fits'].replace('1st', '2nd')
    if not isinstance(result, list):
        t_brightest = round(result[0]['t_0_2'], 2)
        start, f_base = get_initial_t0_u0(data_2_subt, settings,
                                          t_brightest=t_brightest)
    else:
        start, f_base = get_initial_t0_u0(data_2_subt, settings)
    fix_2 = None if n_emcee['fix_blend'] is False else {data_2_subt:
                                                        n_emcee['fix_blend']}
    ev_st = mm.Event(data_2_subt, model=mm.Model(start), fix_blend_flux=fix_2)
    output_2 = fit_emcee(start, n_emcee['sigmas'][0], ln_prob, ev_st, settings)
    model_2 = mm.Model(dict(list(output_2[0].items())[:3]))

    # Subtract data_1 from the second fit
    aux_event = mm.Event(data_1, model=model_2,
                         fix_blend_flux={data_1: n_emcee['fix_blend']})
    (flux, blend) = aux_event.get_flux_for_dataset(0)
    fsub = data_1.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
    data_1_subt = np.c_[data_1.time, fsub, data_1.err_flux][fsub > 0]
    data_1_subt = mm.MulensData(data_1_subt.T, phot_fmt='flux')

    # 3rd PSPL (to data_1_subt)
    settings['123_fits'] = settings['123_fits'].replace('2nd', '1st')
    settings['123_fits'] += ' again'
    fix_1 = None if n_emcee['fix_blend'] is False else {data_1_subt:
                                                        n_emcee['fix_blend']}
    ev_st = mm.Event(data_1_subt, model=model_1, fix_blend_flux=fix_1)
    output_3 = fit_emcee(dict(list(output_1[0].items())[:3]),
                         n_emcee['sigmas'][0], ln_prob, ev_st, settings)

    # Quick plot to check fits
    # plt.figure(figsize=(7.5,4.8))
    # data_1_subt.plot(phot_fmt='mag', label='data_1_subt')
    # data_2_subt.plot(phot_fmt='mag', label='data_2_subt')
    # orig_data = result[4].datasets[0]
    # xlim = fit.get_xlim2(output_1[0], orig_data, n_emcee)
    # plt.scatter(orig_data.time, orig_data.mag, color="#CECECE", label='orig')
    # plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': True,
    #                'color': 'black', 't_start': xlim[0], 't_stop': xlim[1],
    #                'zorder': 10}
    # event_left = mm.Event(data_1_subt, model=model_1, fix_blend_flux=fix_1)
    # event_left.plot_model(label='model_left', **plot_params)
    # event_right = mm.Event(data_2_subt, model=model_2, fix_blend_flux=fix_2)
    # event_right.plot_model(label='model_right', **plot_params)
    # model_1L2S = mm.Model(dict(list(result[0].items())[:5]))
    # event_1L2S = mm.Event(orig_data, model=model_1L2S)
    # event_1L2S.plot_model(label='model_1L2S', ls='--', **plot_params)
    # plt.xlim(xlim)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    if not isinstance(result, list):
        return (output_1[0], output_2[0])
    return (output_3, output_2), (data_1_subt, data_2_subt)
    # return (output_1, output_2, output_3), (data_1_subt, data_2_subt)


def chi2_fun(theta, parameters_to_fit, event):
    """
    Calculate chi2 for given values of parameters

    Keywords :
        theta: *np.ndarray*
            Vector of parameter values, e.g.,
            `np.array([5380., 0.5, 20.])`.

        parameters_to_fit: *list* of *str*
            List of names of parameters corresponding to theta, e.g.,
            `['t_0', 'u_0', 't_E']`.

        event: *MulensModel.Event*
            Event which has datasets for which chi2 will be calculated.

    Returns :
        chi2: *float*
            Chi2 value for given model parameters.
    """
    # First we have to change the values of parameters in
    # event.model.parameters to values given by theta.
    for (parameter, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)

    # After that, calculating chi2 is trivial:
    return event.get_chi2()


def jacobian(theta, parameters_to_fit, event):
    """
    - Set values of microlensing parameters AND
    - Calculate chi^2 gradient (also called Jacobian).

    Note: this implementation is robust but possibly inefficient. If
    chi2_fun() is ALWAYS called before jacobian with the same parameters,
    there is no need to set the parameters in event.model; also,
    event.calculate_chi2_gradient() can be used instead (which avoids fitting
    for the fluxes twice).
    """
    for (key, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, key, value)
    return event.get_chi2_gradient(parameters_to_fit)


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
    yaml_file_2 = yaml_file_1.replace('between', 'beyond')
    pspl_1, pspl_2 = two_pspl[0].copy(), two_pspl[1].copy()
    xlim_str = [str(int(item))+'.' for item in settings['xlim']]

    # phot_settings
    phot_settings = settings['phot_settings'][0]
    phot_name, add_2450000 = list(phot_settings.values())[:-1]
    diff_path = path.replace(os.getcwd(), '.')

    # equations for trajectory between the lenses
    if (pspl_2['t_E'] / pspl_1['t_E']) ** 2 > 1.:
        pspl_1, pspl_2 = pspl_2, pspl_1
    q_2L1S = (pspl_2['t_E'] / pspl_1['t_E']) ** 2
    q_2L1S = max(q_2L1S, 1e-5)
    t_0_2L1S = (q_2L1S*pspl_2['t_0'] + pspl_1['t_0']) / (1 + q_2L1S)
    u_0_2L1S = (q_2L1S*pspl_2['u_0'] - pspl_1['u_0']) / (1 + q_2L1S)  # neg!!!
    t_E_2L1S = np.sqrt(pspl_1['t_E']**2 + pspl_2['t_E']**2)
    sum_u0, diff_t0 = pspl_1['u_0']+pspl_2['u_0'], pspl_2['t_0']-pspl_1['t_0']
    alpha_2L1S = np.degrees(np.arctan(sum_u0*t_E_2L1S / diff_t0))
    alpha_2L1S = 180. + alpha_2L1S if alpha_2L1S < 0. else alpha_2L1S
    s_prime = np.sqrt((diff_t0/t_E_2L1S)**2 + sum_u0**2)
    factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
    s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.

    # traj_between: write yaml file and plot model
    init_2L1S = [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S, alpha_2L1S]
    init_2L1S = [round(param, 5) for param in init_2L1S]
    init_2L1S[0], init_2L1S[2] = round(init_2L1S[0], 2), round(init_2L1S[2], 2)
    init_2L1S = [diff_path, phot_name, name.split('.')[0], add_2450000] + init_2L1S
    init_2L1S += xlim_str + [round(max(3, t_E_2L1S/5.), 3), 'between']
    f_template = settings['other_output']['yaml_files_2L1S']['yaml_template']
    with open(f'{path}/{f_template}', 'r', encoding='utf-8') as template_file_:
        template = template_file_.read()
    with open(f'{path}/{yaml_file_1}', 'w', encoding='utf-8') as out_file_1:
        out_file_1.write(template.format(*init_2L1S))
    with open('2L1S_plot_template.yaml', 'r', encoding='utf-8') as t_file:
        temp_plot = t_file.read()
    plot = yaml.safe_load(temp_plot.format(*init_2L1S[:-2] + init_2L1S[-1:]))
    ulens_model_fit = UlensModelFit(**plot)
    ulens_model_fit.plot_best_model()

    # equations for trajectory beyond the lenses
    u_0_2L1S = -(pspl_1['u_0'] + q_2L1S*pspl_2['u_0']) / (1 + q_2L1S)  # neg!!!
    diff_u0 = abs(pspl_1['u_0']-pspl_2['u_0'])
    alpha_2L1S = np.degrees(np.arctan(diff_u0 * t_E_2L1S / diff_t0))
    alpha_2L1S = 180. + alpha_2L1S if alpha_2L1S < 0. else alpha_2L1S
    s_prime = np.sqrt((diff_t0/t_E_2L1S)**2 + diff_u0**2)
    factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
    s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.
    init_2L1S[-1], init_2L1S[5] = 'beyond', round(u_0_2L1S, 5)
    init_2L1S[7], init_2L1S[9] = round(s_2L1S, 5), round(alpha_2L1S, 5)

    # traj_beyond: write yaml file and plot model
    with open(f'{path}/{yaml_file_2}', 'w', encoding='utf-8') as out_file_2:
        out_file_2.write(template.format(*init_2L1S))
    plot = yaml.safe_load(temp_plot.format(*init_2L1S[:-2] + init_2L1S[-1:]))
    ulens_model_fit = UlensModelFit(**plot)
    ulens_model_fit.plot_best_model()


if __name__ == '__main__':

    np.random.seed(12343)
    path = os.path.dirname(os.path.realpath(sys.argv[1]))
    with open(sys.argv[1], 'r', encoding='utf-8') as in_data:
        settings = yaml.safe_load(in_data)

    # Do it later: calling code as main will require a yaml file with 1L2S
    data_list, filenames = fit.read_data(path, settings['phot_settings'])
    print('Still working on it...')
    # for data, name in zip(data_list, filenames):
    # [...]

    # for data, name in zip(data_list, filenames):
    #     print(f'\n\033[1m * Running fit for {name}\033[0m')
    #     # breakpoint()
    #     path_base = f"{path}/{pdf_dir}/{name.split('.')[0]}"
    #     pdf_dir = settings['plots']['all_plots']['file_dir']
    #     pdf = PdfPages(f"{path_base}_result.pdf")
    #     result, cplot, xlim = make_all_fittings(data, name, settings, pdf)
    #     pdf.close()
    #     write_tables(path, settings, name, result)

    #     pdf_dir = settings['plots']['triangle']['file_dir']
    #     pdf = PdfPages(f"{path_base}_cplot.pdf")
    #     pdf.savefig(cplot)
    #     pdf.close()

    #     pdf_dir = settings['plots']['best model']['file_dir']
    #     pdf = PdfPages(f"{path_base}_fit.pdf")
    #     plot_fit(result[0], data, settings, pdf=pdf)
    #     pdf.close()
    #     print("\n--------------------------------------------------")
    # breakpoint()
