"""
File with general code used in the class FitBinarySource.
The auxiliary classes SaveResultsBinarySource and PrepareBinaryLens also
use one of the functions, namely get_mm_event().
"""
import MulensModel as mm
import numpy as np
import scipy.optimize as op


class Utils(object):
    """ A number of small functions used in different places """

    def chi2_fun_ex02(theta, parameters_to_fit, event):
        """
        Calculate chi2 for given values of parameters.

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
        for (parameter, value) in zip(parameters_to_fit, theta):
            setattr(event.model.parameters, parameter, value)
        chi2 = event.get_chi2()

        return chi2
    chi2_fun_ex02 = staticmethod(chi2_fun_ex02)

    def jacobian_ex09(theta, parameters_to_fit, event):
        """
        Calculate chi2 gradient (Jacobian) for given values of parameters.

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
            chi2_gradient: *np.ndarray*
                Chi2 gradient, i.e. how sensitive chi2 is to changes in
                each parameter of the model.
        """
        for (key, value) in zip(parameters_to_fit, theta):
            setattr(event.model.parameters, key, value)
        chi2_gradient = event.get_chi2_gradient(parameters_to_fit)

        return chi2_gradient
    jacobian_ex09 = staticmethod(jacobian_ex09)

    def guess_initial_t_0(data, t_peaks=None):
        """
        Guess initial t_0 from the median of the 9 brightest points or
        from the closest t_peaks item to it.
        If the data has multiple bumps, it is possible that the array with
        the 9 brightest points contain peaks from different bumps.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which t_0 will be guessed.

            t_peaks: *np.ndarray* or *None*
                List of peak times, as given by event_finder code.

        Returns :
            initial_t_0: *float*
                Initial guess for t_0.

            t_peaks: *np.ndarray* or *None*
                List of peak times, removing the closer to initial_t_0.
        """
        time_sorted_by_mag = data.time[np.argsort(data.mag)]
        initial_t_0 = np.median(time_sorted_by_mag[:9])

        if isinstance(t_peaks, np.ndarray):
            if len(t_peaks) > 0 and np.all(t_peaks != 0):
                subt = np.abs(t_peaks - initial_t_0 + 2450000)
                initial_t_0 = 2450000 + t_peaks[np.argmin(subt)]
                t_peaks = np.delete(t_peaks, np.argmin(subt))
        elif t_peaks not in [None, False]:
            raise ValueError('t_peaks should be a list of peak times, False'
                             ' or None.')

        return initial_t_0, t_peaks
    guess_initial_t_0 = staticmethod(guess_initial_t_0)

    def guess_initial_u_0(data, initial_t_0=None):
        """
        Guess initial u_0 from the flux difference between the brightest
        point and the baseline flux. Both fluxes are calculated from arrays
        of points to avoid outliers. If initial_t_0 is not given, it is
        obtained using the guess_initial_t_0() function.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which u_0 will be guessed.

            initial_t_0: *float* or *None*
                Initial guess for t_0.

        Returns :
            initial_u_0: *float*
                Initial guess for u_0.
        """
        if initial_t_0 is None:
            initial_t_0 = Utils.guess_initial_t_0(data)[0]

        # Starting the baseline (360d or half-data window?)
        t_diff = data.time - initial_t_0
        t_window = [initial_t_0 + min(t_diff)/2., initial_t_0 + max(t_diff)/2.]
        t_mask = (data.time < t_window[0]) | (data.time > t_window[1])
        flux_base = np.median(data.flux[t_mask])
        # flux_mag_base = (flux_base, np.std(data.flux[t_mask]),
        #                  np.median(data.mag[t_mask]),
        #                  np.std(data.mag[t_mask]))

        # Get the brightest flux around initial_t_0 (to avoid outliers)
        idx_min = np.argmin(abs(t_diff))
        min_, max_ = max(idx_min-5, 0), min(idx_min+6, len(data.mag))
        # mag_peak = min(data.mag[min_:max_]) # [idx_min-5:idx_min+6]
        flux_peak = max(data.flux[min_:max_])

        # Compute magnification and corresponding u_0(A)
        magnif_A = flux_peak / flux_base
        if magnif_A >= 1:
            initial_u_0 = np.sqrt(2*magnif_A / np.sqrt(magnif_A**2 - 1) - 2)
        else:
            initial_u_0 = 3.
        initial_u_0 = round(initial_u_0, 3)

        return initial_u_0
    guess_initial_u_0 = staticmethod(guess_initial_u_0)

    def guess_initial_t_E(t_E_prior=None):
        """
        Guess initial t_E from the prior given in the settings file. If no
        prior is given, return 25 days.

        Keywords :
            t_E_prior: *str* or *None*
                Prior for t_E, as given in the settings file. The string
                must contain the type of prior (lognormal, uniform, ...),
                the central value and width of the prior.

        Returns :
            initial_t_E: *float*
                Initial guess for t_E.
        """
        if t_E_prior is None:
            return 25.

        if isinstance(t_E_prior, str) and len(t_E_prior.split()) == 3:
            central_t_E = float(t_E_prior.split()[1])
            if t_E_prior.split()[0] == 'lognormal':
                initial_t_E = round(np.exp(central_t_E), 1)
            elif t_E_prior.split()[0] == 'uniform':
                initial_t_E = round(central_t_E, 1)
            else:
                raise ValueError('t_E_prior type not supported.')
        else:
            raise ValueError('t_E_prior should be a str with 3 elements.')

        return initial_t_E
    guess_initial_t_E = staticmethod(guess_initial_t_E)

    def scipy_minimize_t_E_only(data, model_dict):
        """
        Generates a MulensModel.Event object and minimizes chi2 for t_E only,
        using Nelder-Mead method from scipy.optimize.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which t_E will be optimized.

            model_dict: *dict*
                Dictionary with the model parameters, including t_E.

        Returns :
            solution: *np.ndarray*
                The solution of the optimization, i.e., the value of t_E.
        """
        aux_event = mm.Event(data, model=mm.Model(model_dict))
        x0, arg = [model_dict['t_E']], (['t_E'], aux_event)
        bnds = [(0.1, None)]
        res = op.minimize(Utils.chi2_fun_ex02, x0=x0, args=arg, bounds=bnds,
                          method='Nelder-Mead')
        solution = res.x[0]

        return solution
    scipy_minimize_t_E_only = staticmethod(scipy_minimize_t_E_only)

    def guess_pspl_params(data, t_peaks=None, t_E_prior=None):
        """
        Call each function that guesses the initial values for the PSPL
        model to fit the data. Keywords t_peaks and t_E_prior are used to
        improve the initial guesses. The parameter t_E is optimized using
        a separate Nelder-Mead minimization.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which the PSPL parameters will be guessed.

            t_peaks: *np.ndarray* or *None*
                List of peak times, as given by event_finder code.

            t_E_prior: *str* or *None*
                Prior for t_E, as given in the settings file. The string
                must contain the type of prior (lognormal, uniform, ...),
                the central value and width of the prior.

        Returns :
            model_dict: *dict*
                Dictionary with the model parameters, which will be used
                as starting parameters of the fitting.

            t_peaks: *np.ndarray* or *None*
                List of peak times, removing the closer to initial_t_0.
        """
        initial_t_0, t_peaks = Utils.guess_initial_t_0(data, t_peaks)
        initial_u_0 = Utils.guess_initial_u_0(data, initial_t_0)
        initial_t_E = Utils.guess_initial_t_E(t_E_prior)
        start = {'t_0': round(initial_t_0, 1), 'u_0': initial_u_0,
                 't_E': initial_t_E}

        t_E_scipy = Utils.scipy_minimize_t_E_only(data, start)
        start_dict = start.copy()
        start_dict['t_E'] = round(t_E_scipy, 2)

        return start_dict, t_peaks
    guess_pspl_params = staticmethod(guess_pspl_params)

    def run_scipy_minimize(data, t_peaks=None, t_E_prior=None, fix_blend=None):
        """
        Generates a MulensModel.Event object and makes a quick fitting
        of PSPL model (t_0, u_0 and t_E) using Nelder-Mead and L-BFGS-B
        (with gradient) methods from scipy.optimize.
        The function guess_pspl_params() is used to get the initial guesses.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which the fitting will be done.

            t_peaks: *np.ndarray* or *None*
                List of peak times, as given by event_finder code.

            t_E_prior: *str* or *None*
                Prior for t_E, as given in the settings file. The string
                must contain the type of prior (lognormal, uniform, ...),
                the central value and width of the prior.

            fix_blend: *bool* or *dict*
                If not *False*, the blending flux will be fixed to the value
                given in the dictionary during the fitting.

        Returns :
            results[1]: *dict*
                The solution of the optimization for t_0, u_0 and t_E.

            t_peaks: *np.ndarray* or *None*
                List of peak times, removing the closer to initial_t_0.
        """
        start_dict, t_peaks = Utils.guess_pspl_params(data, t_peaks, t_E_prior)
        model = mm.Model(start_dict)
        ev_st = mm.Event(data, model=model, fix_blend_flux=fix_blend)

        x0 = list(start_dict.values())
        arg = (list(start_dict.keys()), ev_st)
        bnds = [(x0[0]-50, x0[0]+50), (1e-5, 3), (1e-2, None)]
        res = op.minimize(Utils.chi2_fun_ex02, x0=x0, args=arg, bounds=bnds,
                          method='Nelder-Mead')
        results = [{'t_0': res.x[0], 'u_0': res.x[1], 't_E': res.x[2]}]

        # Options with gradient from Jacobian
        # L-BFGS-B and TNC accept `bounds``, Newton-CG doesn't
        res = op.minimize(Utils.chi2_fun_ex02, x0=x0, args=arg, bounds=bnds,
                          method='L-BFGS-B', jac=Utils.jacobian_ex09, tol=1e-3)
        results.append({'t_0': res.x[0], 'u_0': res.x[1], 't_E': res.x[2]})

        return results[1], t_peaks
    run_scipy_minimize = staticmethod(run_scipy_minimize)

    def get_mm_event(data, best):
        """
        Get an instance of mm.Event for a PSPL or binary source event,
        using the `data` and `best` parameters.

        Keywords :
            data: *MulensModel.MulensData*
                Data that will generate the event instance.

            best: *dict*
                Combination of parameters that maximize the likelihood,
                including the source and blending fluxes.

        Returns :
            event_1L2S: *MulensModel.event.Event*
                Data with the model subtracted, point by point.
        """
        bst = dict(b_ for b_ in list(best.items()) if 'flux' not in b_[0])
        fix_source = {data: [best[p] for p in best if 'flux_s' in p]}
        event_1L2S = mm.Event(data, model=mm.Model(bst),
                              fix_source_flux=fix_source,
                              fix_blend_flux={data: best['flux_b_1']})
        event_1L2S.get_chi2()

        return event_1L2S
    get_mm_event = staticmethod(get_mm_event)

    def subtract_model_from_data(data, model, fix_blend=None):
        """
        Calculate the residuals of the data after subtracting the model,
        using the get_model_fluxes() method.

        Keywords :
            data: *MulensModel.MulensData*
                Data from which the model will be subtracted.

            model: *dict* or *MulensModel.Model*
                Dictionary or Model instance with the model parameters.

            fix_blend: *dict*, *bool* or *float*
                If not *False*, the blending flux will be fixed to the value
                given in the dictionary during the fitting.

        Returns :
            subt_data: *MulensModel.MulensData*
                Data with the model subtracted, point by point.
        """
        fix_blend = Utils.check_blending_flux(fix_blend, data)
        if isinstance(model, dict):
            model = mm.Model(model)
        aux_event = mm.Event(data, model=model, fix_blend_flux=fix_blend)
        (flux, blend) = aux_event.get_flux_for_dataset(0)

        fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
        subt_data = np.c_[data.time, fsub, data.err_flux][fsub > 0]
        subt_data = mm.MulensData(subt_data.T, phot_fmt='flux')

        return subt_data
    subtract_model_from_data = staticmethod(subtract_model_from_data)

    def get_model_pts_between_peaks(event, t_peaks):
        """
        Get model points that have data.time between the two peaks. In the
        case of a binary source event, it is done to split the data in two
        parts and fit PSPL separately.

        Keywords :
            event: *MulensModel.Event*
                Fitted binary source event, which contains the datapoints
                that will be split in two.

            t_peaks: *np.ndarray*
                List of peak times, given by event_finder code or result.

        Returns :
            model_between_peaks: *np.ndarray*
                Model points between the two peaks, with three columns:
                time, flux and flux error.
        """
        t_0_left, t_0_right = sorted(t_peaks + 2.45e6)
        mm_data = event.datasets[0]
        flux_model = event.fits[0].get_model_fluxes()
        model_data = np.c_[mm_data.time, flux_model, mm_data.err_flux]
        flag_between = (t_0_left <= mm_data.time) & (mm_data.time <= t_0_right)
        model_between_peaks = model_data[flag_between]

        # Exclude cases where there is no data between peaks or no minimum
        if len(model_between_peaks) == 0:
            raise ValueError('There is no data between peaks, please check.')

        return model_between_peaks
    get_model_pts_between_peaks = staticmethod(get_model_pts_between_peaks)

    def detect_min_flux_in_model(data, t_peaks, model_between_peaks,
                                 t_E_prior=None, fix_blend=None):
        """
        Detect and optimize the minimum flux in the model datapoints
        between the two peaks.
        The for loop is an optimization to get time_min_flux for cases (e.g.
        BLG505.31.30585) where the time of the minimum flux is too close
        (less than 0.1 day) to either of the bump times, or too close to the
        maximum time of the data.

        NOTE: The result is slightly different compared to old codes because
        of the diferent models fitted in each case.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which the minimum flux will be detected.

            t_peaks: *np.ndarray*
                List of peak times, given by event_finder code or result.

            model_between_peaks: *np.ndarray*
                Model points between the two peaks, with three columns:
                time, flux and flux error.

            t_E_prior: *str* or *None*
                Prior for t_E, as given in the settings file. The string
                must contain the type of prior (lognormal, uniform, ...),
                the central value and width of the prior.

            fix_blend: *dict* or *bool*
                If not *False*, the blending flux will be fixed to the value
                given in the dictionary during the fitting.

        Returns :
            time_min_flux: *float*
                Time of the minimum flux in the model.
        """
        idx_min_flux = np.argmin(model_between_peaks[:, 1])
        time_min_flux = model_between_peaks[:, 0][idx_min_flux]
        t_left, t_right = model_between_peaks[0, 0], model_between_peaks[-1, 0]
        chi2_lr, chi2_dof_lr = [], []

        c_1 = time_min_flux - 0.1 < t_left or time_min_flux + 0.1 > t_right
        c_2 = min(idx_min_flux, len(model_between_peaks) - idx_min_flux) < 3
        if not c_1 and not c_2:
            return time_min_flux

        # Change ::10 to dividing into a fixed number of intervals
        for item in model_between_peaks[::10]:  # ::10
            flag = data.time <= item[0]
            data_np = np.c_[data.time, data.mag, data.err_mag]
            data_left = mm.MulensData(data_np[flag].T, phot_fmt='mag')
            data_right = mm.MulensData(data_np[~flag].T, phot_fmt='mag')

            fix = Utils.check_blending_flux(fix_blend, data_left)
            args_ = (t_peaks, t_E_prior, fix)
            model_1 = Utils.run_scipy_minimize(data_left, *args_)[0]
            fix = Utils.check_blending_flux(fix_blend, data_right)
            data_r_subt = Utils.subtract_model_from_data(data_right, model_1,
                                                         fix)
            args_ = (t_peaks, t_E_prior, fix)
            model_2 = Utils.run_scipy_minimize(data_r_subt, *args_)[0]
            fix = Utils.check_blending_flux(fix_blend, data_left)
            data_l_subt = Utils.subtract_model_from_data(data_left, model_2,
                                                         fix)
            args_ = (t_peaks, t_E_prior, fix)
            model_1 = Utils.run_scipy_minimize(data_l_subt, *args_)[0]

            ev_1 = mm.Event(data_l_subt, model=mm.Model(model_1))
            ev_2 = mm.Event(data_r_subt, model=mm.Model(model_2))
            chi2_lr.append([ev_1.get_chi2(), ev_2.get_chi2()])
            chi2_dof_l = ev_1.get_chi2() / (data_left.n_epochs-3)
            chi2_dof_r = ev_2.get_chi2() / (data_right.n_epochs-3)
            chi2_dof_lr.append([chi2_dof_l, chi2_dof_r])

        chi2_dof_l, chi2_dof_r = np.array(chi2_dof_lr).T
        min_args_lr = [np.argmin(chi2_dof_l[:-1]), np.argmin(chi2_dof_r[:-1])]
        idx_min = int(np.mean(min_args_lr))
        time_min_flux = model_between_peaks[idx_min*10][0]

        return time_min_flux
    detect_min_flux_in_model = staticmethod(detect_min_flux_in_model)

    def split_in_min_flux(data, time_min_flux):
        """
        Split the data in two parts, using the datapoint with the minimum
        flux between the bumps as the separation point. An array is returned
        with the two items, the first being the data with the largest flux
        (or minimum magnitude).

        Keywords :
            data: *MulensModel.MulensData*
                Data for which the blending flux will be checked.

            time_min_flux: *float*
                Time of the minimum flux in the model.

        Returns :
            data_left, data_right: *MulensModel.MulensData*
                Data instances separated in two parts, to the left and
                right of the time of minimum flux.
        """
        flag = data.time <= time_min_flux
        mm_data = np.c_[data.time, data.mag, data.err_mag]
        data_left = mm.MulensData(mm_data[flag].T, phot_fmt='mag',
                                  plot_properties={'label': "left"})
        data_right = mm.MulensData(mm_data[~flag].T, phot_fmt='mag',
                                   plot_properties={'label': "right"})

        if min(data_left.mag) < min(data_right.mag):
            return (data_left, data_right)
        return (data_right, data_left)
    split_in_min_flux = staticmethod(split_in_min_flux)

    def check_blending_flux(fix_blend, data=None):
        """
        Check if the blending flux has the right format and, in case it is
        still a float, return a dictionary with the data as key.

        Keywords :
            fix_blend: *bool*, *float* or *dict*
                If not *False*, the blending flux will be fixed to the value
                given in the dictionary during the fitting.

            data: *MulensModel.MulensData*
                Data for which the blending flux will be checked.

        Returns :
            fix_blend: *bool* or *NoneType*
                If not *False*, the blending flux will be fixed to the value
                given in the dictionary during the fitting.
        """
        if fix_blend is False:
            return None
        if data is None:
            raise ValueError('data must be given if fix_blend is not False.')

        if isinstance(fix_blend, dict):
            if data not in fix_blend:
                raise ValueError("fix_blend keys must contain the data.")
        elif isinstance(fix_blend, float):
            fix_blend = {data: fix_blend}

        return fix_blend
    check_blending_flux = staticmethod(check_blending_flux)

    def format_prior_info_for_yaml(fit_constraints):
        """
        Format the information about fit_constraints, which will be added
        to two yaml files: 1L2S results and 2L1S inputs.
        Only the sigma for negative blending flux and priors are available.

        Keywords :
            fit_constraints: *dict* or *NoneType*
                If not *None*, it contains the sigma for negative blending
                flux (float) and/or priors in the parameters.

        Returns :
            str_to_add: *str*
                String to be added to the YAML file, containing the prior
                information in the adequate format.
        """
        if fit_constraints is None:
            return

        str_to_add = "fit_constraints:\n    "
        bflux = fit_constraints.get('negative_blending_flux_sigma_mag')
        if bflux is not None:
            str_to_add += f"negative_blending_flux_sigma_mag: {bflux}\n"

        prior = fit_constraints.get('prior')
        if prior is not None:
            str_to_add += "    prior:\n"
            for key, val in prior.items():
                str_to_add += f"        {key}: {val}\n"

        return str_to_add
    format_prior_info_for_yaml = staticmethod(format_prior_info_for_yaml)
