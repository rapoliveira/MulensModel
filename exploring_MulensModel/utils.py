"""
File with general code used in other parts of fit_binary_source.py.
"""
import numpy as np
import scipy.optimize as op

import MulensModel as mm


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
        initial_u_0 = np.sqrt(2*magnif_A / np.sqrt(magnif_A**2 - 1) - 2)
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
                Data for which t_0 will be guessed.

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
        model to fit the data. The keywords t_peaks and t_E_prior are
        used to improve the initial guesses. The Einstein crossing time
        is optimized using a Nelder-Mead minimization.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which t_0 will be guessed.

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
                Data for which t_0 will be guessed.

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

    def subtract_model_from_data(data, model_dict, fix_blend=None):
        """
        Calculate the residuals of the data after subtracting the model,
        using the get_model_fluxes() method.

        Keywords :
            data: *MulensModel.MulensData*
                Data for which t_0 will be guessed.

            model_dict: *dict*
                Dictionary with the model parameters, including t_E.

            fix_blend: *bool* or *dict*
                If not *False*, the blending flux will be fixed to the value
                given in the dictionary during the fitting.

        Returns :
            subt_data: *MulensModel.MulensData*
                Data with the model subtracted, point by point.
        """
        model = mm.Model(model_dict)
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
                Event which has datasets for which the data will be split.

            t_peaks: *np.ndarray* or *None*
                List of peak times, as given by event_finder code.

        Returns :
            model_data_between_peaks: *np.ndarray*
                Model points between the two peaks, with three columns:
                time, flux and flux error.
        """
        t_0_left, t_0_right = sorted(t_peaks + 2.45e6)
        mm_data = event.datasets[0]
        flux_model = event.fits[0].get_model_fluxes()
        model_data = np.c_[mm_data.time, flux_model, mm_data.err_flux]
        flag_between = (t_0_left <= mm_data.time) & (mm_data.time <= t_0_right)
        model_data_between_peaks = model_data[flag_between]

        # Exclude cases where there is no data between peaks or no minimum
        if len(model_data_between_peaks) == 0:
            raise ValueError('There is no data between peaks, please check.')

        return model_data_between_peaks
    get_model_pts_between_peaks = staticmethod(get_model_pts_between_peaks)

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

        # UP TO HERE IS DONE...

        # Two functions to transfer from class to utils !!!
        # _run_scipy_minimize
        # _subtract_model_from_data

        # Detect the minimum flux in model_data_between_peaks
        idx_min_flux = np.argmin(model_data_between_peaks[:, 1])
        time_min_flux = model_data_between_peaks[:, 0][idx_min_flux]
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

    # def get_flux_from_mag(mag, zeropoint=None):
    #     """
    #     Transform magnitudes into fluxes.

    #     Parameters :
    #         mag: *np.ndarray* or *float*
    #             Values to be transformed.

    #         zeropoint: *float*
    #             Zeropoint of magnitude scale.
    #             Defaults to 22. - double check if you want to change this.

    #     Returns :
    #         flux: *np.ndarray* or *float*
    #             Calculated fluxes. Type is the same as *mag* parameter.
    #     """
    #     if zeropoint is None:
    #         zeropoint = MAG_ZEROPOINT
    #     flux = 10. ** (0.4 * (zeropoint - mag))
    #     return flux
    # get_flux_from_mag = staticmethod(get_flux_from_mag)

    # def get_flux_and_err_from_mag(mag, err_mag, zeropoint=None):
    #     """
    #     Transform magnitudes and their uncertainties into flux space.

    #     Parameters :
    #         mag: *np.ndarray* or *float*
    #             Magnitude values to be transformed.

    #         err_mag: *np.ndarray* or *float*
    #             Uncertainties of magnitudes to be transformed.

    #         zeropoint: *float*
    #             Zeropoint of magnitude scale.
    #             Defaults to 22. - double check if you want to change this.

    #     Returns :
    #         flux: *np.ndarray* or *float*
    #             Calculated fluxes. Type is the same as *mag* parameter.

    #         err_flux: *np.ndarray* or *float*
    #             Calculated flux uncertainties. Type is *float* if both *mag*
    #             and *err_mag* are *floats* and *np.ndarray* otherwise.
    #     """
    #     if zeropoint is None:
    #         zeropoint = MAG_ZEROPOINT
    #     flux = 10. ** (0.4 * (zeropoint - mag))
    #     err_flux = err_mag * flux * np.log(10.) * 0.4
    #     return (flux, err_flux)
    # get_flux_and_err_from_mag = staticmethod(get_flux_and_err_from_mag)
