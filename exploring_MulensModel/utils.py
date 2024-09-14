"""
UPDATE LATER...
File with general code used in other parts of MulensModel package.

Most importantly there are Utils and PlotUtils classes.
"""
import numpy as np
import scipy.optimize as op

import MulensModel as mm


class Utils(object):
    """ A number of small functions used in different places """

    def chi2_fun_ex02(theta, parameters_to_fit, event):
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
        for (parameter, value) in zip(parameters_to_fit, theta):
            setattr(event.model.parameters, parameter, value)

        return event.get_chi2()
    chi2_fun_ex02 = staticmethod(chi2_fun_ex02)

    def jacobian_ex09(theta, parameters_to_fit, event):
        """
        - Set values of microlensing parameters AND
        - Calculate chi^2 gradient (also called Jacobian).

        Note: this implementation is robust but possibly inefficient. If
        chi2_fun() is ALWAYS called before jacobian with the same parameters,
        there is no need to set the parameters in event.model; also,
        event.calculate_chi2_gradient() can be used instead (which avoids
        fitting for the fluxes twice).
        """
        for (key, value) in zip(parameters_to_fit, theta):
            setattr(event.model.parameters, key, value)
        return event.get_chi2_gradient(parameters_to_fit)
    jacobian_ex09 = staticmethod(jacobian_ex09)

    def guess_initial_t_0(data, t_bright=None):
        """
        Write later...
        """
        ### STOPPED HERE PART 2... t_bright or t_peaks ??? Change names?
        t_brightest = np.median(data.time[np.argsort(data.mag)][:9])

        if isinstance(t_bright, np.ndarray):
            if len(t_bright) > 0 and np.all(t_bright != 0):
                subt = np.abs(t_bright - t_brightest + 2450000)
                t_brightest = 2450000 + t_bright[np.argmin(subt)]
                t_bright = np.delete(t_bright, np.argmin(subt))  # Check!!!
        elif t_bright not in [None, False]:
            raise ValueError('t_bright should be a list of peak times, False'
                             ' or None.')

        return t_brightest, t_bright
    guess_initial_t_0 = staticmethod(guess_initial_t_0)

    def guess_pspl_params(data, t_bright=None):
        """
        Guess PSPL parameters: t_0 from brightest points, u_0 from the
        flux difference between baseline and brightest point and t_E from
        a quick Nelder-Mead minimization.
        """
        initial_t_0 = guess_initial_t_0(data, t_bright)  # stopped here...
        self._guess_initial_u_0()
        self._guess_initial_t_E()
        start = {'t_0': round(self.t_init, 1), 'u_0': self.u_init,
                 't_E': self.t_E_init}

        t_E_scipy = self._scipy_minimize_t_E_only(start)
        start_dict = start.copy()
        start_dict['t_E'] = round(t_E_scipy, 2)

        return start_dict
    guess_pspl_params = staticmethod(guess_pspl_params)

    def run_scipy_minimize(data, t_peaks, fix_blend):
        """
        Write later... Decide about gradient and which...
        """
        # STOPPED HERE :: 14/sep @ 10h20
        # I'm transfering this function and _subtract_model_from_data
        # from the class to here, so they are called more easily...
        # if data is None:
        #     data = self._datasets[0]
        start_dict = guess_pspl_params(t_peaks)  # paste function here!!!
        model = mm.Model(start_dict)
        ev_st = mm.Event(data, model=model, fix_blend_flux=fix_blend)

        # Nelder-Mead (no gradient)
        x0 = list(start_dict.values())
        arg = (list(start_dict.keys()), ev_st)
        bnds = [(x0[0]-50, x0[0]+50), (1e-5, 3), (1e-2, None)]
        r_ = op.minimize(Utils.chi2_fun_ex02, x0=x0, args=arg, bounds=bnds,
                         method='Nelder-Mead')
        results = [{'t_0': r_.x[0], 'u_0': r_.x[1], 't_E': r_.x[2]}]

        # Options with gradient from jacobian
        # L-BFGS-B and TNC accept `bounds``, Newton-CG doesn't
        r_ = op.minimize(Utils.chi2_fun_ex02, x0=x0, args=arg, method='L-BFGS-B',
                         bounds=bnds, jac=Utils.jacobian_ex09, tol=1e-3)
        results.append({'t_0': r_.x[0], 'u_0': r_.x[1], 't_E': r_.x[2]})

        return results[1]
    run_scipy_minimize = staticmethod(run_scipy_minimize)

    def get_model_points_between_peaks(event, t_peaks):
        """
        Get model (evaluated in the data.time) between two peaks.

        Args:
            event (mm.Event): derived binary source event
            t_peaks (list): times of the two peaks

        Returns:
            np.ndarray: model points between the two peaks
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

