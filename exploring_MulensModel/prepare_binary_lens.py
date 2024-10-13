import os
import sys
import yaml

import MulensModel as mm
import numpy as np

try:
    ex16_path = os.path.join(mm.__path__[0], '../../examples/example_16')
    sys.path.append(os.path.abspath(ex16_path))
    from ulens_model_fit import UlensModelFit
except ImportError as err:
    print(err)
    print("Please install MulensModel in editable mode (-e) from within the"
          "directory cloned from GitHub. This will allow to import the class"
          "UlensModelFit from example_16.")
    sys.exit(1)
from utils import Utils


class PrepareBinaryLens(object):
    """
    Class to calculate the initial parameters and prepare the YAML files to
    be used as input in binary lens (2L1S) fitting. Two files are produced,
    one for the trajectory between the lenses and another for the trajectory
    beyond the lenses.

    It uses as attributes the results of the binary source (1L2S) fitting
    and the two PSPL models.

    It can be imported with the required attributes or called as standalone
    code with a YAML file (1L2S result) via the command line.

    Usage:
        python3 prepare_binary_lens.py <path>/<id>_results_1L2S.yaml

    Attributes :
        path_orig_settings: *str*
            path to the original settings YAML file, which contains the
            input information for the 1L2S fitting
        pspl_1, pspl_2: *list*
            lists of the PSPL models fitted separately, which will be used
            to get the initial parameters for the 2L1S model
        xlim: *list*
            list of the lower and upper limits for the x-axis in the plot,
            used here to set the limits for point_source model
        fitted_parameters: *dict*
            median and 1-sigma errors for the fitted parameters (t_0_1,
            u_0_1, t_0_2, u_0_2, t_E), from the 1L2S fitting
        fitted_fluxes: *dict*
            median and 1-sigma errors for the fitted fluxes (source_flux_1,
            source_flux_2, blending_flux)
        best_model: *dict*
            dictionary with information about the model with minimum chi2,
            should contain chi2, dof, Parameters and Fluxes
        fit_constraints: *dict*, optional
            dictionary with the fit constraints, allowing only the sigma
            for negative blending flux and t_E prior
    """

    def __init__(self, path_orig_settings, event_id, pspl_1, pspl_2, xlim,
                 fitted_parameters, fitted_fluxes, best_model,
                 fit_constraints=None):

        self.path_orig_settings = path_orig_settings
        self.get_paths_and_orig_settings()

        self.event_id = event_id
        self.pspl_1 = pspl_1
        self.pspl_2 = pspl_2
        self.xlim_str = [str(int(item))+'.' for item in xlim]
        self.fitted_parameters = fitted_parameters
        self.fitted_fluxes = fitted_fluxes
        self.best_chi2 = best_model['chi2']
        self.best_params = best_model['Parameters']
        self.best_fluxes = best_model['Fluxes']
        self.fit_constraints = fit_constraints

        self.check_input_types()
        self.check_binary_source_chi2()
        self.get_filenames_and_templates()
        self.get_initial_params_traj_between()
        self.round_params_and_save(self._temp_params, 'between')
        self.get_initial_params_traj_beyond()
        self.round_params_and_save(self._temp_params, 'beyond')

    def get_paths_and_orig_settings(self):
        """
        Get the paths and original settings from the YAML file.
        """
        self.path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.dirname(os.path.realpath(sys.argv[1]))
        diff_path = yaml_path.replace(os.getcwd(), '.')
        self.base_path = os.path.join('.', diff_path.split('/')[1])

        fname = os.path.join(self.path, self.path_orig_settings)
        with open(fname, 'r', encoding='utf-8') as data:
            self.settings = yaml.safe_load(data)
        self.phot_settings = self.settings['photometry_files'][0]
        self.phot_dir = self.phot_settings['file_name']
        test_name = os.path.join(self.base_path, self.phot_dir)
        if not os.path.isdir(test_name):
            self.phot_dir = os.path.dirname(self.phot_dir)
        if self.phot_dir.endswith('/'):
            self.phot_dir = self.phot_dir[:-1]
        self.add_2450000 = self.phot_settings['add_2450000']

    def check_input_types(self):
        """
        Check the input types: strings, lists, floats and dicts.
        """
        strings = [self.path_orig_settings]
        lists = [self.pspl_1, self.pspl_2, self.xlim_str]
        floats = [self.best_chi2]
        dicts = [self.fitted_parameters, self.fitted_fluxes, self.best_params,
                 self.best_fluxes]

        is_str = [isinstance(val, str) for val in strings]
        is_list = [isinstance(val, list) for val in lists]
        is_float = [isinstance(val, float) for val in floats]
        is_dict = [isinstance(val, dict) for val in dicts]

        if not all(is_str + is_list + is_float + is_dict):
            raise TypeError('At least one type is not correct in YAML file.')

    def check_binary_source_chi2(self):
        """
        Check if the chi2 of the binary source model is consistent.
        """
        filename = os.path.join(self.base_path, self.phot_dir, self.event_id)
        self.phot_settings['file_name'] = filename + '.dat'
        data = mm.MulensData(**self.phot_settings)
        best = dict(self.best_params, **self.best_fluxes)
        event_1l2s = Utils.get_mm_event(data, best)

        if abs(event_1l2s.chi2 - self.best_chi2) > 2e-4:
            raise ValueError('Chi2 of the best model is not consistent.')

    def get_filenames_and_templates(self):
        """
        Get the filenames and templates for the two YAML files: trajectories
        between and beyond the lenses. A template to obtain the plots (using
        UlensModelFit) is also recorded.
        """
        yaml_2L1S = self.settings['additional_inputs']['yaml_files_2L1S']
        yaml_dir = yaml_2L1S['yaml_dir_name']
        yaml_dir = yaml_dir.format(self.event_id)
        yaml_dir = yaml_dir.replace('.yaml', '_traj_between.yaml')
        self.yaml_file = os.path.join(self.base_path, yaml_dir)

        template = os.path.join(self.base_path, yaml_2L1S['yaml_template'])
        with open(template, 'r', encoding='utf-8') as data:
            self.template = data.read()
        str_to_add = Utils.format_prior_info_for_yaml(self.fit_constraints)
        str_to_add += "min_values:"
        self.template = self.template.replace('min_values:', str_to_add)

        plot_template = os.path.join(self.path, '2L1S_plot_template.yaml')
        with open(plot_template, 'r', encoding='utf-8') as data:
            self.template_plot = data.read()

    def get_initial_params_traj_between(self):
        """
        Get the initial parameters for the trajectory between the lenses.
        PSPL lists contain: t_0, u_0, t_E, source_flux, blending_flux.
        """
        # Invert PSPL values if t_E_2 > t_E_1, to avoid q > 1
        if (self.pspl_2[2] / self.pspl_1[2]) ** 2 > 1.:
            self.pspl_1, self.pspl_2 = self.pspl_2, self.pspl_1

        # Calculate t_0, u_0, t_E and q for the 2L1S model
        q_2L1S = (self.pspl_2[2] / self.pspl_1[2]) ** 2
        q_2L1S = max(q_2L1S, 1e-5)
        t_0_2L1S = (q_2L1S * self.pspl_2[0] + self.pspl_1[0]) / (1 + q_2L1S)
        u_0_2L1S = (q_2L1S * self.pspl_2[1] - self.pspl_1[1]) / (1 + q_2L1S)
        t_E_2L1S = np.sqrt(self.pspl_1[2]**2 + self.pspl_2[2]**2)

        # Calculate s with some auxiliary variables
        sum_u0 = self.pspl_1[1] + self.pspl_2[1]
        diff_t0 = self.pspl_2[0] - self.pspl_1[0]
        s_prime = np.sqrt((diff_t0 / t_E_2L1S)**2 + sum_u0**2)
        factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
        s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.
        self._temp_params = [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S]

        # Calculate alpha
        alpha_2L1S = np.degrees(np.arctan(sum_u0 * t_E_2L1S / diff_t0))
        self._check_u_0_and_alpha(alpha_2L1S)

    def _check_u_0_and_alpha(self, alpha):
        """
        If u_0 < 0, the binary lens degeneracy from Skowron et al. (2011)
        is applied inverting the sign of u_0 and alpha.add()
        If alpha is out of the range [0, 360), it is changed to the correct
        value using the operator %. The solutions will probably be in the
        ranges [0, 90) or [270, 360).

        UPDATE DOCSTRINGS...
        """
        if self._temp_params[1] < 0.:
            self._temp_params[1] *= -1.
            alpha_1 = (360. - alpha) % 360.
        else:
            alpha_1 = alpha % 360.
        alpha_2 = (180. + alpha_1) % 360.

        # testing alpha_1 (add it to the docstrings...)
        data = mm.MulensData(**self.phot_settings)
        m_dict = dict(zip(['t_0', 'u_0', 't_E', 's', 'q'], self._temp_params))
        m_dict['alpha'] = alpha_1
        event_1 = Utils.get_mm_event(data, m_dict)

        # testing alpha_2 and selecting the lowest chi2
        m_dict['alpha'] = alpha_2
        event_2 = Utils.get_mm_event(data, m_dict)
        alpha = alpha_1 if event_1.chi2 < event_2.chi2 else alpha_2
        self._temp_params.append(alpha)

    def get_initial_params_traj_beyond(self):
        """
        Change u_0, s and alpha of the 2L1S model between the lenses, in
        order to get the trajectory beyond the lenses.
        """
        # Get t_0, t_E and q, which remain unchanged
        t_0_2L1S, t_E_2L1S, q_2L1S = self._temp_params[::2]

        # Calculate u_0 and alpha for the 2L1S model
        u_0_2L1S = -(self.pspl_1[1] + q_2L1S * self.pspl_2[1]) / (1 + q_2L1S)
        diff_u0 = abs(self.pspl_1[1] - self.pspl_2[1])
        diff_t0 = self.pspl_2[0] - self.pspl_1[0]

        # Calculate s for the 2L1S model
        s_prime = np.sqrt((diff_t0/t_E_2L1S)**2 + diff_u0**2)
        factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
        s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.
        self._temp_params = [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S]

        # Calculate alpha
        alpha_2L1S = np.degrees(np.arctan(diff_u0 * t_E_2L1S / diff_t0))
        self._check_u_0_and_alpha(alpha_2L1S)

    def round_params_and_save(self, params, between_or_beyond):
        """
        Round the parameters to five decimal places (except for t_0 and t_E,
        which are rounded to two decimal places) and save them in the YAML
        file. The best model and trajectory plots are also saved using the
        UlensModelFit class.
        """
        round_dec = (2, 5, 2, 5, 5, 5)
        lst = [round(param, round_dec[i]) for i, param in enumerate(params)]
        phot_params = [self.base_path, self.phot_dir, self.event_id,
                       self.add_2450000]
        min_t_E = round(max(3, lst[2]/5.), 3) if lst[2] > 3 else 1.
        lst = phot_params + lst + self.xlim_str + [min_t_E, between_or_beyond]

        if between_or_beyond == 'beyond':
            self.yaml_file = self.yaml_file.replace('between', 'beyond')

        with open(self.yaml_file, 'w', encoding='utf-8') as out_file:
            out_file.write(self.template.format(*lst))
        lst_plot = lst[:-2] + [between_or_beyond]
        plot = yaml.safe_load(self.template_plot.format(*lst_plot))
        ulens_model_fit = UlensModelFit(**plot)
        ulens_model_fit.plot_best_model()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')

    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as data:
        stg = yaml.safe_load(data)

    stg.pop("Mean acceptance fraction")
    stg.pop("Mean autocorrelation time [steps]")
    new_stg = {key.lower().replace(' ', '_'): val for key, val in stg.items()}
    prepare_binary_lens = PrepareBinaryLens(**new_stg)
