import os
import sys
import yaml

import numpy as np

# import MulensModel as mm
from ulens_model_fit import UlensModelFit


class PrepareBinaryLens(object):
    """
    Auxiliary CLASS to get initial 2L1S parameters from split 1L2S data.

    Remove it? Improve doc...
    The auxiliary code uses the binary source results as input, finds the minimum
    of the light curve between t_0_1 and t_0_2, splits the data into t < t_0_1 and
    t > t_0_2, and fits PSPL to each of them. These results generate the initial
    parameters for the binary lens fitting (ulens_fit, yaml file).

    Args:
        object (_type_): _description_
    """

    def __init__(self, path_orig_settings=None, pspl_1=None, pspl_2=None,
                 xlim=None, fitted_parameters=None, fitted_fluxes=None,
                 best_model=None):

        self.path_orig_settings = path_orig_settings
        self.get_paths_and_orig_settings()
        self.event_id = os.path.basename(sys.argv[1]).split('_results')[0]

        self.pspl_1 = pspl_1
        self.pspl_2 = pspl_2
        self.xlim_str = [str(int(item))+'.' for item in xlim]
        self.fitted_parameters = fitted_parameters
        self.fitted_fluxes = fitted_fluxes
        self.best_chi2 = best_model['chi2']
        self.best_dof = best_model['dof']
        self.best_params = best_model['Parameters']
        self.best_fluxes = best_model['Fluxes']

        self.check_input_types()
        self.get_filenames_and_templates()
        params_between = self.get_initial_params_traj_between()
        params_beyond = self.get_initial_params_traj_beyond(params_between)
        self.round_params_and_save(params_between, 'between')
        self.round_params_and_save(params_beyond, 'beyond')

    def check_input_types(self):
        """
        Check the input types...
        """
        strings = [self.path_orig_settings]
        lists = [self.pspl_1, self.pspl_2, self.xlim_str]
        floats = [self.best_chi2]
        ints = [self.best_dof]
        dicts = [self.fitted_parameters, self.fitted_fluxes, self.best_params,
                 self.best_fluxes]

        is_str = [isinstance(val, str) for val in strings]
        is_list = [isinstance(val, list) for val in lists]
        is_float = [isinstance(val, float) for val in floats]
        is_int = [isinstance(val, int) for val in ints]
        is_dict = [isinstance(val, dict) for val in dicts]

        if not all(is_str + is_list + is_float + is_int + is_dict):
            raise TypeError('Input types are not correct in YAML file.')

    def get_paths_and_orig_settings(self):
        """
        Get the paths and original settings from the yaml file.
        """
        self.path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.dirname(os.path.realpath(sys.argv[1]))
        diff_path = yaml_path.replace(os.getcwd(), '.')
        self.base_path = os.path.join('.', diff_path.split('/')[1])

        fname = os.path.join(self.path, self.path_orig_settings)
        with open(fname, 'r', encoding='utf-8') as data:
            self.settings = yaml.safe_load(data)
        self.phot_name = self.settings['phot_settings'][0]['name']
        self.add_2450000 = self.settings['phot_settings'][0]['add_2450000']

    def get_filenames_and_templates(self):
        """
        Get the filenames for the between/beyond yaml files.
        """
        yaml_2L1S = self.settings['other_output']['yaml_files_2L1S']
        yaml_dir = yaml_2L1S['yaml_dir_name']
        yaml_dir = yaml_dir.format(self.event_id)
        yaml_dir = yaml_dir.replace('.yaml', '_traj_between.yaml')

        self.yaml_file = os.path.join(self.base_path, yaml_dir)
        template = os.path.join(self.base_path, yaml_2L1S['yaml_template'])
        plot_template = os.path.join(self.path, '2L1S_plot_template.yaml')

        with open(template, 'r', encoding='utf-8') as data:
            self.template = data.read()
        with open(plot_template, 'r', encoding='utf-8') as data:
            self.template_plot = data.read()

    def get_initial_params_traj_between(self):
        """
        Get the initial parameters for the trajectory between the lenses.
        PSPL lists contain: t_0, u_0, t_E, source_flux, blending_flux.
        """
        # invert PSPL values if t_E_2 > t_E_1, to avoid q > 1
        if (self.pspl_2[2] / self.pspl_1[2]) ** 2 > 1.:
            self.pspl_1, self.pspl_2 = self.pspl_2, self.pspl_1

        # Calculate t_0, u_0, t_E and q for the 2L1S model
        q_2L1S = (self.pspl_2[2] / self.pspl_1[2]) ** 2
        t_0_2L1S = (q_2L1S * self.pspl_2[0] + self.pspl_1[0]) / (1 + q_2L1S)
        u_0_2L1S = (q_2L1S * self.pspl_2[1] - self.pspl_1[1]) / (1 + q_2L1S)
        t_E_2L1S = np.sqrt(self.pspl_1[2]**2 + self.pspl_2[2]**2)

        # Calculate alpha with some auxiliary variables
        sum_u0 = self.pspl_1[1] + self.pspl_2[1]
        diff_t0 = self.pspl_2[0] - self.pspl_1[0]
        alpha_2L1S = np.degrees(np.arctan(sum_u0 * t_E_2L1S / diff_t0))
        alpha_2L1S = 180. + alpha_2L1S if alpha_2L1S < 0. else alpha_2L1S

        # Calculate s for the 2L1S model
        s_prime = np.sqrt((diff_t0 / t_E_2L1S)**2 + sum_u0**2)
        factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
        s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.

        return [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S, alpha_2L1S]

    def get_initial_params_traj_beyond(self, params_between):
        """
        Change u_0, s and alpha of the 2L1S model between the lenses, in
        order to get the trajectory beyond the lenses.
        """
        # Get t_E and q, which remain unchanged
        t_0_2L1S, t_E_2L1S, q_2L1S = params_between[::2]

        # Calculate u_0 and alpha for the 2L1S model
        u_0_2L1S = -(self.pspl_1[1] + q_2L1S * self.pspl_2[1]) / (1 + q_2L1S)
        diff_u0 = abs(self.pspl_1[1] - self.pspl_2[1])
        diff_t0 = self.pspl_2[0] - self.pspl_1[0]
        alpha_2L1S = np.degrees(np.arctan(diff_u0 * t_E_2L1S / diff_t0))
        alpha_2L1S = 180. + alpha_2L1S if alpha_2L1S < 0. else alpha_2L1S

        # Calculate s for the 2L1S model
        s_prime = np.sqrt((diff_t0/t_E_2L1S)**2 + diff_u0**2)
        factor = 1 if s_prime + np.sqrt(s_prime**2 + 4) > 0. else -1
        s_2L1S = (s_prime + factor*np.sqrt(s_prime**2 + 4)) / 2.

        return [t_0_2L1S, u_0_2L1S, t_E_2L1S, s_2L1S, q_2L1S, alpha_2L1S]

    def round_params_and_save(self, params, between_or_beyond):
        """
        Improve... Write the two 2L1S yaml files using the template.
        """
        round_dec = (2, 5, 2, 5, 5, 5)
        lst = [round(param, round_dec[i]) for i, param in enumerate(params)]
        phot_params = [self.base_path, self.phot_name, self.event_id,
                       self.add_2450000]
        max_t_E = round(max(3, lst[2]/5.), 3)
        lst = phot_params + lst + self.xlim_str + [max_t_E, between_or_beyond]

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
