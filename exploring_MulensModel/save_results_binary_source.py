"""
Class and script for fitting binary source model using MulensModel.
It just working imported from fit_binary_source.py, so far.
"""
import os
import sys
import yaml

from astropy.table import Table, Column
import corner
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import MulensModel as mm
import numpy as np

from prepare_binary_lens import PrepareBinaryLens
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

# To-Do list (27.sep.2024):
# - [X] Finish calling prepare_binary_lens method.
# - [X] Check negative t_E for BLG501_13_138484_OGLE.
# - [X] Check why YAML file has wrong fitted fluxes.
# - [X] URGENT: Test names[1:] producing outputs with name of first event !!!
# - [X] URGENT 2: WEIRD CORNER PLOT FOR SECOND EVENT...
# - [ ] Apply priors to YAML input for 2L1S. ***
# - [X] Remove _get_xlim2 function.
# - [ ] Test _write_results_table function.
# - [ ] Speed-up the saving of the chains/models file
# - [X] ValueError: fix_blend keys must contain the data :: BLG501_28_123287


class SaveResultsBinarySource(UlensModelFit):
    """
    Class to save figures and tables with the results of the fitting to
    a binary source event.
    It is a subclass of UlensModelFit, from example_16.

    Parameters :
        [...]
    """

    def __init__(self, photometry_files, plots, **kwargs):

        self._fitting_parameters_in = kwargs.pop('fitting_parameters')
        attrs = ['additional_inputs', 'event_data', 'event_id', 'res_pspl_1',
                 'res_pspl_2', 'res_1l2s', 'time_min_flux']
        for attr in attrs:
            setattr(self, f'_{attr}', kwargs.pop(attr))
        for attr in ['all_plots', 'triangle']:
            setattr(self, f'_{attr}', plots.pop(attr))
        model_1l2s = self._get_model_yaml(self._res_1l2s[0])
        super().__init__(
            photometry_files, model=model_1l2s, plots=plots, **kwargs)

        self.path = os.path.dirname(os.path.realpath(sys.argv[1]))
        self._xlim = self._get_time_limits_for_plot(3.0, 'best model')
        self._prepare_file_names()

        if self._all_plots:
            self.create_all_plots()
        if self._triangle:
            self.create_triangle()
        if 'best model' in self._plots:
            self.create_best_model()

        self._parse_other_output_parameters()
        self.create_tables()
        self.call_prepare_binary_lens()

    def _get_model_yaml(self, model_dict):
        """
        Get model instance in yaml format, passed to UlensModelFit. The
        keys are `parameters`, `values`, `source_flux` and `blending_flux`,
        where the last two are optional.
        """
        if all(key in model_dict for key in ["parameters", "values"]):
            return model_dict

        dict_copy = model_dict.copy()
        try:
            sflux = [dict_copy.pop('flux_s_1')]
        except KeyError:
            sflux = [dict_copy.pop(key) for key in ['flux_s1_1', 'flux_s2_1']]
        bflux = dict_copy.pop('flux_b_1')
        self._model = mm.Model(dict_copy)

        model_params = ' '.join(dict_copy.keys())
        model_values = ' '.join(map(str, dict_copy.values()))
        model = {"parameters": model_params, "values": model_values,
                 "source_flux": sflux, "blending_flux": bflux}

        return model

    def _prepare_file_names(self):
        """
        Add event_id to the names of `plots` and `other_output`, just in
        the cases they contain '{}'.
        """
        names_dict = [
            (self._all_plots, 'file'),
            (self._triangle, 'file'),
            (self._plots.get('best model'), 'file'),
            (self._other_output.get('models'), 'file name'),
            (self._other_output.get('yaml output'), 'file name')
        ]

        for data, key in names_dict:
            if data and '{}' in data.get(key, ''):
                new_name = data[key].format(self._event_id)
                data[key] = os.path.join(self.path, new_name)

    def create_all_plots(self):
        """
        Create pdf for all_plots and call the functions to make tracer,
        triangle and best model plots.
        """
        self._pdf = PdfPages(self._all_plots['file'])
        data_1_subt = self._res_pspl_1.pop()
        data_2_subt = self._res_pspl_2.pop()

        self._make_pdf_plots(self._res_pspl_1, data_1_subt)
        self._make_pdf_plots(self._res_pspl_2, data_2_subt)
        self.cplot = self._make_pdf_plots(self._res_1l2s, self._event_data[0])
        self._pdf.close()

    def _make_pdf_plots(self, results_states, data):
        """
        Make three plots: tracer plot, corner plot and best model.

        Keywords :
            results_states: *tuple*
                Contains all the results, in order: an array with the best
                combination of parameters, the sampler, the states and the
                percentiles.

            data: *mm.MulensData*
                Data instance of a single event.

        Returns :
            cplot: *matplotlib.figure.Figure*
                Instante of the corner plot, so it can be reused.
        """
        best, sampler, states = results_states[:-1]
        self._fix_blend = self._additional_inputs['fix_blend']
        self._n_burn = self._fitting_parameters_in['n_burn']

        # Check: PSPL with blending_flux fixed or binary
        pspl_fix = (self._fix_blend is not False) and (len(best) != 8)
        c_states = states[:, :-2] if pspl_fix else states[:, :-1]
        params = list(best.keys())[:-1] if pspl_fix else list(best.keys())
        values = list(best.values())[:-1] if pspl_fix else list(best.values())

        self._plot_tracer(params, sampler)
        if len(best) == 8:
            c_states, params, values = c_states[:, :-3], params[:5], values[:5]
        cplot = self._plot_triangle(c_states, params, values)
        self._plot_fit(best, data)

        return cplot

    def _plot_tracer(self, fitted_params, sampler):
        """
        Plot tracer plots (or time series) of the walkers.

        Keywords :
            fitted_params: *list*
                Names of the fitted parameters.

            sampler: *emcee.ensemble.EnsembleSampler*
                Sampler that contain the chains generated in the fits.
        """
        npars = sampler.ndim
        fig, axes = plt.subplots(npars, 1, sharex=True, figsize=(10, 10))
        for i in range(npars):
            axes[i].plot(np.array(sampler.chain[:, :, i]).T, rasterized=True)
            axes[i].axvline(x=self._n_burn, ls='--', color='gray', lw=1.5)
            axes[i].set_ylabel(fitted_params[i], fontsize=16)

        axes[npars-1].set_xlabel(r'steps', fontsize=16)
        plt.tight_layout()
        self._pdf.savefig(fig)

    def _plot_triangle(self, states, labels, truths):
        """
        Plot triangle plots (or corner plots) of the simulated states.

        Keywords :
            states: *np.ndarray*
                States of the simulated chains from emcee, with shape
                (n_walkers * (n_steps - n_burn), n_dim).

            labels: *list*
                Names of the fitted parameters.

            truths: *list*
                Combination of parameters that maximize the likelihood.

        Returns :
            cplot: *matplotlib.figure.Figure*
                Instante of the corner plot, so it can be reused.
        """
        cplot = corner.corner(states, quantiles=[0.16, 0.50, 0.84],
                              labels=labels, truths=truths, show_titles=True)
        self._pdf.savefig(cplot)

        return cplot

    def _plot_fit(self, best, data):
        """
        Plot the best-fitting model(s) over the light curve in mag or flux.
        In the case of PSPL, the original data is plotted together with
        the subtracted one.

        Keywords :
            best: *np.array*
                Contain the best combination of parameters fitted to the
                data: for PSPL (3+2 params) or 1L2S (5+3 params).

            data: *mm.MulensData*
                Data instance of a single event.
        """
        fig = plt.figure(figsize=(7.5, 5.5))
        gs = GridSpec(3, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:-1, :])
        ans = self._additional_inputs.get('ans', 'max_prob')

        self._event = Utils.get_mm_event(data, best)
        if self._event.model.n_sources == 1:
            data_label = "Subtracted data"
            self._event_data[0].plot(phot_fmt='mag', color='gray', alpha=0.2,
                                     label="Original data")
            txt = f'PSPL ({ans}):'
        else:
            data_label = "Original data"
            txt = f'1L2S ({ans}):'
        self._event.plot_data(subtract_2450000=True, label=data_label)

        plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': True,
                       't_start': self._xlim[0], 't_stop': self._xlim[1],
                       'color': 'black', 'zorder': 10}
        for key, val in best.items():
            txt += f'\n{key} = {val:.2f}' if 'flux' not in key else ""
        self._event.plot_model(label=rf"{txt}", **plot_params)
        plt.tick_params(axis='both', direction='in')
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')

        ax2 = fig.add_subplot(gs[2:, :], sharex=ax1)
        self._event.plot_residuals(subtract_2450000=True, zorder=10)
        plt.tick_params(axis='both', direction='in')
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        plt.xlim(*np.array(self._xlim) - 2450000)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.axes(ax1)
        ax1.legend(loc='best')
        self._pdf.savefig(fig)
        plt.close('all')

    def create_triangle(self):
        """
        Create pdf for triangle plot and call the function to make it.
        """
        self._pdf = PdfPages(self._triangle['file'])

        if hasattr(self, 'cplot'):
            self._pdf.savefig(self.cplot)
        else:
            raise NotImplementedError("still working on it...")
            # self._plot_triangle(self, states, labels, truths)
        self._pdf.close()

    def create_best_model(self):
        """
        Create pdf for best model plot and call the function to make it.
        """
        self._pdf = PdfPages(self._plots['best model']['file'])
        self._plot_fit(self._res_1l2s[0], self._event_data[0])
        self._pdf.close()

    def create_tables(self):
        """
        Central function to create all tables (table with all simulated
        chains, yaml file with results and table with events).
        """
        if 'models' in self._other_output:
            thetas = self._res_1l2s[2][:, :5]
            ln_probs = self._res_1l2s[2][:, -1]
            for (theta, ln_prob) in zip(thetas, ln_probs):
                theta_str = " ".join([repr(x) for x in theta])
                out = "{:.4f}  {:}".format(ln_prob, theta_str)
                print(out, file=self._print_model_file, flush=False)
            self._print_model_file.close()

        if 'yaml output' in self._other_output:
            lst, aux_dict = self._organizing_yaml_content()
            self._write_yaml_output(lst, aux_dict)

        if 'table output' in self._additional_inputs:
            self._write_results_table()

    def _organizing_yaml_content(self):
        """
        Organize the results into a list and a dictionary to be saved in a
        in a yaml file (as in example_16).
        """
        best_items = list(self._res_1l2s[0].items())
        sampler = self._res_1l2s[1]
        perc_items = list(self._res_1l2s[3].items())
        bst = dict(item for item in best_items if 'flux' not in item[0])
        fluxes = dict(item for item in best_items if 'flux' in item[0])
        perc = dict(item for item in perc_items if 'flux' not in item[0])
        perc_fluxes = dict(item for item in perc_items if 'flux' in item[0])

        acc_fraction = np.mean(sampler.acceptance_fraction)
        acor = sampler.get_autocorr_time(quiet=True, discard=self._n_burn)
        deg_of_freedom = self._event_data[0].n_epochs - len(bst)
        pspl_1, pspl_2 = self._res_pspl_1[0], self._res_pspl_2[0]
        pspl_1 = str([round(val, 7) for val in pspl_1.values()])
        pspl_2 = str([round(val, 7) for val in pspl_2.values()])
        xlim = str([round(val, 2) for val in self._xlim])

        lst = ['', '', pspl_1, pspl_2, xlim, acc_fraction, np.mean(acor),
               '', '', self._event.chi2, deg_of_freedom, '', '']
        dict_perc_best = {7: perc, 8: perc_fluxes, 11: bst, 12: fluxes}

        return lst, dict_perc_best

    def _write_yaml_output(self, lst, dict_perc_best):
        """
        Fill the template with the results and write the yaml file.
        """
        lst[0], lst[1] = sys.argv[1], self._event_id
        for idx, dict_obj in dict_perc_best.items():
            for key, val in dict_obj.items():
                if idx in [7, 8]:
                    uncerts = f'+{val[2]-val[1]:.5f}, -{val[1]-val[0]:.5f}'
                    lst[idx] += f'    {key}: [{val[1]:.5f}, {uncerts}]\n'
                else:
                    lst[idx] += f'    {key}: {val}\n'
            lst[idx] = lst[idx][:-1]

        template = os.path.join(self.path, '../1L2S-result_template.yaml')
        with open(template) as file_:
            template_result = file_.read()
        self._template_result = template_result.format(*lst)
        yaml_fname = self._other_output['yaml output']['file name']
        with open(yaml_fname, 'w') as yaml_results:
            yaml_results.write(self._template_result)

    def _write_results_table(self):
        """
        Write the results to a table, saving the results of the best model
        for each event. The table is read, the data of the event is updated
        and then written again.

        NOTE: Function needs to be tested again...
        """
        input_dict = self._additional_inputs['table output']
        fname, columns, dtypes = input_dict.values()
        fname = os.path.join(self.path, fname)

        if not os.path.isfile(fname):
            res_tab = Table()
            for col, dtype in zip(columns, dtypes):
                res_tab[col] = Column(name=col, dtype=dtype)
        else:
            res_tab = Table.read(fname, format='ascii')

        name = self._event_id
        best_items = list(self._res_1l2s[0].items())
        bst = dict(item for item in best_items if 'flux' not in item[0])
        bst_values = [round(val, 5) for val in bst.values()]
        lst = bst_values+[0., 0.] if len(bst) == 3 else bst_values
        deg_of_freedom = self._event_data[0].n_epochs - len(bst)
        lst = [name] + lst + [round(self._event.chi2, 4), deg_of_freedom]
        if name in res_tab['id']:
            idx_event = np.where(res_tab['id'] == name)[0]
            if res_tab[idx_event]['chi2'] > lst[-2]:
                res_tab[idx_event] = lst
        else:
            res_tab.add_row(lst)
        res_tab.sort('id')
        res_tab.write(fname, format="ascii.commented_header", overwrite=True)

    def call_prepare_binary_lens(self):
        """
        Call class that prepares the binary lens model, using the results
        from the fitting.
        """
        if self._additional_inputs['yaml_files_2L1S']['t_or_f'] is False:
            return
        if not hasattr(self, '_template_result'):
            raise NotImplementedError('Still working on it...')

        stg = yaml.safe_load(self._template_result)
        stg.pop("Mean acceptance fraction")
        stg.pop("Mean autocorrelation time [steps]")
        new_stg = {k.lower().replace(' ', '_'): v for k, v in stg.items()}
        PrepareBinaryLens(**new_stg)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')
    with open(sys.argv[1], 'r', encoding="utf-8") as yaml_input:
        all_settings = yaml.safe_load(yaml_input)

    raise NotImplementedError("This code cannot be called as main code "
                              "because it is unlikely that someone will "
                              "input everything necessary to save.")
