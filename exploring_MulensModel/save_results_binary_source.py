"""
Plot several fits (PSPL, parallax, 1L2S, 2L1S) to data

The code produces a plot figure containing several fits (standard is three)
to the data in magnitudes, including a panel with the residuals and an optional
one with the cumulative delta_chi2.
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


class SaveResultsBinarySource(UlensModelFit):
    """
    Update later...

    Class for plotting multiple models to data, including the relative
    residuals and optional cumulative delta_chi2.

    Args:
        UlensModelFit (class): parent class from example16
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
        # self._get_xlim2(ref=self._time_min_flux)
        self._xlim = self._get_time_limits_for_plot(3.0, 'best model')

        if self._all_plots:
            self.create_all_plots()
        if self._triangle:
            self.create_triangle()
        if 'best model' in self._plots:
            self.create_best_model()

        # only tables missing now...
        breakpoint()

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

    def _get_xlim2(self, ref=None):
        """
        Get the optimal range for the x-axis, considering the event results.
        # Radek: using get_data_magnification from MulensModel
        Returns a list with range for the x-axis, without subtracting 2450000.
        Still shorten it...

        Args:
            ref (float, optional): reference for t_0. Defaults to None.
        """
        best = self._res_pspl_1[0]
        data = self._event_data[0]
        self._fix_blend = self._additional_inputs['fix_blend']

        bst = dict(itm for itm in list(best.items()) if 'flux' not in itm[0])
        fix = None if self._fix_blend is False else {data: self._fix_blend}
        event = mm.Event(data, model=mm.Model(bst), fix_blend_flux=fix)
        event.get_flux_for_dataset(0)
        Amax = max(event.fits[0].get_data_magnification())
        dividend = best['flux_s_1']*Amax + best['flux_b_1']
        divisor = best['flux_s_1'] + best['flux_b_1']
        deltaI = 2.5*np.log10(dividend/divisor)  # deltaI ~ 3 for PAR-46 :: OK!

        # Get the magnitude at the model peak (mag_peak ~ comp? ok)
        idx_peak = np.argmin(abs(data.time - best['t_0']))
        model_mag = event.fits[0].get_model_magnitudes()
        mag_peak = model_mag[idx_peak]  # comp = data.mag[idx_peak]

        # Summing 0.85*deltaI to the mag_peak, then obtain t_range (+3%)
        mag_baseline = mag_peak + 0.85*deltaI
        idx1 = np.argmin(abs(mag_baseline - model_mag[:idx_peak]))
        idx2 = idx_peak + np.argmin(abs(mag_baseline - model_mag[idx_peak:]))
        t_range = [0.97*(data.time[idx1]-2450000) + 2450000,
                   1.03*(data.time[idx2]-2450000) + 2450000]
        t_cen = best['t_0'] if ref is None else ref
        max_diff_t_0 = max(abs(np.array(t_range) - t_cen)) + 100

        if max_diff_t_0 > 250:
            self._xlim = [t_cen-max_diff_t_0, t_cen+max_diff_t_0]
        self._xlim = [t_cen-500, t_cen+500]

    def create_all_plots(self):
        """
        Write...
        """
        pdf_dir = os.path.join(self.path, self._all_plots['file_dir'])
        self._pdf = PdfPages(pdf_dir.format(self._event_id))
        data_1_subt = self._res_pspl_1.pop()
        data_2_subt = self._res_pspl_2.pop()

        self._make_pdf_plots(self._res_pspl_1, data_1_subt)
        self._make_pdf_plots(self._res_pspl_2, data_2_subt)
        self.cplot = self._make_pdf_plots(self._res_1l2s, self._event_data[0])
        self._pdf.close()

    def _make_pdf_plots(self, results_states, data):
        """
        Make three plots: tracer plot, corner plot and best model.

        Args:
            results_states (tuple): contains best results, sampler and states.
            data (mm.MulensData): data instance of a single event.

        Returns:
            tuple: mm.Event and corner plot instances, to be used later.
        """
        best, sampler, states = results_states
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
        """
        cplot = corner.corner(states, quantiles=[0.16, 0.50, 0.84],
                              labels=labels, truths=truths, show_titles=True)
        self._pdf.savefig(cplot)

        return cplot

    def _plot_fit(self, best, data):
        """
        Plot the best-fitting model(s) over the light curve in mag or flux.
        Update docstrings...

        Args:
            best (dict): results from PSPL (3+2 params) or 1L2S (5+3 params).
            data (mm.MulensData instance): object containing all the data.

        Returns:
            mm.Event: final event containing the model and datasets.
        """
        fig = plt.figure(figsize=(7.5, 5.5))
        gs = GridSpec(3, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:-1, :])
        ans = self._additional_inputs.get('ans', 'max_prob')

        event = Utils.get_mm_event(data, best)
        if event.model.n_sources == 1:
            data_label = "Subtracted data"
            self._event_data[0].plot(phot_fmt='mag', color='gray', alpha=0.2,
                                     label="Original data")
            txt = f'PSPL ({ans}):'
        else:
            data_label = "Original data"
            txt = f'1L2S ({ans}):'
        event.plot_data(subtract_2450000=True, label=data_label)

        plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': True,
                       't_start': self._xlim[0], 't_stop': self._xlim[1],
                       'color': 'black', 'zorder': 10}
        for key, val in best.items():
            txt += f'\n{key} = {val:.2f}' if 'flux' not in key else ""
        event.plot_model(label=rf"{txt}", **plot_params)
        plt.tick_params(axis='both', direction='in')
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')

        ax2 = fig.add_subplot(gs[2:, :], sharex=ax1)
        event.plot_residuals(subtract_2450000=True, zorder=10)
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

        # return event

    def create_triangle(self):
        """
        Write...
        """
        pdf_dir = os.path.join(self.path, self._triangle['file_dir'])
        self._pdf = PdfPages(pdf_dir.format(self._event_id))

        if hasattr(self, 'cplot'):
            self._pdf.savefig(self.cplot)
        else:
            raise NotImplementedError("to be worked on...")
            # self._plot_triangle(self, states, labels, truths)
        self._pdf.close()

    def create_best_model(self):
        """
        Write...
        """
        pdf_dir = self._plots['best model']['file_dir']
        pdf_dir = os.path.join(self.path, pdf_dir)
        self._pdf = PdfPages(pdf_dir.format(self._event_id))

        self._plot_fit(self._res_1l2s[0], self._event_data[0])
        self._pdf.close()


def write_tables(path, settings, name, two_pspl, result,
                 fmt="ascii.commented_header"):
    """
    Save the chains, yaml results and table with results, according with the
    paths and other informations provided in the settings file.

    Args:
        path (str): directory of the Python script and catalogues.
        settings (dict): all input settings from yaml file.
        name (str): name of the photometry file.
        result (tuple): contains the EMCEE outputs and mm.Event instance.
        fmt (str, optional): format of the ascii tables.
    """

    # saving the states to file
    best, name = result[0], name.split('.')[0]
    outputs, n_emcee = settings['other_output'], settings['fitting_parameters']
    bst = dict(item for item in list(best.items()) if 'flux' not in item[0])
    if 'models' in outputs.keys():
        fname = f'{path}/' + outputs['models']['file_dir'].format(name)
        idxs_remove = list(np.arange(len(bst), len(best)))
        chains = np.delete(result[2], idxs_remove, axis=1)
        chains = Table(chains, names=list(bst.keys())+['chi2'])
        chains.write(fname, format=fmt, overwrite=True)

    # organizing results to be saved in yaml file (as in example16)
    fluxes = dict(item for item in list(best.items()) if 'flux' in item[0])
    perc = dict(item for item in result[3].items() if 'flux' not in item[0])
    perc_fluxes = dict(item for item in result[3].items() if 'flux' in item[0])
    print()
    acc_fraction = np.mean(result[1].acceptance_fraction)
    acor = result[1].get_autocorr_time(quiet=True, discard=n_emcee['nburn'])
    deg_of_freedom = result[4].datasets[0].n_epochs - len(bst)
    pspl_1, pspl_2 = two_pspl
    pspl_1 = str([round(val, 7) for val in pspl_1.values()])
    pspl_2 = str([round(val, 7) for val in pspl_2.values()])
    xlim = str([round(val, 2) for val in settings['xlim']])
    lst = ['', pspl_1, pspl_2, xlim, acc_fraction, np.mean(acor), '', '',
           result[4].chi2, deg_of_freedom, '', '']
    dict_perc_best = {6: perc, 7: perc_fluxes, 10: bst, 11: fluxes}

    # filling and writing the template
    for idx, dict_obj in dict_perc_best.items():
        for key, val in dict_obj.items():
            if idx in [6, 7]:
                uncerts = f'+{val[2]-val[1]:.5f}, -{val[1]-val[0]:.5f}'
                lst[idx] += f'    {key}: [{val[1]:.5f}, {uncerts}]\n'
            else:
                lst[idx] += f'    {key}: {val}\n'
        lst[idx] = lst[idx][:-1]
    with open(f'{path}/../1L2S-result_template.yaml') as file_:
        template_result = file_.read()
    if 'yaml output' in outputs.keys():
        yaml_fname = outputs['yaml output']['file name'].format(name)
        yaml_path = os.path.join(path, yaml_fname)
        lst[0] = sys.argv[1]
        with open(yaml_path, 'w') as yaml_results:
            yaml_results.write(template_result.format(*lst))

    # saving results to table with all the events (e.g. W16)
    if 'table output' in outputs.keys():
        fname, columns, dtypes = outputs['table output'].values()
        if not os.path.isfile(f'{path}/{fname}'):
            result_tab = Table()
            for col, dtype in zip(columns, dtypes):
                result_tab[col] = Column(name=col, dtype=dtype)
        else:
            result_tab = Table.read(f'{path}/{fname}', format='ascii')
        bst_values = [round(val, 5) for val in bst.values()]
        lst = bst_values+[0., 0.] if len(bst) == 3 else bst_values
        lst = [name] + lst + [round(result[4].chi2, 4), deg_of_freedom]
        if name in result_tab['id']:
            idx_event = np.where(result_tab['id'] == name)[0]
            if result_tab[idx_event]['chi2'] > lst[-2]:
                result_tab[idx_event] = lst
        else:
            result_tab.add_row(lst)
        result_tab.sort('id')
        result_tab.write(f'{path}/{fname}', format=fmt, overwrite=True)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')
    with open(sys.argv[1], 'r', encoding="utf-8") as yaml_input:
        all_settings = yaml.safe_load(yaml_input)
    print("Still not working as main code...")

    # Still not working...
    # save_results_binary_source = SaveResultsBinarySource(**all_settings)
    # fig_ = save_results_binary_source.best_model_plot_multiple()
    # save_results_binary_source.save_or_show_final_plot(fig_)
