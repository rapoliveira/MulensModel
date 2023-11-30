"""
Plot several fits (PSPL, parallax, 1L2S, 2L1S) to data

The code produces a plot figure containing several fits (standard is three)
to the data in magnitudes, including a panel with the residuals and an optional
one with the cumulative delta_chi2.
"""
import os
import sys
import warnings
import yaml
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import MulensModel as mm
import numpy as np

from ulens_model_fit import UlensModelFit


class PlotMultipleModels(UlensModelFit):
    """
    Class for plotting multiple models to data, including the relative
    residuals and optional cumulative delta_chi2.

    Args:
        UlensModelFit (_type_): _description_  ==>> MISSING!
    """

    def __init__(self, photometry_files, models, plots, other_output=None):

        # UlensModelFit.__init__(self, photometry_files)
        self._photometry_files = photometry_files
        filename = photometry_files[0]['file_name']
        self._event_id = filename.split('/')[-1].split('.')[0]
        self._plot_settings = plots['best model']
        self._list_of_models = [*models.values()]

        # self._read_data()
        self._residuals_output = False
        self._get_datasets()
        self._check_plot_settings(**self._plot_settings)
        self._events = self._get_list_of_events()

        # self._fig = self._best_model_plot_multiple()
        # self._save_or_show_final_plot(self._fig)

    def _read_data(self, test=None):
        """
        Read a catalogue and create MulensData instance

        Args:
            self (_type_): _description_

        Returns:
            tuple: identification of the event and MulensData instance
        """

        # dat_file = "./OB231171/phot/OB231171-v1-reduced.dat"
        # W16: 25, 29, 34, 53, 57 (2L1S); 24, 35 (1L2S);
        # W16_bad_canditates: candidates: 15, 20, 27, 48, 52, 56...
        # dat_file, add_2450000 = settings['photometry_files'][0].values()
        dat_file, add_2450000 = self._photometry_files[0].values()
        # self._event_id = os.path.splitext(dat_file)[0].split('/')[-1]
        tab = Table.read(dat_file, format='ascii')
        my_dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
        if add_2450000:
            my_dataset[0] += 2450000
        self._datasets = []
        self._datasets.append(mm.MulensData(my_dataset, phot_fmt='mag'))

        if test is not None:
            plt.figure(tight_layout=True)
            my_dataset.plot(phot_fmt='mag')
            plt.show()

    def _check_plot_settings(
            self, file, time_range, magnitude_range, coords, chi2,
            show_model_parameters_in_label=False, third_panel=False,
            labels=None, colors=None, subtract_2450000=True
            ):
        """
        Check if the plot settings are complete and consistent with models.

        Args:
            models (List[object]): list of all inserted models
            plot_settings (dict): dictionary with the plot settings

        Raises:
            ValueError: if some of the main inputs are not given
            ValueError: if the number of models and chi2 values are different
            ValueError: if the number of models and labels are different
        """

        # main_keys = ['time range', 'magnitude range', 'coords', 'chi2']
        # if any(key not in self._plot_settings.keys() for key in main_keys):
        #     raise ValueError('Some of the main keys are not given in input.')

        # TO-DO :: Still need to check all the keys... !!!

        if len(self._list_of_models) != len(chi2.split()):
            raise ValueError('The number of models and length of the list with'
                             'the chi2 values are different.')

        aux = [f'Model {i}' for i in range(len(self._list_of_models))]
        self._model_labels = labels.split() if labels else aux
        if labels and len(self._list_of_models) != len(labels.split()):
            raise ValueError('The number of models and length of the list with'
                             'labels are different.')

    def _get_list_of_events(self):
        """
        Get list of mm.Event instances according to input values

        Args:
            list_of_models (list): list with all the inserted models (dicts)
            data (mm.Data): _description_
            plot_settings (dict): settings of the plot figure

        Returns:
            list: all mm.Event instances to be plotted inside a list
        """

        events = []
        coords = self._plot_settings.get('coords', '18:00:00 -30:00:00')
        data = self._datasets[0]  # increase for more than one catalogue?
        chi2_in = [float(val) for val in self._plot_settings['chi2'].split()]
        self._task = 'plot'
        self._set_default_parameters()
        self._all_MM_parameters.append('t_0_par')  # temporary

        for i, item in enumerate(self._list_of_models):
            params = item['parameters'].split()
            if any(param not in self._all_MM_parameters for param in params):
                raise ValueError(f'Parameters of event_{i} are not recognized'
                                 ' in MulensModel.')
            values = [float(value) for value in item['values'].split()]
            fix_source = item.get('source_flux', None)
            fix_blend = item.get('blending_flux', None)
            model = mm.Model(dict(zip(params, values)), coords=coords)
            if model.n_lenses == 2:
                methods = enumerate(item['methods'].split())
                methods = [float(x) if i % 2 == 0 else x for (i, x) in methods]
                model.set_default_magnification_method(item['default method'])
                model.set_magnification_methods(methods)
            events.append(mm.Event(data, model=model,
                                   fix_source_flux={data: fix_source},
                                   fix_blend_flux={data: fix_blend}))

            if abs(events[i].get_chi2() - chi2_in[i]) > 0.01:
                print()
                msg = ("Chi2 of the event {:} is more than 0.01 different "
                       "than the provided chi2:\n{:}")
                warnings.warn(msg.format(self._model_labels[i],
                                         events[i].model))

        return events

    def _make_plot_figure(self):
        """
        Generate plot figure with matplotlib and assign all settings.

        Raises:
            ValueError: if key for third_panel is invalid.

        Returns:
            matplotlib.figure.Figure: figure instance with settings.
        """

        third_panel = self._plot_settings.get('third_panel', False)
        n_panels = 3 if third_panel is not False else 2
        figsize = (6.6, 6.5) if third_panel is not False else (6.6, 5.5)
        fig = plt.figure(figsize=figsize)

        gs = GridSpec(n_panels+1, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:2, :])
        if self._plot_settings.get('magnitude_range', False) is not False:
            ax1.set_ylim(*self._plot_settings['magnitude_range'].split())
        fig.add_subplot(gs[2:3, :], sharex=ax1)

        if third_panel is not False:
            ax3 = fig.add_subplot(gs[3:, :], sharex=ax1)
            subtract = self._plot_settings.get('subtract_2450000', True)
            ax3.set_xlabel('Time - 2450000' if subtract else 'Time')
            if third_panel == "cumulative":
                ax3.set_ylabel(r'Cumulative $\Delta\chi^2$')
            elif third_panel == "delta":
                ax3.set_ylabel(r'$\Delta\chi^2$')
                ax3.set_ylim(-9.9, 9.9)
            else:
                raise ValueError('Invalid value for third_panel key. Options'
                                 ' are: False, cumulative or delta.')

        for ax in fig.get_axes():
            ax.tick_params(axis='both', direction='in')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')

        return fig

    def _get_delta_chi2_two_events(self, event1, event2, cumulative=True):
        """
        Get delta chi2 for event2 points (or cumulative) compared to event1.

        Args:
            event1 (mm.Event): event used as basis (parallax for W16)
            event2 (mm.Event): event being compared (binary models for W16)
            cumulative (bool, optional): cumulative or not. Defaults to True.

        Returns:
            np.array: chi2 difference for each point or cumulative
        """

        # 1 way to do it: calculating flux for each point and subtracting
        # data = event1.datasets[0]
        # (flux, blend) = aux_event.get_flux_for_dataset(0)
        # fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux+blend
        # [...]

        # 2nd way: event1.get_chi2_per_point()
        chi2_per_point_1 = event1.get_chi2_per_point()[0]
        chi2_per_point_2 = event2.get_chi2_per_point()[0]
        chi2_diff = chi2_per_point_2 - chi2_per_point_1

        self._chi2_ans = np.cumsum(chi2_diff) if cumulative else chi2_diff

        return self._chi2_ans

    def _get_label_with_model_parameters(self, event):
        """
        Get the label with model parameters for a specific event.

        Args:
            event (mm.Event): instance of MulensModel event.

        Returns:
            str: the label with chi2 and model parameters.
        """

        label_more = f': chi2={event.get_chi2():.2f}'
        if self._plot_settings.get('show_model_parameters_in_label', False):
            params = event.model.parameters.as_dict()
            param_names = ', '.join(params.keys())
            param_values = ', '.join(str(value) for value in params.values())
            label_more += f'\n[{param_names}] =\n[{param_values}]'

        return label_more

    def _plot_model_upper_panel(self, fig, all_params):
        """
        Plot the upper panel with the data and best models

        Args:
            fig (matplotlib.figure.Figure): figure to plot the models.
            all_params (dict): kwarg settings for plots
        """

        plt.axes(fig.get_axes()[0])
        event_base, subt = self._events[0], all_params['subtract_2450000']
        event_base.plot_data(subtract_2450000=subt, label=self._event_id)

        zip_models = zip(self._events, self._model_labels, self._colors)
        for (event, label, color) in zip_models:
            label += self._get_label_with_model_parameters(event)
            event.plot_model(label=label, color=color, **all_params)

    def _plot_residuals_chi2(self, fig, all_params):
        """
        Plot residuals and delta chi2 in the lower panels

        Args:
            fig (matplotlib.figure.Figure): figure to plot the models.
            all_params (dict): kwarg settings for plots
        """

        plt.axes(fig.get_axes()[1])
        xlim = [all_params.pop('t_start'), all_params.pop('t_stop')]
        subtract = all_params.pop('subtract_2450000')

        events = self._events
        events[0].plot_residuals(subtract_2450000=subtract, zorder=1)
        epochs = np.linspace(*xlim, 10000)
        base_lc = events[0].model.get_lc(epochs,
                                         source_flux=events[0].fluxes[0][0],
                                         blend_flux=events[0].fluxes[0][1])

        for (event, color) in zip(events[1:], self._colors[1:]):
            comp_lc = event.model.get_lc(epochs,
                                         source_flux=event.fluxes[0][0],
                                         blend_flux=event.fluxes[0][1])
            plt.plot(epochs-self._subt_value, comp_lc-base_lc, color=color,
                     **all_params)
            if self._plot_settings.get('third_panel', False) is not False:
                ax3 = fig.get_axes()[-1]
                cumulative = self._plot_settings['third_panel'] == 'cumulative'
                self._get_delta_chi2_two_events(events[0], event, cumulative)
                ax3.plot(self._datasets[0].time - self._subt_value,
                         self._chi2_ans, color=color, **all_params)
                ax3.axhline(0, color='black', lw=1.5, ls='--', zorder=1)

    def _best_model_plot_multiple(self):
        """
        Plot multiple fits on a figure.

        Args:
            fig (matplotlib.figure.Figure): The figure to plot on.
            plot_settings (dict): dictionary with the plot settings.

        Returns:
            matplotlib.figure.Figure: The updated figure with the plotted fits.
        """

        if 'colors' in self._plot_settings.keys():
            self._colors = self._plot_settings['colors'].split()
        else:
            self._colors = ['black'] + list(mcolors.TABLEAU_COLORS.keys())[1:]
        t_min_max = [self._datasets[0].time[0], self._datasets[0].time[-1]]
        xlim = self._plot_settings.get('time_range')
        xlim = [float(val) for val in xlim.split()] if xlim else t_min_max
        self._subtract = self._plot_settings.get('subtract_2450000', True)
        plt_params = {'lw': 2.5, 'alpha': 0.8, 'zorder': 10}
        all_params = {**plt_params, **{'t_start': xlim[0], 't_stop': xlim[1],
                                       'subtract_2450000': self._subtract}}
        self._subt_value = 2450000 if self._subtract else 0
        self._xlim = np.array(xlim) - self._subt_value
        # breakpoint()

        fig = self._make_plot_figure()
        self._plot_model_upper_panel(fig, all_params)
        self._plot_residuals_chi2(fig, all_params)
        plt.xlim(*self._xlim)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fig.get_axes()[0].legend(loc='best')

        return fig

    def _save_or_show_final_plot(self, fig):
        """
        Save or show the final plot based on the provided plot settings.

        Args:
            fig (matplotlib.figure.Figure): figure to be saved or shown.

        Raises:
            ValueError: If the output format is not PDF or PNG.
        """

        if 'file' in self._plot_settings.keys():

            file_format = os.path.splitext(self._plot_settings['file'])[1]
            if file_format == '.pdf':
                pdf = PdfPages(self._plot_settings['file'])
                pdf.savefig(fig)
                pdf.close()
            elif file_format == '.png':
                plt.savefig(self._plot_settings['file'])
            else:
                raise ValueError('PDF or PNG format is required for output.')

        else:
            plt.show()

        plt.close('all')


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')
    with open(sys.argv[1], 'r', encoding="utf-8") as yaml_input:
        all_settings = yaml.safe_load(yaml_input)

    plot_multiple_models = PlotMultipleModels(**all_settings)
    fig_ = plot_multiple_models._best_model_plot_multiple()
    plot_multiple_models._save_or_show_final_plot(fig_)
