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
# from astropy.table import Table
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
        UlensModelFit (class): parent class from example16
    """

    def __init__(self, photometry_files, models, plots, other_output=None):

        super().__init__(photometry_files, model=models['model_1'])
        filename = self._photometry_files[0]['file_name']
        self._event_id = filename.split('/')[-1].split('.')[0]
        self._plot_settings = plots['best model']
        self._list_of_models, self._colors = [*models.values()], None

        # dat_file = "./OB231171/phot/OB231171-v1-reduced.dat"
        # W16: 25, 29, 34, 53, 57 (2L1S); 24, 35 (1L2S);
        # W16_bad_canditates: candidates: 15, 20, 27, 48, 52, 56...
        self._get_datasets()
        self._check_plot_settings(**self._plot_settings)
        self._events = self._get_list_of_events()

    def _check_plot_settings(
            self, file, t_range, mag_range, coords, chi2, labels=None,
            colors=None, show_params_in_label=False, third_panel=False,
            subtract_2450000=True
            ):
        """Check if plot settings are complete and consistent with models."""

        for (key, val) in list(self._plot_settings.items())[:4]:
            if not isinstance(val, str) and val is not False:
                raise ValueError(f'{key} should be string or False.')
            if key != 'file' and val is not False and len(val.split()) != 2:
                raise ValueError(f'{key} should contain only two values.')

        self._plot_file = file
        t_r = [self._datasets[0].time[0], self._datasets[0].time[-1]]
        self._xlim = [float(t) for t in t_range.split()] if t_range else t_r
        self._mag_range = mag_range.split() if mag_range else mag_range
        self._coords = coords or '18:00:00 -30:00:00'

        if len(self._list_of_models) != len(chi2.split()):
            raise ValueError('The number of models and length of the list with'
                             'the chi2 values are different.')
        self._input_chi2 = [float(val) for val in chi2.split()]

        if labels is not None:
            if len(self._list_of_models) != len(labels.split()):
                raise ValueError('The number of models and length of the list'
                                 'with labels are different.')
            self._labels = labels.split()
        else:
            self._labels = [f'Model {i}' for i in range(len(self._input_chi2))]

        if isinstance(colors, str) and len(colors.split()) < len(chi2.split()):
            raise ValueError('More colors should be provided in the input.')

        for check in [show_params_in_label, subtract_2450000]:
            if not isinstance(check, bool):
                raise ValueError('show_params_in_label or subtract_2450000.')
        self._show_params_in_label = show_params_in_label
        self._subtract = subtract_2450000

        if third_panel not in [False, 'cumulative', 'delta']:
            raise ValueError('third_panel has invalid input.')
        self._third_panel = third_panel

    def _get_ln_probability_for_other_parameters(self):
        pass

    def _get_list_of_events(self):
        """
        Get list of mm.Event instances according to input values

        Raises:
            ValueError: parameter of a specific model is not recognized

        Returns:
            list: all mm.Event instances to be plotted inside a list
        """

        events = []
        data = self._datasets[0]  # increase for more than one catalogue?
        self._all_MM_parameters.append('t_0_par')  # temporary

        for i, item in enumerate(self._list_of_models):
            params = item['parameters'].split()
            if any(param not in self._all_MM_parameters for param in params):
                raise ValueError(f'Parameters of event_{i} are not recognized'
                                 ' in MulensModel.')
            values = [float(value) for value in item['values'].split()]
            fix_source = item.get('source_flux', None)
            fix_blend = item.get('blending_flux', None)
            model = mm.Model(dict(zip(params, values)), coords=self._coords)
            if model.n_lenses == 2:
                methods = enumerate(item['methods'].split())
                methods = [float(x) if i % 2 == 0 else x for (i, x) in methods]
                model.set_default_magnification_method(item['default method'])
                model.set_magnification_methods(methods)
            events.append(mm.Event(data, model=model,
                                   fix_source_flux={data: fix_source},
                                   fix_blend_flux={data: fix_blend}))

            if abs(events[i].get_chi2() - self._input_chi2[i]) > 0.01:
                print()
                msg = ("Chi2 of the event {:} is more than 0.01 different "
                       "than the provided chi2:\n{:}")
                warnings.warn(msg.format(self._labels[i], events[i].model))

        return events

    def _make_plot_figure(self):
        """
        Generate plot figure with matplotlib and assign all settings.

        Returns:
            matplotlib.figure.Figure: figure instance with settings.
        """

        n_panels = 3 if self._third_panel is not False else 2
        figsize = (6.6, 6.5) if self._third_panel is not False else (6.6, 5.5)
        fig = plt.figure(figsize=figsize)

        gs = GridSpec(n_panels+1, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:2, :])
        if self._mag_range is not False:
            ax1.set_ylim(*self._mag_range)
        fig.add_subplot(gs[2:3, :], sharex=ax1)

        if self._third_panel is not False:
            ax3 = fig.add_subplot(gs[3:, :], sharex=ax1)
            ax3.set_xlabel('Time - 2450000' if self._subtract else 'Time')
            if self._third_panel == "cumulative":
                ax3.set_ylabel(r'Cumulative $\Delta\chi^2$')
            else:
                ax3.set_ylabel(r'$\Delta\chi^2$')
                ax3.set_ylim(-9.9, 9.9)

        for ax in fig.get_axes():
            ax.tick_params(axis='both', direction='in')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')

        return fig

    def _get_delta_chi2_events(self, event1, event2, cumulative=True):
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

        return np.cumsum(chi2_diff) if cumulative else chi2_diff

    def _get_label_with_model_parameters(self, event):
        """
        Get the label with model parameters for a specific event.

        Args:
            event (mm.Event): instance of MulensModel event.

        Returns:
            str: the label with chi2 and model parameters.
        """

        label_more = f': chi2={event.get_chi2():.2f}'
        if self._show_params_in_label:
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

        zip_models = zip(self._events, self._labels, self._colors)
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
        events = self._events
        events[0].plot_residuals(subtract_2450000=self._subtract, zorder=1)
        epochs = np.linspace(*self._xlim, 10000)
        base_lc = events[0].model.get_lc(epochs,
                                         source_flux=events[0].fluxes[0][0],
                                         blend_flux=events[0].fluxes[0][1])

        for (event, color) in zip(events[1:], self._colors[1:]):
            comp_lc = event.model.get_lc(epochs,
                                         source_flux=event.fluxes[0][0],
                                         blend_flux=event.fluxes[0][1])
            plt.plot(epochs - int(self._subtract)*2450000, comp_lc-base_lc,
                     color=color, **all_params)
            if self._third_panel is not False:
                ax3 = fig.get_axes()[-1]
                cumul = self._third_panel == 'cumulative'
                chi2_ = self._get_delta_chi2_events(events[0], event, cumul)
                ax3.plot(self._datasets[0].time - int(self._subtract)*2450000,
                         chi2_, color=color, **all_params)
                ax3.axhline(0, color='black', lw=1.5, ls='--', zorder=1)

    def best_model_plot_multiple(self):
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
        self._colors = self._colors[:len(self._list_of_models)]
        plt_params = {'lw': 2.5, 'alpha': 0.8, 'zorder': 10}
        model_params = {'t_start': self._xlim[0], 't_stop': self._xlim[1],
                        'subtract_2450000': self._subtract}
        xlim_new = np.array(self._xlim) - int(self._subtract)*2450000

        fig = self._make_plot_figure()
        self._plot_model_upper_panel(fig, {**plt_params, **model_params})
        self._plot_residuals_chi2(fig, plt_params)
        plt.xlim(*xlim_new)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fig.get_axes()[0].legend(loc='best')

        return fig

    def save_or_show_final_plot(self, fig):
        """
        Save or show the final plot based on the provided plot settings.

        Args:
            fig (matplotlib.figure.Figure): figure to be saved or shown.

        Raises:
            ValueError: If the output format is not PDF or PNG.
        """

        if self._plot_file is not False:

            file_format = os.path.splitext(self._plot_file)[1]
            if file_format == '.pdf':
                pdf = PdfPages(self._plot_file)
                pdf.savefig(fig)
                pdf.close()
            elif file_format == '.png':
                plt.savefig(self._plot_file)
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
    fig_ = plot_multiple_models.best_model_plot_multiple()
    plot_multiple_models.save_or_show_final_plot(fig_)
