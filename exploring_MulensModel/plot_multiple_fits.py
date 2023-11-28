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


# def produce_yaml_file_from...
# Take the prepare code from Radek as a basis


def read_data(settings):
    """
    Read a catalogue and create MulensData instance

    Args:
        settings (dict): all settings from yaml file.

    Returns:
        tuple: identification of the event and MulensData instance
    """

    # dat_file = "./OB231171/phot/OB231171-v1-reduced.dat"
    # W16: 25, 29, 34, 53, 57 (2L1S); 24, 35 (1L2S); 15, 20, 27, 48, 52, 56...
    dat_file, add_2450000 = settings['photometry_files'][0].values()
    dat_id = os.path.splitext(dat_file)[0].split('/')[-1]
    tab = Table.read(dat_file, format='ascii')
    my_dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
    if add_2450000:
        my_dataset[0] += 2450000
    my_dataset = mm.MulensData(my_dataset, phot_fmt='mag')

    # fig = plt.figure(tight_layout=True)
    # my_dataset.plot(phot_fmt='mag')
    # plt.show()

    return (dat_id, my_dataset)


def check_model_inputs(list_of_models, plot_settings):
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

    main_keys = ['time range', 'magnitude range', 'coords', 'chi2']
    if any(key not in plot_settings.keys() for key in main_keys):
        raise ValueError('Some of the main keys are not given in input file.')

    if len(list_of_models) != len(plot_settings['chi2'].split()):
        raise ValueError('The number of models and length of the list with the'
                         'chi2 values are different.')
    if ('labels' in plot_settings.keys() and
            len(list_of_models) != len(plot_settings['labels'].split())):
        raise ValueError('The number of models and length of the list with'
                         'labels are different.')


def get_list_of_events(list_of_models, data, plot_settings):
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
    coords = plot_settings.get('coords', '18:00:00 -30:00:00')
    input_chi2 = [float(val) for val in plot_settings['chi2'].split()]

    for i, item in enumerate(list_of_models):

        params = item['parameters'].split()
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

        if abs(events[i].get_chi2() - input_chi2[i]) > 0.01:
            print()
            msg = ("Chi2 of the event {:} is more than 0.01 different than "
                   "the provided chi2:\n{:}")
            warnings.warn(msg.format(plot_settings['labels'].split()[i],
                                     events[i].model))

    return events


def generate_plot_figure(plot_settings):
    """
    _summary_

    Args:
        plot_settings (_type_): _description_

    Returns:
        _type_: _description_
    """

    third_panel = plot_settings.get('third panel', False)
    n_panels = 3 if third_panel is not False else 2
    figsize = (6.6, 6.5) if third_panel is not False else (6.6, 5.5)
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(n_panels+1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :])
    if plot_settings.get('magnitude range', False) is not False:
        ax1.set_ylim(*plot_settings['magnitude range'].split())
    fig.add_subplot(gs[2:3, :], sharex=ax1)

    if third_panel is not False:
        ax3 = fig.add_subplot(gs[3:, :], sharex=ax1)
        subtract = plot_settings.get('subtract_2450000', True)
        ax3.set_xlabel('Time - 2450000' if subtract else 'Time')
        if third_panel == "cumulative":
            ax3.set_ylabel(r'Cumulative $\Delta\chi^2$')
        elif third_panel == "delta":
            ax3.set_ylabel(r'$\Delta\chi^2$')
            ax3.set_ylim(-9.9, 9.9)
        else:
            raise ValueError('Invalid input in third panel of the best model.')

    for ax in fig.get_axes():
        ax.tick_params(axis='both', direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

    return fig


def get_delta_chi2_two_events(event1, event2, cumulative=True):
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
    # fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend
    # [...]

    # 2nd way: event1.get_chi2_per_point()
    chi2_per_point_1 = event1.get_chi2_per_point()[0]
    chi2_per_point_2 = event2.get_chi2_per_point()[0]
    chi2_diff = chi2_per_point_2 - chi2_per_point_1

    return np.cumsum(chi2_diff) if cumulative else chi2_diff


def get_label_with_model_parameters(event, plot_settings):
    """
    Get the label with model parameters for an event.

    Args:
        event (mm.Event): The event.
        plot_settings (dict): A dictionary containing plot settings.

    Returns:
        str: the label with chi2 and model parameters.
    """

    label_more = f': chi2={event.get_chi2():.2f}'
    if plot_settings.get('show_model_parameters_in_label', False):
        dict_params = event.model.parameters.as_dict()
        param_names = ', '.join(dict_params.keys())
        param_values = ', '.join(str(value) for value in dict_params.values())
        label_more += f'\n[{param_names}] =\n[{param_values}]'

    return label_more


def plot_model_upper_panel(fig, events, plot_settings, all_params):
    """
    Plot the upper panel with the data and best models

    Args:
        fig (matplotlib.figure.Figure): figure to plot the models.
        events (List[object]): list of MulensModel events.
        plot_settings (dict): A dictionary containing plot settings.
        all_params (dict): kwarg settings for plots
    """

    plt.axes(fig.get_axes()[0])
    subt, colors = all_params['subtract_2450000'], plot_settings['colors']
    events[0].plot_data(subtract_2450000=subt, label=plot_settings['event_id'])

    model_labels = plot_settings.get('labels', 'Parallax 1L2S 2L1S').split()
    for (event, label, color) in zip(events, model_labels, colors):
        label += get_label_with_model_parameters(event, plot_settings)
        event.plot_model(label=f'{label}', color=color, **all_params)


def plot_residuals_chi2(fig, events, plot_settings, all_params):
    """
    Plot residuals and delta chi2 in the lower panels

    Args:
        fig (matplotlib.figure.Figure): figure to plot the models.
        events (List[object]): list of MulensModel events.
        plot_settings (dict): A dictionary containing plot settings.
        all_params (dict): kwarg settings for plots
    """

    plt.axes(fig.get_axes()[1])
    xlim = [all_params.pop('t_start'), all_params.pop('t_stop')]
    subtract = all_params.pop('subtract_2450000')
    subt_value = 2450000 if subtract else 0
    events[0].plot_residuals(subtract_2450000=subtract, zorder=1)
    epochs = np.linspace(*xlim, 10000)
    base_lc = events[0].model.get_lc(epochs,
                                     source_flux=events[0].fluxes[0][0],
                                     blend_flux=events[0].fluxes[0][1])

    for (event, color) in zip(events[1:], plot_settings['colors'][1:]):
        comp_lc = event.model.get_lc(epochs,
                                     source_flux=event.fluxes[0][0],
                                     blend_flux=event.fluxes[0][1])
        plt.plot(epochs-subt_value, comp_lc-base_lc, color=color, **all_params)
        if plot_settings.get('third panel', False) is not False:
            ax3 = fig.get_axes()[-1]
            cumulative = plot_settings.get('third panel') == 'cumulative'
            chi2_ = get_delta_chi2_two_events(events[0], event, cumulative)
            ax3.plot(events[0].datasets[0].time - subt_value, chi2_,
                     color=color, **all_params)
            ax3.axhline(0, color='black', lw=1.5, ls='--', zorder=1)


def plot_multiple_fits(events, plot_settings):
    """
    Plot multiple fits on a figure.

    Args:
        fig (matplotlib.figure.Figure): The figure to plot on.
        events (List[object]): list of MulensModel events.
        data_label (str): the label for the data.
        plot_settings (dict): dictionary with the plot settings.

    Returns:
        matplotlib.figure.Figure: The updated figure with the plotted fits.
    """

    if 'colors' in plot_settings.keys():
        plot_settings['colors'] = plot_settings['colors'].split()
    else:
        std_colors = mcolors.TABLEAU_COLORS.keys()
        plot_settings['colors'] = ['black'] + list(std_colors)[1:]
    data = events[0].datasets[0]
    xlim = plot_settings.get('time range', f'{data.time[0]} {data.time[-1]}')
    xlim = [float(val) for val in xlim.split()]
    subtract = plot_settings.get('subtract_2450000', True)
    plt_params = {'lw': 2.5, 'alpha': 0.8, 'zorder': 10}
    all_params = {**plt_params, **{'t_start': xlim[0], 't_stop': xlim[1],
                                   'subtract_2450000': subtract}}

    fig = generate_plot_figure(plot_settings)
    plot_model_upper_panel(fig, events, plot_settings, all_params)
    plot_residuals_chi2(fig, events, plot_settings, all_params)

    subt_value = 2450000 if subtract else 0
    plt.xlim(xlim[0]-subt_value, xlim[1]-subt_value)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    fig.get_axes()[0].legend(loc='best')

    return fig


def save_or_show_final_plot(fig, plot_settings):
    """
    Save or show the final plot based on the provided plot settings.

    Args:
        fig (matplotlib.figure.Figure): figure to be saved or shown.
        plot_settings (Dict[str, str]): A dictionary containing plot settings.

    Raises:
        ValueError: If the output format is not PDF or PNG.
    """

    if 'file' in plot_settings.keys():

        if os.path.splitext(plot_settings['file'])[1] == '.pdf':
            pdf = PdfPages(plot_settings['file'])
            pdf.savefig(fig)
            pdf.close()
        elif os.path.splitext(plot_settings['file'])[1] == '.png':
            plt.savefig(plot_settings['file'])
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

    event_id, event_data = read_data(all_settings)
    plt_settings = all_settings['plots']['best model']
    plt_settings['event_id'] = event_id
    models = [val for (key, val) in all_settings.items() if 'model' in key]
    check_model_inputs(models, plt_settings)
    list_of_events = get_list_of_events(models, event_data, plt_settings)

    fig_ = plot_multiple_fits(list_of_events, plt_settings)
    save_or_show_final_plot(fig_, plt_settings)
