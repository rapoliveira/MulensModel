"""
TO EDIT ::: Fits binary source model using EMCEE sampler.

The code simulates binary source light curve and fits the model twice:
with source flux ratio found via linear regression and
with source flux ratio as a chain parameter.
"""
import os
import sys
import warnings
import yaml
import numpy as np
import MulensModel as mm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from matplotlib.gridspec import GridSpec


# def produce_yaml_file_from...


def read_data(settings):

    # dat_file = "./OB231171/phot/OB231171-v1-reduced.dat"
    # dat_file = 'W16-59events/phot/PAR-29.dat'  # 25** (both), 29, 34, 53, 57 (best in 2L1S)
    # dat_file = 'W16-59events/phot/PAR-35.dat' # 24 (t), 35 (y), run2Sand2L: 15*/20*/25ok/27no/48*/52no/56no
    dat_file, subtract = settings['photometry_files'][0].values()
    event_id = os.path.splitext(dat_file)[0].split('/')[-1]
    tab = Table.read(dat_file, format='ascii')
    my_dataset = np.array([tab['col1'], tab['col2'], tab['col3']])
    if subtract:
        my_dataset[0] += 2450000
    my_dataset = mm.MulensData(my_dataset, phot_fmt='mag')

    # fig = plt.figure(tight_layout=True)
    # my_dataset.plot(phot_fmt='mag')
    # plt.show()

    return event_id, my_dataset


def get_cumul_chi2_two_models(event1, event2):

    # 1 way to do it: calculating flux for each point and subtracting
    # data = event1.datasets[0]
    # (flux, blend) = aux_event.get_flux_for_dataset(0)
    # fsub = data.flux - aux_event.fits[0].get_model_fluxes() + flux + blend

    # 2nd way: event1.get_chi2_per_point()
    chi2_per_point_1 = event1.get_chi2_per_point()[0]
    chi2_per_point_2 = event2.get_chi2_per_point()[0]
    chi2_diff = chi2_per_point_1 - chi2_per_point_2

    return chi2_diff, np.cumsum(chi2_diff)


def check_model_parameters(models, plot_settings):

    if len(models) != len(plot_settings['labels'].split()):
        raise ValueError('The length of the lists containing best parameters '
                         'and model labels are different.')
    elif len(models) != len(plot_settings['chi2'].split()):
        raise ValueError('The length of the lists containing best parameters '
                         'and chi2 values are different.')


def get_list_of_events(dict_models, data, plot_settings):

    events = []
    coords = plot_settings.get('coords', '18:00:00 -30:00:00')
    input_chi2 = [float(val) for val in plot_settings['chi2'].split()]
    
    for i, item in enumerate(dict_models):
        
        params = item['parameters'].split()
        values = [float(value) for value in item['values'].split()]
        fix_source = item.get('source_flux', None)
        fix_blend = item.get('blending_flux', None)
        model = mm.Model(dict(zip(params, values)), coords=coords)
        if model.n_lenses == 2:
            default_method = item['default method']
            methods = enumerate(item['methods'].split())
            methods = [float(x) if i % 2 == 0 else x for (i, x) in methods]
            model.set_default_magnification_method(default_method)
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

    n_panels = 3 if plot_settings['cumulative chi2'] else 2
    figsize = (6.6, 6.5) if plot_settings['cumulative chi2'] else (6.6, 5.5)
    fig = plt.figure(figsize=figsize)
    
    gs = GridSpec(n_panels+1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :])
    if plot_settings.get('magnitude range', 'None') != 'None':
        ax1.set_ylim(*plot_settings['magnitude range'])

    ax2 = fig.add_subplot(gs[2:3, :], sharex=ax1)

    if plot_settings['cumulative chi2']:
        ax3 = fig.add_subplot(gs[3:, :], sharex=ax1)
        ax3.set_xlabel('Time - 2450000')
        ax3.set_ylabel(r'Cumulative $\Delta\chi^2$')
    
    for ax in fig.get_axes():
        ax.tick_params(axis='both', direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

    return fig


def plot_multiple_fits(fig, events, data_label, plot_settings, pdf=None):

    # get important values
    data = events[0].datasets[0]
    model_labels = plot_settings.get('labels').split()
    colors = plot_settings.get('colors').split()
    xlim = plot_settings.get('time range', f'{min(data.time)} {max(data.time)}')
    xlim = [float(val) for val in xlim.split()]

    ax1, ax2 = fig.get_axes()[:2]
    plt.axes(ax1)
    events[0].plot_data(subtract_2450000=True, label=data_label)
    plt_params = {'lw': 2.5, 'alpha': 0.8, 'zorder': 10}
    params = {**plt_params, **{'t_start': xlim[0], 't_stop': xlim[1],
                               'subtract_2450000': True}}
    
    # Setting lists to get everything in a for loop
    for (event, label, color) in zip(events, model_labels, colors):
        label += f': chi2={event.get_chi2():.2f}'
        if plot_settings.get('show_model_parameters_in_label', False):
            label += '\n[t_0_1, u_0_1, t_0_2, u_0_2, t_E] =\n'
            label += '[10178.71, 0.15, 10112.82, 0.59, 15.26]'
        event.plot_model(label=r'%s'%label, color=color, **params)

    # Obtaining and plotting residuals for the binary models
    plt.axes(ax2)
    events[0].plot_residuals(subtract_2450000=True, zorder=1)
    epochs = np.linspace(*xlim, 1000)
    base_lc = events[0].model.get_lc(epochs, source_flux=events[0].fluxes[0][0],
                                     blend_flux=events[0].fluxes[0][1])
    for (event, color) in zip(events[1:], colors[1:]):
        comp_lc = event.model.get_lc(epochs, source_flux=event.fluxes[0][0],
                                     blend_flux=event.fluxes[0][1])
        ax2.plot(epochs-2450000, base_lc-comp_lc, color=color, **plt_params)
    
    # Plot the cumulative chi2 distribution
    if plot_settings['cumulative chi2']:
        ax3 = fig.get_axes()[-1]
        chi2_diff_12, cumul_chi2_12 = get_cumul_chi2_two_models(events[0], events[1])
        chi2_diff_13, cumul_chi2_13 = get_cumul_chi2_two_models(events[0], events[2])
        # ax3.plot(data.time-2450000, chi2_diff_12, lw=2, color='green', zorder=10,
        #          label='1L2S')
        # ax3.plot(data.time-2450000, chi2_diff_13, lw=2, color='gray', zorder=10,
        #          label='2L1S')
        # ax3.axhline(0, color='black', lw=1.5, ls='--', label='Parallax', zorder=1)
        # ax3.set_ylim(-9.9, 9.9)
        # ax3.set_ylabel(r'$\Delta\chi^2$')
        ax3.plot(data.time-2450000, cumul_chi2_12, lw=2, color='green', zorder=10)
        ax3.plot(data.time-2450000, cumul_chi2_13, lw=2, color='gray', zorder=10)
        ax3.axhline(0, color='black', lw=1.5, ls='--', zorder=1)
    ax1.legend(loc='best')
    plt.xlim(xlim[0]-2450000, xlim[1]-2450000)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if pdf is not None:
        pdf.savefig(fig)
        plt.close('all')
    else:
        plt.show()
    pdf.close()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError('Exactly one argument needed - YAML file')

    with open(sys.argv[1], 'r') as input_data:
        settings = yaml.safe_load(input_data)
    event_id, data = read_data(settings)
    plot_settings = settings['plots']['best model']
    pdf = PdfPages(plot_settings['file'])

    dict_models = [val for (key, val) in settings.items() if 'model' in key]
    check_model_parameters(dict_models, plot_settings)
    events = get_list_of_events(dict_models, data, plot_settings)
    fig = generate_plot_figure(plot_settings)
    plot_multiple_fits(fig, events, event_id, plot_settings, pdf=pdf)
