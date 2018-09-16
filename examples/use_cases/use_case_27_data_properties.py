import os
import glob
import matplotlib.pyplot as pl
import numpy as np

import MulensModel as mm


#raise NotImplementedError('This use case has not been implemented')

#data_path = os.path.join(mm.MODULE_PATH, 'data', 'photometry_files')
data_path = os.path.join(mm.MODULE_PATH, 'data')
ex_arv_comments = ['\\', '|']

# Basic: Two datasets with specified data properties
ob03235_ogle_data = mm.MulensData(
    file_name=os.path.join(data_path, 'OB03235', 'OB03235_OGLE.tbl.txt'),
    phot_fmt='mag', comments=ex_arv_comments,
    plot_properties={'color': 'black', 'zorder': 10, 'show_bad': True})
ob03235_moa_data = mm.MulensData(
    file_name=os.path.join(data_path, 'OB03235', 'OB03235_MOA.tbl.txt'),
    phot_fmt='flux', comments=ex_arv_comments,
    plot_properties={'marker': 's', 'markersize': 2, 'color': 'red', 'zorder': 2,
                     'show_errorbars': False})

def suppress_a_season(data):
    data.bad = (data.time > 2451900) & (data.time < 2452300)

# Set one season to "bad"
suppress_a_season(ob03235_ogle_data)
suppress_a_season(ob03235_moa_data)

# Setting plot properties after MulensData is defined
ob03235_ogle_data.plot_properties['markersize'] = 5

# Making a plot
pl.figure()
pl.suptitle('OB03235 Data')

pl.subplot(2, 1, 1)
# Expected plot properties:
# black circles of size 5 with error bars. First season of data marker = 'x'.
# horizontal line plots *behind* data.
pl.title('OGLE Data w/ errors and bad data')
ob03235_ogle_data.plot()
pl.axhline(np.median(ob03235_ogle_data.mag), zorder=5)

pl.subplot(2, 1, 2)
# Expected plot properties:
# red squares of size 2, no error bars. Second season of data suppressed.
# horizontal line plots *in front* of data.
# legend shows file path as the data label.
pl.title('MOA Data w/o errors or bad data')
ob03235_moa_data.plot()
pl.axhline(np.median(ob03235_moa_data.flux), zorder=5)
pl.legend(loc='best')
pl.show()

# Set plot_properties for many datasets
def set_plot_properties(filename):
    """
    Get "standard" plot properties based on the filename.

    :param filename:
    :return plot_properties:
    """

    plot_properties = {}
    if 'OGLE' in filename:
        plot_properties['color']  = 'black'
        plot_properties['zorder'] = 10
    elif 'MOA' in filename:
        plot_properties['color'] = 'red'
        plot_properties['zorder'] = 2
        plot_properties['marker'] = 's'
        plot_properties['show_errorbars'] = False
    elif 'CTIO_I' in filename:
        plot_properties['color'] = 'green'

    return plot_properties


file_list = glob.glob(os.path.join(data_path, 'MB08310', '*'))
datasets = []
for file_ in file_list:
    plot_properties = set_plot_properties(file_)
    plot_properties['label'] = os.path.basename(file_).split(
        '_', maxsplit=2)[0]
    datasets.append(
        mm.MulensData(file_name=file_, comments=ex_arv_comments,
        plot_properties=plot_properties))

t_0 = 2454656.39975
u_0 = 0.00300
t_E = 11.14
t_star = 0.05487
model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 't_star': t_star})
model.set_magnification_methods(
    [t_0 - 2.* t_star, 'finite_source_uniform_Gould94', t_0 + 2. * t_star])

event = mm.Event(datasets=datasets, model=model)

pl.figure()
# Expected Behavior:
# MOA data plotted in red with points of size 2, no error bars.
# All other data have error bars.
# CTIO_I plotted in green.
# All other data sets plotted in random colors (different from each other).
# Labels set by first part of the filename.
pl.title('MB08310 Data and Model')
event.plot_data()
event.plot_model()
pl.legend(loc='best')

# Plot axes
t_start = t_0 - 3.
t_stop = t_0 + 1.
pl.ylim(17.5, 12.5)
pl.xlim(t_start, t_stop)
pl.show()
