"""
Script for plotting the model using UlensModelFit class.
All the settings are read from a YAML file.
"""
import sys
from astropy.table import Table
# from os import path
import os
import yaml

from ulens_model_fit import UlensModelFit


def get_59_yaml_files(path):

    coord_w16 = Table.read('W16-coordinates.txt', format='ascii')
    names, coords = coord_w16['lens_name'], coord_w16['ra_j2000', 'dec_j2000']

    with open(f'{path}/W16_PAR07_parallax-bkpi.yaml') as data:
        settings = yaml.safe_load(data)

    for event, coord in zip(names, coords):
        
        # phot_file = f'data/Wyrzykowski+16/{event}.dat'
        phot_file = {'file_name': f'../W16_photometry/{event}.dat',
                     'add_2450000': True}
        settings['photometry_files'][0] = phot_file
        settings['model']['coords'] = f'{coord[0]} {coord[1]}'
        settings['plots']['best model']['file'] = f'figs/{event}_model.png'
        settings['plots']['triangle']['file'] = f'figs/{event}_triangle.png'
        settings['fitting_parameters'] = {'n_walkers': 20, 'n_steps': 20000,
                                          'n_burn': 10000}

        filename = f"W16_{event.replace('-','')}_parallax.yaml"
        with open(f'{path}/yaml_files/{filename}', 'w') as file:
            yaml.dump(settings, file, sort_keys=False, indent=4)
        print(event, 'done!')

    return names, coords


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     raise ValueError('Exactly one argument needed - YAML file')
    # input_file = sys.argv[1]
    # input_file_root = os.path.splitext(input_file)[0]

    # BEFORE: run get_59_yaml_files just once, edit after that 
    path = os.path.dirname(os.path.realpath(__file__))
    # names, coords = get_59_yaml_files(path)

    # AFTER:
    all_events = sorted(os.listdir(f'{path}/yaml_files/'))
    for item in all_events:
        if 'bkpi' in item:  # not in
            with open(f'{path}/yaml_files/{item}', 'r') as data:
                settings = yaml.safe_load(data)
            
            print(f'\n- Starting fit for \033[1m{item}\033[0m')
            settings['fitting_parameters']['n_walkers'] = 10
            settings['fitting_parameters']['n_steps'] = 10000
            settings['fitting_parameters']['n_burn'] = 5000
            ulens_model_fit = UlensModelFit(**settings)
            ulens_model_fit.run_fit()
            # test = ulens_model_fit._parse_results()
            print('----------------------------------------------')
            # breakpoint()
    
    breakpoint()


    breakpoint()

    # with open(input_file, 'r') as data:
    #     settings = yaml.safe_load(data)

    # # Remove settings that are not used for plotting:
    # keys = ["starting_parameters", "min_values", "max_values",
    #         "fitting_parameters"]
    # for key in keys:
    #     settings[key] = None
    # if "plots" in settings:
    #     if "triangle" in settings["plots"]:
    #         settings["plots"].pop("triangle")

    # ulens_model_fit = UlensModelFit(**settings)

    # ulens_model_fit.plot_best_model()
