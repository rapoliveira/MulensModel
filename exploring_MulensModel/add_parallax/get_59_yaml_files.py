"""
Write documentation later
"""
import sys
from astropy.io import ascii
from astropy.table import Table
import numpy as np
import os
# import yaml

# from ulens_model_fit import UlensModelFit


def get_59_yaml_files(path, template, coord_w16, all_settings):

    names, coords = coord_w16['lens_name'], coord_w16['ra_j2000', 'dec_j2000']
    n_emcee = [20, 20000, 10000]  # [10, 10000, 5000]

    # with open(f'{path}/W16_PAR07_parallax-bkpi.yaml') as data:
    #     settings = yaml.safe_load(data)
    
    for event, coord, line in zip(names, coords, all_settings):
        
        # new code: Radek (16.aug.2023)
        lst = np.array(list(line))[1:].tolist()
        lst.insert(4, f'{coord[0]} {coord[1]}')
        lst[5:5] = n_emcee
        with open(line[0], 'w') as out_file:
            if float(lst[2]) >= 0.:
                out_file.write(template.format(*lst))
            else:  # if initial u_0 < 0 (change from min to max value)
                split = template.split('\n')
                split.insert(17, split.pop(14))
                out_file.write('\n'.join(split).format(*lst))

        # old code: Raphael (14.aug.2023)
        # phot_file = {'file_name': f'../W16_photometry/{event}.dat',
        #              'add_2450000': True}
        # settings['photometry_files'][0] = phot_file
        # settings['model']['coords'] = f'{coord[0]} {coord[1]}'
        # settings['plots']['best model']['file'] = f'figs/{event}_model.png'
        # settings['plots']['triangle']['file'] = f'figs/{event}_triangle.png'
        # settings['fitting_parameters'] = {'n_walkers': 20, 'n_steps': 20000,
        #                                   'n_burn': 10000}

        # filename = f"W16_{event.replace('-','')}_parallax.yaml"
        # with open(f'{path}/yaml_files/{filename}', 'w') as file:
        #     yaml.dump(settings, file, sort_keys=False, indent=4)
        
        print(event, 'done!')

    return names, coords


if __name__ == '__main__':

    # BEFORE: run get_59_yaml_files just once, edit after that 
    path = os.path.dirname(os.path.realpath(__file__))

    with open('W16_template.yaml') as template_file_:
        template = template_file_.read()
    coord_w16 = Table.read('W16-coordinates.txt', format='ascii')
    all_settings = ascii.read("W16_all_settings.txt")

    get_59_yaml_files(path, template, coord_w16, all_settings)
    
    # breakpoint()

    # AFTER: leave it to another script... run_59_events.py!
    # all_events = sorted(os.listdir(f'{path}/yaml_files/'))
    # for item in all_events:
    #     if 'bkpi' in item:  # not in
    #         with open(f'{path}/yaml_files/{item}', 'r') as data:
    #             settings = yaml.safe_load(data)
            
    #         print(f'\n- Starting fit for \033[1m{item}\033[0m')
    #         settings['fitting_parameters']['n_walkers'] = 10
    #         settings['fitting_parameters']['n_steps'] = 10000
    #         settings['fitting_parameters']['n_burn'] = 5000
    #         ulens_model_fit = UlensModelFit(**settings)
    #         ulens_model_fit.run_fit()
    #         # test = ulens_model_fit._parse_results()
    #         print('----------------------------------------------')
    #         breakpoint()
