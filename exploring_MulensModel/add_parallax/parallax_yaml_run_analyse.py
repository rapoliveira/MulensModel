"""
Write documentation later
"""
from astropy.io import ascii
from astropy.table import Table, Column
import inquirer as iq
import numpy as np
import os
import sys
import yaml

from ulens_model_fit import UlensModelFit


def get_yaml_files(path, files):
    '''
    Get the yaml files for all the 59 events. Input parameters are the path,
    and the files with template, coordinates and settings. 
    '''
    with open(files[0]) as template_file_:
        template = template_file_.read()
    coord_w16 = Table.read(files[1], format='ascii')
    names, coords = coord_w16['lens_name'], coord_w16['ra_j2000','dec_j2000']
    all_settings = ascii.read(files[2])
    n_emcee = [10, 10000, 5000]  # [20, 20000, 10000] or [10, 10000, 5000]
    
    for event, coord, line in zip(names, coords, all_settings):
        
        # new code: Radek, 16.aug.23 (old code is in bkpi folder)
        lst = np.array(list(line))[1:].tolist()
        lst.insert(4, f'{coord[0]} {coord[1]}')
        lst.insert(5, round(float(lst[1])))
        u0_range = (0, 3, 'u0-pos') if float(lst[2]) > 0 else (-3, 0, 'u0-neg')
        lst[6:6] = u0_range[:-1]
        lst[8:8] = n_emcee
        lst.insert(11, u0_range[-1])
        
        with open(line[0], 'w') as out_file:
            out_file.write(template.format(*lst))
        print(event, end=', ')
        
    print(event, 'done!\n')

    return names, coords

def run_events(path, events_to_run, u0):
    """
    Run MulensModel with parallax for the selection of events in events_to_run.
    The negative or positive value for u_0 is checked with the 3rd argument.
    """

    # TO-DO: empty the bkpi-folder and doing a backup of the previous results
    if len(os.listdir(f'{path}/chains/')) > 0:
        pass

    # run MulensModel in loop
    for item in events_to_run:
        with open(f'{path}/yaml_files/{item}', 'r') as data:
            settings = yaml.safe_load(data)

        init_u0 = float(settings['starting_parameters']['u_0'].split()[1])
        if (init_u0 > 0. and u0 == 'neg') or (init_u0 < 0. and u0 == 'pos'):
            raise ValueError(f'Incoherent initial u_0 value and input.')
        # settings['fitting_parameters']['n_walkers'] = 10  # quick-test
        # settings['fitting_parameters']['n_steps'] = 10000
        # settings['fitting_parameters']['n_burn'] = 5000
        print(f'\n- Starting fit for \033[1m{item}\033[0m\n')
        ulens_model_fit = UlensModelFit(**settings)
        ulens_model_fit.run_fit()
        print('----------------------------------------------')

    return ulens_model_fit

def analyse_results(path_pos_neg, pos_or_neg):
    """
    Read the outputs and generate a single table (u0 > 0, u0 < 0 or both)
    """

    # Initiating full table of results (with uncertainties)
    colnames = ['t_0', 't_0_+', 't_0_-', 'u_0', 'u_0_+', 'u_0_-', 't_E',
                't_E_+', 't_E_-', 'pi_E_N', 'pi_E_N_+', 'pi_E_N_-', 'pi_E_E',
                'pi_E_E_+', 'pi_E_E_-', 'flux_s_1', 'flux_b_1', 'ln_prob']
    tab = Table(np.ones(len(colnames)), names=colnames)
    tab.add_column(Column(['PAR-01'], name='event'), index=0)
    tab.remove_row(0)

    results_pos = sorted(os.listdir(f'{path_pos_neg[0]}/results'))
    results_neg = sorted(os.listdir(f'{path_pos_neg[1]}/results'))
    params = ['t_0', 'u_0', 't_E', 'pi_E_N', 'pi_E_E']
    
    # Get the result (from u0>0, <0 or best) and fill the table
    for (item_pos, item_neg) in zip(results_pos, results_neg):
        
        with open(f'{path_pos_neg[0]}/results/{item_pos}') as posit:
            res_pos = yaml.safe_load(posit)
        with open(f'{path_pos_neg[1]}/results/{item_neg}') as negat:
            res_neg = yaml.safe_load(negat)
        result = res_pos if pos_or_neg == 'pos' else res_neg
        if pos_or_neg == 'both':
            pos = res_pos['Best model']['chi2'] < res_neg['Best model']['chi2']
            result = res_pos if pos else res_neg

        event_id = item_pos.split('_')[0]
        chi2, params_best, fluxes = result['Best model'].values()
        params_q = result['Fitted parameters']
        values = []
        for par in params:
            values += [params_best[par], params_q[par][1], -params_q[par][2]]
        values += [fluxes['flux_s_1'], fluxes['flux_b_1'], chi2]
        values = [round(val,5) for val in values]
        tab.add_row([event_id] + values)
    
    # Writing tables with and without uncertainties
    folder = {'pos': path_pos_neg[0], 'neg': path_pos_neg[1],
              'both': os.path.dirname(path_pos_neg[0])}
    filename = f'{folder[pos_or_neg]}/results-parallax-{pos_or_neg}-full.txt'
    tab.write(filename, format='ascii', overwrite=True)
    tab2 = tab['event', 't_0', 'u_0', 't_E', 'pi_E_N', 'pi_E_E', 'flux_s_1',
               'flux_b_1', 'ln_prob']
    tab2.write(filename.replace('-full',''), format='ascii', overwrite=True)

    return tab

if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(__file__))
    paths = [f'{path}/2nd-results-u0-pos', f'{path}/2nd-results-u0-neg']
    opts1 = '(yaml, run, analyse or all)'
    opts2 = '(positive/pos or negative/neg or both)'
    print()
    
    # Validating the inputs (method and pos_or_neg)
    if len(sys.argv) != 3:
        raise ValueError(f'Exactly two arguments needed {opts1} + {opts2}.')
    elif sys.argv[1] not in ['yaml', 'run', 'analyse', 'all']:
        raise ValueError(f'Wrong argument: first {opts1}.')
    elif sys.argv[2].lower() not in ['positive','pos','negative','neg','both']:
        raise ValueError(f'Wrong argument: second {opts2}.')
    pos_or_neg = sys.argv[2] if len(sys.argv[2]) < 5 else sys.argv[2][:3]

    # Calling the get_yaml_files() function
    if sys.argv[1] in ['yaml', 'all']:
        if pos_or_neg.lower() == 'both':
            raise ValueError('Impossible to run the get_yaml_files() function'+
                             ' with both positive and negative u_0.')                     
        settings_file = f"W16_all_settings_u0-{pos_or_neg.lower()}.txt"
        files = ['W16_template.yaml', 'W16-coordinates.txt', settings_file]
        if not all([os.path.isfile(f) for f in files]):
            raise ValueError(f'Any of these files is unavailable: {files}.')
        names, coords = get_yaml_files(path, files)

    # Calling the run_events() function
    if sys.argv[1] in ['run', 'all']:
        if pos_or_neg.lower() == 'both':
            raise ValueError('Impossible to run the run_events() function' +
                             'with both positive and negative u_0.')
        all_events = sorted(os.listdir(f'{path}/yaml_files/'))
        all_events = [event for event in all_events if 'bkpi' not in event]
        to_run = all_events[:] # e.g. [X], [:X], [X:], [X:X], [:]
        if len(to_run) > 1:
            msg = f"MulensModel will be run from {to_run[0].split('_')[1]}" +\
                  f" to {to_run[-1].split('_')[1]} events (i.e. {len(to_run)}"+\
                  f" events). Confirm?"
            ans = iq.prompt({iq.Confirm("c", message=msg, default=False)})['c']
            if not ans: sys.exit()
        ulens = run_events(path, to_run, pos_or_neg.lower())
    
    # Calling the analyse_results() function
    if sys.argv[1] in ['analyse', 'all']:
        full_tab = analyse_results(paths, pos_or_neg.lower())
    