import os
import sys
import yaml

from astropy.table import Table


path = os.path.dirname(os.path.abspath(__file__))
project_name, phot_name, path_output = sys.argv[1:]
phot_files = os.listdir(os.path.join(path, project_name, 'phot', phot_name))
phot_files = [f.split('_OGLE.dat')[0] for f in phot_files if 'dat' in f]
all_datasets = ["BLG50X1", "BLG50X2", "BLG50X3", "lcmin3to5", "lcall1",
                "lcall2", "lcall3"]

with open("plot_multiple_fits_template.yaml", "r", encoding='utf-8') as stream:
    template = stream.read()

for event_id in sorted(phot_files):

    for dataset in all_datasets:
        t_peak = os.path.join(path, project_name, f"t_peak_{dataset}.dat")
        t_peak = Table.read(t_peak, format='ascii')
        if event_id in t_peak['obj_id']:
            break

    fname_1l2s = os.path.join(path, project_name, 'results_1L2S', dataset,
                              "yaml_results", f"{event_id}_OGLE_results.yaml")
    with open(fname_1l2s, 'r') as stream:
        res = yaml.safe_load(stream)["Best model"]
    chi2_1l2s, dof = res['chi2'], res['dof']
    params = [str(p) for p in res['Parameters'].values()]
    fluxes = [f for f in res['Fluxes'].values()]

    fname_2l2s = os.path.join(path, project_name, 'results_2L1S', dataset,
                              f"{event_id}_OGLE_2L1S_all_results_between.yaml")
    with open(fname_2l2s, 'r') as stream:
        res = yaml.safe_load(stream)["Best model"]
    with open(fname_2l2s.replace('between', 'beyond'), 'r') as stream:
        res_2 = yaml.safe_load(stream)["Best model"]
        res = res if res['chi2'] < res_2['chi2'] else res_2
    chi2_2l1s = res['chi2']
    params_2 = [str(p) for p in res['Parameters'].values()]
    fluxes_2 = [f for f in res['Fluxes'].values()]

    yaml_2l1S = os.path.join(path, project_name, 'yaml_files_2L1S', dataset,
                             f"{event_id}_OGLE-2L1S_traj_between.yaml")
    with open(yaml_2l1S, 'r') as stream:
        methods = yaml.safe_load(stream)["model"]["methods"]
    x_lim = methods.split()[0] + " " + methods.split()[-1]
    chi2_both = f"{chi2_1l2s:.4f} {chi2_2l1s:.4f}"

    lst = [project_name, phot_name, event_id, " ".join(params), fluxes[:2],
           fluxes[-1], " ".join(params_2), fluxes_2[:1], fluxes_2[-1],
           methods, path_output, x_lim, chi2_both]
    out_path = os.path.join(path, project_name, path_output,
                            f"{event_id}_multiple_fits.yaml")
    with open(out_path, "w", encoding='utf-8') as stream:
        stream.write(template.format(*lst))
