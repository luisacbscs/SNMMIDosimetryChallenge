import os
import itk
from utilities import get_dose_rate
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt


path = "C:\\Users\\Luisa\\Documents\\SNMMIDosimetryChallenge\\clean\\patient_6"

dr_dict = {
    "ID": path.split(os.sep)[-1]
}

timepoints_file = os.path.join(path, "timepoints.json")

with open(timepoints_file, "r") as f:
    timepoints = json.load(f)

injection_time = datetime.datetime(int(timepoints['injection']['date'][:4]),    # date: YYYYMMDD
                                   int(timepoints['injection']['date'][4:6]),
                                   int(timepoints['injection']['date'][6:]),
                                   int(timepoints['injection']['time'][:2]),    # time: HHMMSS
                                   int(timepoints['injection']['time'][2:4]),
                                   int(timepoints['injection']['time'][4:]))

counter = 0
N_labels = None
for scan in sorted(os.listdir(path)):
    scan_dict = {}
    if os.path.isdir(os.path.join(path, scan)):
        counter += 1

        scan_time = datetime.datetime(int(timepoints[scan]['date'][:4]),    # date: YYYYMMDD
                                      int(timepoints[scan]['date'][4:6]),
                                      int(timepoints[scan]['date'][6:]),
                                      int(timepoints[scan]['time'][:2]),    # time: HHMMSS
                                      int(timepoints[scan]['time'][2:4]),
                                      int(timepoints[scan]['time'][4:]))

        t = (scan_time - injection_time).total_seconds()

        scan_dict = {
            "t": t
        }

        spect_path = None
        org_path = None
        spect = None
        msk = None
        for im in os.listdir(os.path.join(path, scan)):
            if 'spect' in im:
                spect_path = os.path.join(path, scan, im)
                spect = itk.imread(spect_path)
            elif 'organs' in im:
                org_path = os.path.join(path, scan, im)
                msk = itk.imread(org_path)

        N_labels = int(np.max(np.asarray(msk)))
        label_dict = {}
        for label in range(1, N_labels+1):
            dr = get_dose_rate(spect, msk, label, return_dose_map=False)
            print(scan, '\t',  spect_path.split('\\')[-1], '\t', str(datetime.timedelta(seconds=t)), '\t', label, '\t',
                  np.round(dr, decimals=5), "mGy/s")
            label_dict[label] = dr

        scan_dict["D"] = label_dict

        dr_dict[f"scan{counter}"] = scan_dict

dr_plot_file = os.path.join(path, f"dose_rate_plot_points.json")

open(dr_plot_file, 'w+').close()
with open(dr_plot_file, 'a+') as f:
    json.dump(dr_dict, f, indent=4)

n_rows = 1 if N_labels/4 == 1 else int(np.ceil(N_labels/4))
n_cols = N_labels if N_labels < 4 else 4

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4, n_rows * 4))
ax = ax.flat

fig.suptitle(f"Mean Dose Rates for {path.split(os.sep)[-1]}", fontweight="bold")

t_lst = [dr_dict[k]["t"] for k in dr_dict.keys() if "scan" in k]

for i in range(N_labels):
    D_lst = [dr_dict[k]["D"][i+1] for k in dr_dict.keys() if "scan" in k]

    ax[i].plot(np.asarray(t_lst)/(60*60), D_lst, marker='o', linestyle='None', color='#C39BD3')
    ax[i].set_title(f"Label {i+1}")
    ax[i].set_xlabel("Time (h)")
    ax[i].set_ylabel("Mean Absorbed Dose Rate (mGy/s)")
    ax[i].set_xlim(0)
    ax[i].set_axisbelow(True)
    ax[i].yaxis.grid(color='lightgrey')
    ax[i].xaxis.grid(color='lightgrey')

fig.tight_layout()

plt.show()
