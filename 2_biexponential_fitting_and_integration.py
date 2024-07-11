import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate


path = "C:\\Users\\Luisa\\Documents\\SNMMIDosimetryChallenge\\clean\\patient_6"
dr_plot_file = os.path.join(path, "dose_rate_plot_points.json")


def biexponential_decay_177Lu(t, k1, k2, lambda2):

    lutetium177_halflife = 6.6443 * 24 * 60 * 60  # 177Lu half-life in seconds

    lambda1 = - (np.log(0.5) * 1/lutetium177_halflife)

    return k1 * np.exp(-lambda1 * t) + k2 * np.exp(-lambda2 * t)


def linear(x, a, b):
    return a * x + b


with open(dr_plot_file, "r") as f:
    dr_points = json.load(f)

N_labels = len(dr_points["scan1"]["D"])

n_rows = 1 if N_labels/2 == 1 else int(np.ceil(N_labels/2))
n_cols = N_labels if N_labels < 2 else 2

del_axes = 0 if N_labels == n_cols * n_rows else n_cols * n_rows - N_labels

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 8, n_rows * 4))
ax = ax.flat

fig.suptitle(f"Mean Dose Rates for {path.split(os.sep)[-1]}", fontweight="bold")

t = [dr_points[k]["t"] for k in dr_points.keys() if "scan" in k]

for i in range(N_labels):

    D = [dr_points[k]["D"][str(i+1)] for k in dr_points.keys() if "scan" in k]

    # Initialising parameters k1, k2, lambda2 - from which Python then iterates over to find the best fit
    k1 = (np.max(D) - np.min(D)) / 2
    k2 = k1
    lambda2 = 0.0

    fit = curve_fit(biexponential_decay_177Lu, xdata=t, ydata=D, p0=[k1, k2, lambda2], bounds=(0, np.inf),
                    full_output=True, maxfev=5000)
    params = fit[0]

    # Integration: total dose calculation
    # From t0 to t1 (injection to first SPECT acquisition)
    total_dose_until_t1 = (t[0] * biexponential_decay_177Lu(t[0], *params))/2
    # From t1 to 10 177Lu half-lives
    total_dose_from_t1 = integrate.quad(lambda x: biexponential_decay_177Lu(x, *params), t[0], 6.6443 * 24 * 60 * 60 * 10)[0]
    total_dose = total_dose_until_t1 + total_dose_from_t1
    print(f"Mean absorbed dose (label {i+1}):", np.round(total_dose * 0.001, decimals=2), "Gy")

    x = np.linspace(t[0], max(t) * 5, 500)
    y = biexponential_decay_177Lu(x, *params)
    ax[i].plot(x/(60*60), y * 0.001, '--', color='#424949')

    x = np.linspace(0, t[0], 50)
    y = linear(x, biexponential_decay_177Lu(t[0], *params)/t[0], 0)
    ax[i].plot(x/(60*60), y * 0.001, '--', color='#424949')

    ax[i].plot(np.asarray(t)/(60*60), np.asarray(D) * 0.001, 'o', color='#C39BD3')

    ax[i].set_title(f"Label {i+1}")
    ax[i].set_xlabel("Time (h)")
    ax[i].set_ylabel("Mean Absorbed Dose Rate (Gy/s)")
    ax[i].set_xlim(0, max(t) * 5 / (60 * 60))
    ax[i].set_ylim(0)

    txt = ax[i].text((t[2]-t[1])/(2 * 60 * 60), np.max(D) * 0.20 * 0.001,
                     f"AUC={np.round(total_dose * 0.001, decimals=2)}",
                     bbox=dict(boxstyle="round", ec="black", fc="white"))

    #plt.grid('#F2F4F4')
    ax[i].set_axisbelow(True)
    ax[i].yaxis.grid(color='lightgrey')
    ax[i].xaxis.grid(color='lightgrey')

if del_axes != 0:
    for d in range(1, del_axes + 1):
        fig.delaxes(ax[-d])

fig.tight_layout()
plt.show()
