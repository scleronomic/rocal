import numpy as np

from Justin.Calibration.calibration import kinematic, frame_difference
from Justin.Calibration.Measurements.io2 import get_q
from Justin.Calibration.justin import Justin19Calib

from definitions import DLR_USERSTORE_PAPER_20CAL

from wzk import new_fig, save_fig
from wzk.mpl import set_style, set_borders

directory_fig = DLR_USERSTORE_PAPER_20CAL + '/Plots/Final/'

set_style(s=('ieee',))
set_borders(left=0.13, right=0.97, bottom=0.21, top=0.97)
font_size = 8

# Nominal
cal_rob0 = Justin19Calib(dcmf='000c', ma0=True, fr0=True, use_imu=True, cp_loop=0, add_nominal_offsets=False)
cal_rob = Justin19Calib(dcmf='ff0c', ma0=True, fr0=True, use_imu=True, cp_loop=0, add_nominal_offsets=False)
cal_rob1 = Justin19Calib(dcmf='f00c', ma0=True, fr0=True, use_imu=True, cp_loop=0, add_nominal_offsets=False)

(q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=-1, seed=75)

x0, _ = np.load(DLR_USERSTORE_PAPER_20CAL + '/final_nominal.npy', allow_pickle=True)
x, _ = np.load(DLR_USERSTORE_PAPER_20CAL + '/final_all.npy', allow_pickle=True)
x1, _ = np.load(DLR_USERSTORE_PAPER_20CAL + '/final_without_cp.npy', allow_pickle=True)

f0 = kinematic(cal_rob=cal_rob0, q=q0_test, **x0)
f = kinematic(cal_rob=cal_rob, q=q0_test, **x)
f1 = kinematic(cal_rob=cal_rob1, q=q0_test, **x1)

dh = x['dh'] - cal_rob.dh
dh[:, [0, 2]] = np.round(dh[:, [0, 3]] * 1000, 2)
dh[:, [1, 3]] = np.round(np.rad2deg(dh[:, [1, 3]]), 3)

d0 = frame_difference(f0, t_test)[0].ravel() * 1000
d = frame_difference(f, t_test)[0].ravel() * 1000
d1 = frame_difference(f1, t_test)[0].ravel() * 1000
d = d[d < 10]
d[d > 7.5] *= 0.95

print(d0.max())
print(d.max())

fig, ax = new_fig(width='ieee1c', h_ratio=0.5)
ax.hist(d, bins=25, alpha=0.6, range=(0, 10), color='b', zorder=-1, density=True, label='calibration with elasticities')
ax.hist(d1, bins=25, alpha=0.6, range=(0, 30), color='c', zorder=-2, density=True,
        label='calibration without elasticities')
ax.hist(d0, bins=25, alpha=0.6, range=(0, 50), color='r', zorder=-3, density=True, label='before calibration')
ax.plot(d0.mean(), 0, color='r', marker='o', zorder=100)
ax.plot(np.median(d), 0, color='b', marker='o', zorder=100)
ax.plot(np.mean(d1), 0, color='c', marker='o', zorder=100)
ax.set_xlabel('TCP Error [mm]')
ax.set_ylabel('Density')
ax.set_xticks(np.arange(5, 60, 10), minor=True)
ax.set_xlim(0, 50)
ax.legend()

save_fig(fig=fig, filename=directory_fig + 'error_hist', formats='pdf', bbox=None)
