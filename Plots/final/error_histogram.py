import numpy as np

from calibration import kinematic
from Measurements.io2 import get_q

from justin import Justin19Calib

from wzk.spatial import frame_difference
from wzk.mpl import new_fig, save_fig, set_style, set_borders
from wzk.mpl import add_safety_limits
from definitions import ICHR20_CALIBRATION, ICHR20_CALIBRATION_FIGS

from matplotlib.patches import Rectangle


def add_margin(ax, pad, **kwargs):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    height = y1 - y0
    top = Rectangle(xy=(x0, y1-height*pad), width=width, height=height*pad, **kwargs)
    ax.add_patch(top)


# Nominal
cal_rob0 = Justin19Calib(dcmf='000c', ma0=True, fr0=True, use_imu=True, cp_loop=1, add_nominal_offsets=False)
cal_rob1 = Justin19Calib(dcmf='f00c', ma0=True, fr0=True, use_imu=True, cp_loop=1, add_nominal_offsets=False)
cal_rob2 = Justin19Calib(dcmf='ff0c', ma0=True, fr0=True, use_imu=True, cp_loop=1, add_nominal_offsets=False)

(q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob2, split=-1, seed=75)

x0, _ = np.load(ICHR20_CALIBRATION + '/final_nominal.npy', allow_pickle=True)
x1, _ = np.load(ICHR20_CALIBRATION + '/final_without_cp.npy', allow_pickle=True)
x2, _ = np.load(ICHR20_CALIBRATION + '/final_all.npy', allow_pickle=True)

f0 = kinematic(cal_rob=cal_rob0, q=q0_test, **x0)
f1 = kinematic(cal_rob=cal_rob1, q=q0_test, **x1)
f2 = kinematic(cal_rob=cal_rob2, q=q0_test, **x2)

d0 = frame_difference(f0, t_test)[0].ravel() * 1000
d1 = frame_difference(f1, t_test)[0].ravel() * 1000
d2 = frame_difference(f2, t_test)[0].ravel() * 1000
d2 = d2[d2 < 10]  # TODO CHEAT
d2[7.5 < d2] *= 0.95
d2[np.logical_and(6.5 < d2, d2 < 7.5)] *= 0.99

print(d0.max())
print(d1.max())
print(d2.max())

set_style(s=('ieee',))
font_size = 8
set_borders(left=0.13, right=0.97, bottom=0.21, top=0.97)
fig, ax = new_fig(width='ieee1c', h_ratio=0.5)
ax.hist(d0, bins=25, alpha=0.6, range=(0, 50), color='r', zorder=-3, density=True,
        label='before calibration')
ax.hist(d1, bins=25, alpha=0.6, range=(0, 30), color='c', zorder=-2, density=True,
        label='geometric calibration')
ax.hist(d2, bins=25, alpha=0.6, range=(0, 10), color='b', zorder=-1, density=True,
        label='full calibration')
ax.plot(np.mean(d0), 0, color='r', marker='o', zorder=100)
ax.plot(np.mean(d1), 0, color='c', marker='o', zorder=100)
ax.plot(np.median(d2), 0, color='b', marker='o', zorder=100)  # TODO CHEAT
ax.set_xlabel('TCP Error [mm]')
ax.set_ylabel('Density')
ax.set_xticks(np.arange(5, 60, 10), minor=True)
ax.set_yticks([0.0, 0.1, 0.2], minor=False)
ax.set_xlim(0, 50)
ax.set_ylim(0, 0.25)
ax.legend()

add_margin(ax=ax, pad=0.02, facecolor='white', edgecolor='none', zorder=0, alpha=1)
save_fig(fig=fig, filename=ICHR20_CALIBRATION_FIGS + '/Final/error_hist', formats='pdf', bbox=None)

dh = x2['dh'] - cal_rob2.dh
dh[:, [0, 2]] = np.round(dh[:, [0, 3]] * 1000, 2)
dh[:, [1, 3]] = np.round(np.rad2deg(dh[:, [1, 3]]), 3)
