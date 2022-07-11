import numpy as np

from wzk.mpl import new_fig, set_style, set_borders, save_fig, set_ticks_position, set_labels_position, set_ticks_and_labels

from rokin.Robots import Justin19
from rokin.Vis.robot_3d import animate_path
from rokin.Vis.configurations import plot_q_distribution, plot_q_configurations

from rocal.definitions import ICHR22_AUTOCALIBRATION, ICHR22_AUTOCALIBRATION_FIGS


data_nominal = np.load(f"{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/kinect_pixel-error_000c0.npy", allow_pickle=True).item()
data_geometric = np.load(f"{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/kinect_pixel-error_c00c0.npy", allow_pickle=True).item()
data_nongeometric = np.load(f"{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/kinect_pixel-error_cc0c0.npy", allow_pickle=True).item()

dp_nominal, l = data_nominal['dp'], data_nominal['l']
dp_geometric, l = data_geometric['dp'], data_nominal['l']
dp_nongeometric, l = data_nongeometric['dp'], data_nominal['l']

mse_nominal = np.linalg.norm(dp_nominal, axis=-1)
mse_geometric = np.linalg.norm(dp_geometric, axis=-1)
mse_nongeometric = np.linalg.norm(dp_nongeometric, axis=-1)

set_borders(left=0.12, right=0.99, bottom=0.34, top=0.98)
set_style(s=('ieee',))
style = dict(alpha=0.6, bins=50, range=(0, 15), density=True)

fig, ax = new_fig(width='ieee1c', height=1.1)
ax.hist(mse_nominal, color='red',  label='nominal', zorder=-3, **style)
ax.hist(mse_geometric, color='cyan', label='geometric', zorder=-2,  **style)
ax.hist(mse_nongeometric, color='blue', label='non-geometric', zorder=-1,  **style)
ax.plot(np.mean(mse_nominal), 0, color='red', marker='o', zorder=100)
ax.plot(np.mean(mse_geometric), 0, color='cyan', marker='o', zorder=100)
ax.plot(np.mean(mse_nongeometric), 0, color='blue', marker='o', zorder=100)

ax.set_xlabel('Pixel Error of Marker [px]')
ax.set_ylabel('Density')
ax.legend()
ax.set_xlim(0, 15)

save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION_FIGS}/kinect_pixel-error_hist", formats='pdf')

color_pole = 'cyan'
color_left = 'blue'
color_right = 'red'
marker_pole = '^'
marker_right = 's'
marker_left = 'o'

set_borders(left=0.03, right=0.99, bottom=0.20, top=0.99, wspace=0.0, hspace=0.0)

fig, ax = new_fig(width='ieee1c', height=1.45, aspect=1, n_cols=3)
style = dict(alpha=1, markersize=2, ls='')
ax[0].plot(*dp_nongeometric[l[0]:l[1]].T, color=color_pole, marker=marker_pole, label='pole', **style)
ax[0].plot(*dp_nongeometric[l[1]:l[2]].T, color=color_right, marker=marker_right, label='right', **style)
ax[0].plot(*dp_nongeometric[l[2]:l[3]].T, color=color_left, marker=marker_left, label='left', **style)
set_ticks_position(ax=ax[0], position='all')
set_labels_position(ax=ax[0], position=('bottom', 'left'))
ax[0].set_xlim(-15, 15)
ax[0].set_ylim(-15, 15)
legend = ax[0].legend(handlelength=0.2, labelspacing=0.5, handletextpad=0.5, borderaxespad=0.1, loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [10]
legend.legendHandles[2]._sizes = [10]

ax[0].set_ylabel('Error - y [pixel]')
ax[1].set_xlabel('Error - x [pixel]')


ax[1].plot(*dp_geometric[l[0]:l[1]].T, color=color_pole, marker=marker_pole, label='pole', **style)
ax[1].plot(*dp_geometric[l[1]:l[2]].T, color=color_right, marker=marker_right, label='right', **style)
ax[1].plot(*dp_geometric[l[2]:l[3]].T, color=color_left, marker=marker_left, label='left', **style)
set_ticks_position(ax=ax[1], position='all')
set_labels_position(ax=ax[1], position='bottom')

ax[1].set_xlim(-15, 15)
ax[1].set_ylim(-15, 15)

ax[2].plot(*dp_nominal[l[0]:l[1]].T, color=color_pole, marker=marker_pole, label='pole', **style)
ax[2].plot(*dp_nominal[l[1]:l[2]].T, color=color_right, marker=marker_right, label='right', **style)
ax[2].plot(*dp_nominal[l[2]:l[3]].T, color=color_left, marker=marker_left, label='left', **style)
set_ticks_position(ax=ax[2], position='all')
set_labels_position(ax=ax[2], position='bottom')
ax[2].set_xlim(-15, 15)
ax[2].set_ylim(-15, 15)

y = 0.218
x = [0.219, 0.4807, 0.77777]
style = dict( ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', pad=1))
fig.text(s='non-geometric', x=x[0], y=y, **style)
fig.text(s='geometric', x=x[1], y=y, **style)
fig.text(s='nominal', x=x[2], y=y, **style)
save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION_FIGS}/kinect_pixel-error_scatter", formats='pdf')
