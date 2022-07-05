import numpy as np
from wzk.mpl import new_fig, set_style, save_fig, set_ticks_and_labels, set_borders, colors2

from rocal.Tools import KINECT, MARKER_POLE, MARKER_LEFT, MARKER_RIGHT
from rocal.definitions import ICHR22_AUTOCALIBRATION


def plot_u4(ax, camera, ub4, n=None):

    u0, u4, b4 = ub4['u0'], ub4['u4'], ub4['b4']
    if n is not None:
        u0, u4, b4 = u0[:n], u4[:n], b4[:n]

    x_px, y_px = camera.resolution

    # Switch x and y
    u4 = u4[:, :, ::-1]
    u0 = u0[:, ::-1]
    ax.set_xlim(0, y_px)
    ax.set_ylim(0, x_px)
    ax.plot(*u0.T, ls='', marker='x', color='k', zorder=10)
    color_list = colors2.tum_mix5
    # color_list = ['red', 'green', 'blue', 'yellow']
    for i in range(0, 4):
        ax.plot(*u4[b4[:, i], i, :].T, ls='', marker='o', color=color_list[i], alpha=0.1)


# Legend
font_size = 8
ha = 'left'
va = 'center'

# Main
set_style(s=('ieee',))
set_borders(left=0.15, right=0.95, bottom=0.175, top=0.95)

fig, ax = new_fig(n_cols=3, share_y=True, aspect=1, width='ieee1c')

ub4_left = np.load(f"{ICHR22_AUTOCALIBRATION}/{repr(MARKER_LEFT)}/ub4.npy", allow_pickle=True).item()
ub4_pole = np.load(f"{ICHR22_AUTOCALIBRATION}/{repr(MARKER_POLE)}/ub4.npy", allow_pickle=True).item()
ub4_right = np.load(f"{ICHR22_AUTOCALIBRATION}/{repr(MARKER_RIGHT)}/ub4.npy", allow_pickle=True).item()

n = 10000
plot_u4(ax=ax[0], camera=KINECT, ub4=ub4_left, n=n)
plot_u4(ax=ax[1], camera=KINECT, ub4=ub4_pole, n=n)
plot_u4(ax=ax[2], camera=KINECT, ub4=ub4_right, n=n)

set_ticks_and_labels(ax=ax[0], ticks=[0, 240, 480], axis='x')
set_ticks_and_labels(ax=ax[1], ticks=[0, 240, 480], axis='x')
set_ticks_and_labels(ax=ax[2], ticks=[0, 240, 480], axis='x')
set_ticks_and_labels(ax=ax[2], ticks=[0, 160, 320, 480, 640], axis='y')
ax[1].set_xlabel('y [px]')
ax[0].set_ylabel('x [px]')

save_fig(f"{ICHR22_AUTOCALIBRATION}/plots/ub4", formats='pdf')
