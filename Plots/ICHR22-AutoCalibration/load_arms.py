import numpy as np

from wzk.mpl import new_fig, set_style, set_borders, save_fig

from rokin.Robots import Justin19
from rokin.Vis.robot_3d import animate_path
from rokin.Vis.configurations import plot_q_distribution, plot_q_configurations

from rocal.definitions import ICHR22_AUTOCALIBRATION


def load_arm(file):
    a = np.loadtxt(file)

    i = a[:, 0]
    m = a[:, 1:5]
    f_idx = a[:, 5]
    f = a[:, 6:20]
    q = a[:, 20:]

    return q


file_right = '/Users/jote/Documents/Code/Python/src/rocal/Plots/ICHR22-AutoCalibration/right_arm.txt'
file_left = '/Users/jote/Documents/Code/Python/src/rocal/Plots/ICHR22-AutoCalibration/left_arm.txt'

q_r = load_arm(file_right)
q_l = load_arm(file_left)

robot = Justin19()
# plot_q_distribution(q_r[:, 3:10], limits=robot.limits[3:10, :])
# plot_q_distribution(q_l[:, 10:17], limits=robot.limits[10:17, :])

# animate_path(q=q, robot=)

set_borders(left=0.13, right=0.97, bottom=0.25, top=0.95)
set_style(s=('ieee'))
fig, ax = new_fig(width='ieee1c', height=1.5)
ax = plot_q_configurations(ax=ax, q=q_r[:, 3:10], limits=robot.limits[3:10, :], marker='o', color='red', zorder=10)
# plot_q_configurations(q_l[:, 10:17], limits=robot.limits[10:17, :], marker='o')


mode = 'kinect-right'
q10000 = np.load(f"{ICHR22_AUTOCALIBRATION}/q10000_random_{mode}.npy")
i = np.random.choice(np.arange(len(q10000)), size=1000, replace=False)
q_new = q10000[i]

plot_q_configurations(ax=ax, q=q_new[:, 3:10], limits=robot.limits[3:10, :], marker='o', alpha=0.1, color='blue', zorder=1)

save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION}/{mode}_q_configurations", formats='pdf')