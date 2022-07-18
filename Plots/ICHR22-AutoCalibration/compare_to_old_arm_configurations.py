import numpy as np

from wzk.mpl import new_fig, set_style, set_borders, save_fig, remove_duplicate_labels

from rokin.Robots import Justin19
from rokin.Vis.robot_3d import animate_path
from rokin.Vis.configurations import plot_q_distribution, plot_q_configurations

from rocal.definitions import ICHR22_AUTOCALIBRATION, ICHR22_AUTOCALIBRATION_FIGS


def load_arm(file):
    a = np.loadtxt(file)

    i = a[:, 0]
    m = a[:, 1:5]
    f_idx = a[:, 5]
    f = a[:, 6:20]
    q = a[:, 20:]

    return q


file_right = f"{ICHR22_AUTOCALIBRATION}/Birbach/right_arm.txt"
file_left = f"{ICHR22_AUTOCALIBRATION}/Birbach/left_arm.txt"

q_r = load_arm(file_right)
q_l = load_arm(file_left)

robot = Justin19()
# plot_q_distribution(q_r[:, 3:10], limits=robot.limits[3:10, :])
# plot_q_distribution(q_l[:, 10:17], limits=robot.limits[10:17, :])

# animate_path(q=q, robot=)

set_borders(left=0.12, right=0.99, bottom=0.28, top=0.99)
set_style(s=('ieee',))
fig, ax = new_fig(width='ieee1c', height=1.3)
ax = plot_q_configurations(ax=ax, q=q_r[:, 3:10], limits=robot.limits[3:10, :], marker='o', color='red', zorder=2, label='taught poses')
# plot_q_configurations(q_l[:, 10:17], limits=robot.limits[10:17, :], marker='o')


mode = 'kinect-right'
n = 1000
q10000 = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/q10000_random_{mode}.npy")
i = np.random.choice(np.arange(len(q10000)), size=n, replace=False)
q_new = q10000[i]
q_new[:, 9] = robot.sample_q(n)[:, 9]
plot_q_configurations(ax=ax, q=q_new[:, 3:10], limits=robot.limits[3:10, :], marker='o', alpha=0.1, color='blue', zorder=1, label='rejection sampling')
ax.legend()
remove_duplicate_labels(ax)
save_fig(fig=fig, file=f"{ICHR22_AUTOCALIBRATION_FIGS}/{mode}_q_configurations", formats='pdf', bbox=None)
