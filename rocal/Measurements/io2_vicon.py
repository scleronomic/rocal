import numpy as np

from wzk.numpy2 import object2numeric_array
from wzk import print_dict
from mopla.Planner.ardx2 import ardx, pkt2dict

from rocal.Measurements.from_ardx_packets import get_q_list
from rocal.definitions import ICHR22_AUTOCALIBRATION

file = f"{ICHR22_AUTOCALIBRATION}/Vicon/random_poses_smooth_100-1657536656-measurements.npy"


if __name__ == '__main__':
    pass
    from rokin.Robots import Justin19
    from rocal.Robots import Justin19CalVicon
    robot = Justin19()
    robot_cal = Justin19CalVicon(dkmca='00000')

    t2 = robot.get_frames(q)[:, [13, 22]]

    t2 = robot_cal.cm[0] @ t2 @ robot_cal.cm[1:]

    from rocal.Vis.plotting import plot_frame_difference

    plot_frame_difference(t0, t2, verbose=10)

    from wzk.mpl import plot_projections_2d
    ax = plot_projections_2d(x=t0[:, 0, :-1, -1], color='red', marker='o', ls='')
    ax = plot_projections_2d(ax=ax, x=t0[:, 1, :-1, -1], color='blue', marker='x', ls='')
