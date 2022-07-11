import numpy as np
from wzk import spatial, train_test_split

from rocal.calibration import calibrate
from rocal.Measurements.io2 import get_q
from rocal.Measurements.from_ardx_packets import get_qt_vicon

from rocal.Robots import Justin19CalVicon
from rocal.Vis.plotting import print_stats2
from rocal.parameter import Parameter, unwrap_x

from rocal.definitions import ICHR20_CALIBRATION, ICHR22_AUTOCALIBRATION

# FINDING 10000 test configurations is easily enough to have a small variance


def search_world_frame(q, t):
    """
    its hard / impossible to find the correct world frame if the initial guess is far off
       -> try multi start and safe the good fits
    """
    cal_rob = Justin19CalVicon(dkmca='000c0', add_nominal_offsets=True, use_imu=False, el_loop=1)
    cal_par = Parameter(x_weighting=0)

    n = 100
    threshold = 0.1  # m
    for i in range(n):
        _x, _stats = calibrate(q_cal=q, t_cal=t, q_test=q, t_test=t, x0_noise=10., verbose=1,
                               cal_rob=cal_rob, cal_par=cal_par)
        if _stats[0, 0, 0] < threshold:
            print(_stats[:, 0, 0].mean(axis=0))
            print(spatial.trans_rotvec2frame(trans=_x[:3], rotvec=_x[3:6]))
            return _x, _stats


if __name__ == '__main__':

    cal_par = Parameter(x_weighting=0)
    cal_rob = Justin19CalVicon(dkmca='ccfc0', add_nominal_offsets=True, use_imu=False, el_loop=1)

    # directory = ICHR20_CALIBRATION + '/Measurements/600'
    # (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=-1, seed=75)

    file = f"{ICHR22_AUTOCALIBRATION}/Vicon/random_poses_smooth_100-1657536656-measurements.npy"
    q, t = get_qt_vicon(file=file)
    i = np.array([69,  70,  45,  84,  65,  28,  98,  51,  85, 100])
    q, t = np.delete(q, i, axis=0), np.delete(t, i, axis=0)
    (q0_cal, t_cal), (q0_test, t_test) = train_test_split(q, t, split=-1, shuffle=False)

    from wzk import tic, toc

    tic()
    x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=4,
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    toc()

    # search_world_frame(q, t)
    # x, stats = calibrate(q_cal=q0_cal, t_cal=q_cal, q_test=q0_test, t_test=q_cal, verbose=1,
    #                      obj_fun='joints',
    #                      cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    print_stats2(stats)
    # x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True )

    # print(x['cm'])

    # from rokin.Robots import Justin19
    # robot = Justin19()
    # q = robot.sample_q(100)
    # f0 = robot.get_frames_dh(q=q, dh=dh0)[:, 13:14, :, :]
    # f2 = robot.get_frames_dh(q=q, dh=dh2)[:, 13:14, :, :]
    #
    # from rocal.Vis.plotting import plot_frame_difference
    # plot_frame_difference(f0, f2, verbose=3)
    #
    # print(pd.DataFrame(np.round(np.abs(d).max(axis=1), 3)))

    # stats = np.rad2deg(np.sqrt(stats))
    # from wzk.mpl import new_fig
    #
    # fig, ax = new_fig(n_cols=3, title='torso')
    # for i in range(3):
    #     ax[i].set_xlabel('|q_c - q_m| [Degree]')
    #     ax[i].hist(np.rad2deg(np.abs(q0_cal - q_cal))[:, i], color='red', alpha=0.3)
    #     ax[i].hist(stats[:, i], color='blue', alpha=0.3)
    #
    # fig, ax = new_fig(n_cols=7, title='right')
    # for i in range(3, 10):
    #     ax[i-3].set_xlabel('|q_c - q_m| [Degree]')
    #     ax[i-3].hist(np.rad2deg(np.abs(q0_cal - q_cal))[:, i], color='red', alpha=0.3)
    #     ax[i-3].hist(stats[:, i], color='blue', alpha=0.3)
