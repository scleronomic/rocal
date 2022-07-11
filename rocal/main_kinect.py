
import numpy as np

from wzk import new_fig, save_fig
from wzk import round_dict, print_dict

from rokin.Robots import Justin19

from rocal.Robots import Justin19CalKinect
from rocal.parameter import Parameter, unwrap_x
from rocal.Measurements.from_ardx_packets import get_qt_kinect
from rocal.calibration import calibrate, create_wrapper_kinematic
from rocal.definitions import ICHR22_AUTOCALIBRATION, ICHR22_AUTOCALIBRATION_FIGS


# When using the values for the vicon calibration [corresponding to an error of 4mm] we get roughly a pixel error of 2.
# But if we calibrate the pixel model [corresponding to an error of 0.8 pixel] we get a tracker error of 50mm.


def get_qt(n, mode):

    file_pole = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-pole-1657128676-measurements.npy", allow_pickle=True)
    file_right = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-right-measurements.npy", allow_pickle=True)
    file_left = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-left-measurements.npy", allow_pickle=True)

    q_pole, t_pole = get_qt_kinect(d=file_pole, mode=mode)
    q_right, t_right = get_qt_kinect(d=file_right, mode=mode)
    q_left, t_left = get_qt_kinect(d=file_left, mode=mode)

    q_pole, t_pole = q_pole[:n], t_pole[:n]
    q_right, t_right = q_right[:n], t_right[:n]
    q_left, t_left = q_left[:n], t_left[:n]

    i_delete_pole = np.array([28,  29,  61,  92,  93,  94,  95,  96,  98, 147])
    i_delete_right = np.array([84,  85,  86,  88,  89,  90,  91, 105, 106, 132])
    i_delete_left = np.array([14,  15,  35,  59,  74, 123, 158, 159, 181, 182])

    q_pole, t_pole = np.delete(q_pole, i_delete_pole, axis=0), np.delete(t_pole, i_delete_pole, axis=0)
    q_right, t_right = np.delete(q_right, i_delete_right, axis=0), np.delete(t_right, i_delete_right, axis=0)
    q_left, t_left = np.delete(q_left, i_delete_left, axis=0), np.delete(t_left, i_delete_left, axis=0)

    q = np.concatenate((q_pole, q_right, q_left), axis=0)
    l = np.cumsum([0, len(q_pole), len(q_right), len(q_left)])

    t = np.zeros((len(q), 3, 2))
    t[l[0]:l[1], 0, ...] = t_pole
    t[l[1]:l[2], 1, ...] = t_right
    t[l[2]:l[3], 2, ...] = t_left

    return q, t, l


def plot_pixel_error(dkmca, stats, l):
    color_pole = 'cyan'
    color_left = 'blue'
    color_right = 'red'
    marker_pole = '^'
    marker_right = 's'
    marker_left = 'o'

    fig, ax = new_fig(aspect=1)
    ax.plot(*stats[l[0]:l[1]].T, color=color_pole, marker=marker_pole, label='pole', ls='')
    ax.plot(*stats[l[1]:l[2]].T, color=color_right, marker=marker_right, label='right', ls='')
    ax.plot(*stats[l[2]:l[3]].T, color=color_left, marker=marker_left, label='left', ls='')
    ax.legend()

    mse = np.linalg.norm(stats, axis=-1)
    mse_pole = mse[l[0]:l[1]]
    mse_right = mse[l[1]:l[2]]
    mse_left = mse[l[2]:l[3]]

    str_pole = f"pole:  {np.round(mse_pole.mean(), 3)}"
    str_right = f"right: {np.round(mse_right.mean(), 3)}"
    str_left = f"left:  {np.round(mse_left.mean(), 3)}"
    print(str_pole)
    print(str_right)
    print(str_left)

    fig, ax = new_fig(title=f'Calibration Mode: {dkmca}')
    ax.hist(mse_pole, color=color_pole, range=(0, 15), bins=50, alpha=0.5, label=str_pole)
    ax.hist(mse_right, color=color_right, range=(0, 15), bins=50, alpha=0.5, label=str_right)
    ax.hist(mse_left, color=color_left, range=(0, 15), bins=50, alpha=0.5, label=str_left)
    ax.set_xlabel('|pixel difference|')
    ax.legend()
    save_fig(fig=fig, file=f'{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/pixel_error_{dkmca}', formats='pdf')
    np.save(f"{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/kinect_pixel-error_{dkmca}.npy", dict(dp=stats, l=l))

    perc = 95
    i_delete_pole = np.nonzero(mse_pole > np.percentile(mse_pole, perc))[0]
    i_delete_right = np.nonzero(mse_right > np.percentile(mse_right, perc))[0]
    i_delete_left = np.nonzero(mse_left > np.percentile(mse_left, perc))[0]
    print(f'i_delete_pole = np.{repr(i_delete_pole)}')
    print(f'i_delete_right = np.{repr(i_delete_right)}')
    print(f'i_delete_left = np.{repr(i_delete_left)}')


def test_vicon(dkmca, x):
    from rocal.Measurements.io2 import get_q
    cm_vicon = np.array([[[0.02294921, -0.9997311, -0.00332708, -2.55103349],
                          [0.99961656, 0.02289475, 0.01557431, 1.99076737],
                          [-0.01549395, -0.00368323, 0.99987318, 0.1061276],
                          [0., 0., 0., 1.]],
                         [[0., 0., -1., -0.11154556],
                          [0., -1., 0., -0.02077577],
                          [-1., 0., 0., 0.08216418],
                          [0., 0., 0., 1.]],
                         [[0., 0., -1., -0.10078115],
                          [-1., 0., 0., 0.02026033],
                          [0., 1., 0., 0.05189488],
                          [0., 0., 0., 1.]]])

    # cm_vicon = np.array([[[ 0.02322987, -0.99972573, -0.00297448, -2.55079078],
    #                       [ 0.99959207,  0.02317711,  0.0166887 ,  1.98926354],
    #                       [-0.01661518, -0.00336094,  0.99985631,  0.1020042 ],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                      [[ 0.        ,  0.        , -1.        , -0.09915985],
    #                       [ 0.        , -1.        ,  0.        , -0.02133832],
    #                       [-1.        ,  0.        ,  0.        ,  0.08904842],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                      [[ 0.        ,  0.        , -1.        , -0.10507792],
    #                       [-1.        ,  0.        ,  0.        ,  0.02153678],
    #                       [ 0.        ,  1.        ,  0.        ,  0.05301567],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]]])
    cal_rob = Justin19CalKinect(dkmca=dkmca, add_nominal_offsets=True, use_imu=False, el_loop=1)
    (q0_cal, q_cal, t_cal), _ = get_q(cal_rob=cal_rob, split=-1, seed=75)
    get_frames = create_wrapper_kinematic(x=x, cal_rob=cal_rob)

    f = get_frames(q=q0_cal)[0]

    f_right = cm_vicon[0] @ f[:, 13, :, :] @ cm_vicon[1]
    f_left = cm_vicon[0] @ f[:, 22, :, :] @ cm_vicon[2]

    from rocal.Vis.plotting import plot_frame_difference
    print('Kinect Vs Vicon')
    plot_frame_difference(f0=t_cal,
                          f1=np.stack((f_right, f_left), axis=1),
                          verbose=2)

    robot = Justin19()
    f0 = robot.get_frames(q0_cal)[:, [13, 22]]
    f2 = get_frames(q0_cal)[0][:, [13, 22]]
    print('Kinect Vs Nominal')
    plot_frame_difference(f0=f0, f1=f2, verbose=1)

    print('Nominal vs Vicon')
    plot_frame_difference(f0=t_cal, f1=cm_vicon[0] @ f0 @ cm_vicon[[1, 2]], verbose=1)


def main():
    mode = 'commanded'  # 'commanded' or 'measured'
    dkmca = 'cc0c0'
    x_weighting = 10
    q, t, l = get_qt(n=200, mode=mode)

    cal_par = Parameter(x_weighting=x_weighting, t_weighting=1000000*np.array([1, 1, 1]))
    cal_rob = Justin19CalKinect(dkmca=dkmca, add_nominal_offsets=True, use_imu=False, el_loop=1)

    x, stats = calibrate(q_cal=q, t_cal=t, q_test=None, t_test=None, verbose=3, obj_fun='marker_image',
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0.0)

    x0 = x.copy()

    x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)

    x = round_dict(d=x, decimals=5)
    print_dict(x)

    dh0 = Justin19().dh
    dh2 = x['dh']
    d = dh0 - dh2
    print(np.abs(d).max(axis=0))
    import pandas as pd

    print('DH Difference:')
    print(pd.DataFrame(np.round(d, 3)))

    plot_pixel_error(dkmca=dkmca, stats=stats, l=l)
    test_vicon(dkmca=dkmca, x=x0)


if __name__ == '__main__':
    main()

# pass
# Full without mass
# pole:  0.831
# right: 0.686
# left:  0.525


# Fuller
# pole:  0.582
# right: 0.605
# left:  0.593


# print('Results:')
# cm = (x['cm'])
# print(cal_rob.cm[3] - cm[3])

# print('Marker Pole:')
# print(cm[0])
#
# print('Marker Right:')
# print(cm[1])
#
# print('Marker Left:')
# print(cm[2])
#
# print('Kinect:')
# print(cm[3])


# print(x)