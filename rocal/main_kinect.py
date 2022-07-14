import numpy as np

from wzk import new_fig, save_fig
from wzk import round_dict, print_dict, object2numeric_array, train_test_split

from rokin.Robots import Justin19

from rocal import parameter, calibration
from rocal.Robots import Justin19CalKinect, Justin19CalVicon
from rocal.Measurements import from_ardx_packets, io2_kinect
from rocal.Vis.plotting import print_stats2

from rocal.definitions import ICHR22_AUTOCALIBRATION, ICHR22_AUTOCALIBRATION_FIGS


# When using the values for the vicon calibration [corresponding to an error of 4mm] we get roughly a pixel error of 2.
# But if we calibrate the pixel model [corresponding to an error of 0.8 pixel] we get a tracker error of 50mm.

# x_weighting too high works against a goof fit for the pixel error, but weighting off, does not lead to the correct minimum

# not only the x_weighting factor is important, but also which values are fixed and which are free
# to ensure that the model does not put parameters in strange places at least the ends and the long lengths must be fixed


def compare_image_modes():

    file = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-pole-1657128676-measurements.npy", allow_pickle=True)
    # file = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-right-1657126485-measurements.npy", allow_pickle=True)
    # file = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-left-1657121972-measurements.npy", allow_pickle=True)

    io2_kinect.plot_all_images(file)

    t_corrected = from_ardx_packets.get_marker_list(d=file, mode='corrected')
    b_corrected = np.array([False if ti is False else True for ti in t_corrected])
    t_corrected2 = np.zeros((len(t_corrected), 2))
    t_corrected2[b_corrected, :] = object2numeric_array(t_corrected[b_corrected])

    t_normal = from_ardx_packets.get_marker_list(d=file, mode='normal')
    b_normal = np.array([False if ti is False else True for ti in t_normal])
    t_normal2 = np.zeros((len(t_corrected), 2))
    t_normal2[b_normal, :] = object2numeric_array(t_normal[b_normal])

    # fig, ax = new_fig(aspect=1)
    # ax.plot(*t_corrected.T, color='blue', marker='o', ls='', label='corrected')
    # ax.plot(*t_normal.T, color='red', marker='x', ls='', label='normal')
    #
    fig, ax = new_fig()
    ax.plot(t_normal2[b_normal, 0], color='red', marker='x', markersize=10, ls='', label='normal')
    ax.plot(t_normal2[b_normal, 1], color='red', marker='<', markersize=10, ls='')

    ax.plot(t_corrected2[b_normal, 0], color='blue', marker='x', ls='', label='corrected')
    ax.plot(t_corrected2[b_normal, 1], color='blue', marker='<', ls='')

    d = t_normal2[b_normal] - t_corrected2[b_normal]


def get_qt(n, q_mode, m_mode):

    file_pole = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-pole-1657128676-measurements.npy", allow_pickle=True)
    file_right = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-right-1657126485-measurements.npy", allow_pickle=True)
    # file_right = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-right-measurements.npy", allow_pickle=True)
    file_left = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_50_kinect-left-1657121972-measurements.npy", allow_pickle=True)
    # file_left = np.load(f"{ICHR22_AUTOCALIBRATION}/Measurements/Real/paths_70_kinect-left-measurements.npy", allow_pickle=True)

    q_pole, t_pole = from_ardx_packets.get_qt_kinect(d=file_pole, q_mode=q_mode, m_mode=m_mode)
    q_right, t_right = from_ardx_packets.get_qt_kinect(d=file_right, q_mode=q_mode, m_mode=m_mode)
    q_left, t_left = from_ardx_packets.get_qt_kinect(d=file_left, q_mode=q_mode, m_mode=m_mode)

    q_pole, t_pole = q_pole[:n], t_pole[:n]
    q_right, t_right = q_right[:n], t_right[:n]
    q_left, t_left = q_left[:n], t_left[:n]

    # i_delete_pole = np.array([85, 86, 87, 88, 89, 158, 190, 191, 192, 193, 194, 216, 229])
    # i_delete_right = np.array([65, 77, 94, 95, 96, 97, 131, 134, 149, 160, 170, 200, 228, 233, 273])
    # i_delete_left = np.array([45, 56, 65, 66, 93, 94, 95, 96, 97, 121, 147, 148, 149, 197, 200])
    # q_pole, t_pole = np.delete(q_pole, i_delete_pole, axis=0), np.delete(t_pole, i_delete_pole, axis=0)
    # q_right, t_right = np.delete(q_right, i_delete_right, axis=0), np.delete(t_right, i_delete_right, axis=0)
    # q_left, t_left = np.delete(q_left, i_delete_left, axis=0), np.delete(t_left, i_delete_left, axis=0)
    #
    # i_delete_pole = np.array([53, 81, 84, 93, 99, 110, 111, 112, 113, 114, 158, 197])
    # i_delete_right = np.array([16, 48, 49, 50, 51, 55, 56, 123, 124, 126, 127, 128, 246, 247, 248])
    # i_delete_left = np.array([47, 48, 49, 50, 51, 81, 82, 97, 98, 147, 167, 217, 218, 219, 220])
    # q_pole, t_pole = np.delete(q_pole, i_delete_pole, axis=0), np.delete(t_pole, i_delete_pole, axis=0)
    # q_right, t_right = np.delete(q_right, i_delete_right, axis=0), np.delete(t_right, i_delete_right, axis=0)
    # q_left, t_left = np.delete(q_left, i_delete_left, axis=0), np.delete(t_left, i_delete_left, axis=0)

    # i_delete_left = np.array([63, 82, 98, 121, 122, 124, 125, 126, 184, 212, 214, 217, 226,
    #                           229, 230])
    # q_left, t_left = np.delete(q_left, i_delete_left, axis=0), np.delete(t_left, i_delete_left, axis=0)

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
    save_fig(fig=fig, file=f'{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/pixel_error_hist_{dkmca}', formats='pdf')

    mse = np.linalg.norm(stats, axis=-1)
    mse_pole = mse[l[0]:l[1]]
    mse_right = mse[l[1]:l[2]]
    mse_left = mse[l[2]:l[3]]

    str_pole =  f"pole:  mean={np.round(mse_pole.mean(),  3)} | std={np.round(mse_pole.std(),  3)} | max={np.round(mse_pole.max(), 3)}"
    str_right = f"right: mean={np.round(mse_right.mean(), 3)} | std={np.round(mse_right.std(), 3)} | max={np.round(mse_right.max(), 3)}"
    str_left =  f"left:  mean={np.round(mse_left.mean(),  3)} | std={np.round(mse_left.std(),  3)} | max={np.round(mse_left.max(), 3)}"
    print(str_pole)
    print(str_right)
    print(str_left)

    fig, ax = new_fig(title=f'Calibration Mode: {dkmca}')
    ax.hist(mse_pole, color=color_pole, range=(0, 15), bins=50, alpha=0.5, label=str_pole)
    ax.hist(mse_right, color=color_right, range=(0, 15), bins=50, alpha=0.5, label=str_right)
    ax.hist(mse_left, color=color_left, range=(0, 15), bins=50, alpha=0.5, label=str_left)
    ax.set_xlabel('|pixel difference|')
    ax.legend()
    save_fig(fig=fig, file=f'{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/pixel_error_scatter_{dkmca}', formats='pdf')
    # np.save(f"{ICHR22_AUTOCALIBRATION_FIGS}/Calibrations/kinect_pixel-error_{dkmca}.npy", dict(dp=stats, l=l))

    perc = 95
    i_delete_pole = np.nonzero(mse_pole > np.percentile(mse_pole, perc))[0]
    i_delete_right = np.nonzero(mse_right > np.percentile(mse_right, perc))[0]
    i_delete_left = np.nonzero(mse_left > np.percentile(mse_left, perc))[0]
    print(f'i_delete_pole = np.{repr(i_delete_pole)}')
    print(f'i_delete_right = np.{repr(i_delete_right)}')
    print(f'i_delete_left = np.{repr(i_delete_left)}')


def test_vicon2(x, el_loop):
    print('Calibrate frames of vicon targets with the new model:')
    cal_par = parameter.Parameter(x_weighting=0)
    cal_rob = Justin19CalVicon(dkmca='0g0c0', add_nominal_offsets=True, use_imu=False, el_loop=el_loop)

    cal_rob.dh = x['dh']
    cal_rob.el = x['el']
    cal_rob.ma = x['ma']

    file = f"{ICHR22_AUTOCALIBRATION}/Vicon/2/random_poses_smooth_100-1657536656-measurements.npy"
    q, t = from_ardx_packets.get_qt_vicon(file=file)
    i = np.array([69,  70,  45,  84,  65,  28,  98,  51,  85, 100])
    q, t = np.delete(q, i, axis=0), np.delete(t, i, axis=0)
    (q0_cal, t_cal), (q0_test, t_test) = train_test_split(q, t, split=-1, shuffle=False)

    x, stats = calibration.calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=4,
                                     cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    print_stats2(stats)


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
    get_frames = calibration.create_wrapper_kinematic(x=x, cal_rob=cal_rob)

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


def print_dh_differences():
    dh0 = Justin19().dh
    dh2 = x['dh']
    d = dh0 - dh2
    print(np.abs(d).max(axis=0))
    import pandas as pd
    print('DH Difference:')
    print(pd.DataFrame(np.round(d, 3)))


def main():
    dkmca = 'c00cf'
    el_loop = 1
    q, t, l = get_qt(n=300, q_mode='commanded', m_mode='corrected')

    cal_par = parameter.Parameter(x_weighting=10, t_weighting=1*np.array([1, 1, 1]))
    cal_rob = Justin19CalKinect(dkmca=dkmca, add_nominal_offsets=True, use_imu=False, el_loop=el_loop)

    (q_cal, t_cal), (q_test, t_test) = train_test_split(q, t, split=-1, shuffle=False)
    x, stats = calibration.calibrate(q_cal=q_cal, t_cal=t_cal, q_test=q_test, t_test=t_test, verbose=3, obj_fun='marker_image',
                                     cal_par=cal_par, cal_rob=cal_rob, x0_noise=0.0)

    x0 = x.copy()

    x = parameter.unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)

    x = round_dict(d=x, decimals=5)
    print_dict(x)

    plot_pixel_error(dkmca=dkmca, stats=stats, l=l)
    test_vicon2(x=x, el_loop=el_loop)


if __name__ == '__main__':
    main()
    # compare_image_modes()


# with x0 from new vicon
# with z
# pole:  0.947
# right: 1.02
# left:  1.244
# Translation [mm]   9.07897   3.74407   8.48967   1.89093  19.13859
#   Rotation [deg]   0.38770   0.01376   0.38592   0.35884   0.41981

# without z
# pole:  1.021
# right: 0.84
# left:  1.129
#                       mean       std    median       min       max
# Translation [mm]   8.99132   4.20818   8.30040   1.79872  22.18619
#   Rotation [deg]   0.38658   0.01070   0.38546   0.36688   0.41213


# with nominal x0
# with z
#                       mean       std    median       min       max
# Translation [mm]   8.27980   3.33803   7.94582   1.38547  17.76661
#   Rotation [deg]   0.38811   0.01295   0.38713   0.36102   0.41835

# without z
#                       mean       std    median       min       max
# Translation [mm]   9.12452   4.07334   8.68069   1.74044  21.06296
#   Rotation [deg]   0.39141   0.01111   0.39071   0.36968   0.41627


#                       mean       std    median       min       max
# Translation [mm]   4.06009   2.09373   3.72899   0.82851  13.36096
#   Rotation [deg]   0.40257   0.00402   0.40241   0.39279   0.41378


# with z
#                       mean       std    median       min       max
# Translation [mm]   4.31208   2.13528   3.88466   0.61600  10.02752
#   Rotation [deg]   0.39934   0.00572   0.39977   0.38492   0.41285



####
# FINDING in this case better kinect calibration improved the results of the vicon calibration - nice, but only slightly
# z
# free joints right
# pole:  1.422
# right: 1.152
# left:  1.736
# Frame__0
#                       mean       std    median       min       max
# Translation [mm]    3.1557    1.6422    2.8508    0.4256    7.7200
#   Rotation [deg]   13.2247    0.1952   13.2050   12.6815   13.6812
# Frame__1
#                       mean       std    median       min       max
# Translation [mm]    5.2623    2.5397    4.7285    0.9364   11.4020
#   Rotation [deg]   32.7090    0.3708   32.7178   31.7615   33.7118

# fix joints right
# pole:  1.44
# right: 1.399
# left:  1.639
# Frame__0
#                       mean       std    median       min       max
# Translation [mm]    3.2579    1.7401    3.2928    0.2630    8.1484
#   Rotation [deg]   13.0505    0.3147   13.0590   12.2663   13.7344
# Frame__1
#                       mean       std    median       min       max
# Translation [mm]    5.2849    2.5300    4.7602    0.7360   11.6310
#   Rotation [deg]   32.6011    0.3088   32.6048   31.7684   33.4458

# However there are cases where you can only free one single parameter in the torso and it gets so much worse in the cartesian
#   while being slightly better for the


# One marker gets the worse than the others, most of the time its the left one, but sometimes it is the right one
# berthold thinks the reason for this is that the elasticities get shifted between the different body parts and can not fully be explained by the other calibration method



