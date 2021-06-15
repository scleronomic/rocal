import numpy as np
from itertools import combinations
from wzk import tic, toc
from wzk.spatial import sample_frame_noise, trans_rotvec2frame, frame2trans_rotvec, frame_difference, invert
from wzk.mpl import new_fig, save_fig

from rocal.calibration import calibrate
from rocal.Robots import JustinHand12Cal, JustinHand12CalCal
from rocal.parameter import Parameter
from rocal.measurment_functions import _cal_touch


def load_data():
    import scipy.io
    mat = scipy.io.loadmat('/Users/jote/Documents/Code/Python/DLR/rocal/export-1.mat')['data'][0, 0]
    q = np.concatenate((mat[1], mat[2], mat[3], mat[4], mat[5], mat[6]))
    # q = np.ones((6, 12))
    n = len(q)

    # t_f t_m t_r f_m f_r m_r
    finger_dict = {'r': 0,
                   'm': 1,
                   'f': 2,
                   't': 3}

    pairs = np.array([[3, 2],
                      [3, 1],
                      [3, 0],
                      [2, 1],
                      [2, 0],
                      [1, 0]])

    pairs = np.repeat(pairs, n//6, axis=0)
    pairs = cal_rob.cm_f_idx[pairs]
    d = np.zeros(n)
    return q, d, pairs


if __name__ == '__main__':
    pass
    cal_par = Parameter()
    cal_rob = JustinHand12Cal(dkmc='f000')
    # cal_rob = JustinHand12CalCal(dkmc='000f')

    # n = 60
    # pairs = np.array(list(combinations(cal_rob.cm_f_idx, 2)))
    # pairs = np.repeat(pairs, n/6, axis=0)
    # q = cal_rob.sample_q(n)

    # cm = sample_frame_noise(shape=4, trans=0.05, rot=0.2)
    cm0 = sample_frame_noise(shape=4, trans=0., rot=0.0)

    cm = cm0
    q, d, pairs = load_data()
    q[10] = np.array([0.3027871, -0.3161505, 0.6642020,
                      -0.0171715, 0.4152261, 0.8547665,
                      -0.4747111, 0.3777183, 0.1215826,
                      -0.0106201, -0.1294758, 0.9147583])
    f = cal_rob.get_frames(q)

    # d = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm)d
    d0 = _cal_touch(f=f, pairs=pairs, cal_rob=cal_rob, cm=cm0)
    tic()
    x, d2 = calibrate(q_cal=q, t_cal=(pairs, d), q_test=q, t_test=(pairs, d), verbose=1,
                      obj_fun='touch',
                      cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    toc('CALIBRATION')
    d0, d2 = d0*1000, d2*1000
    print('max error 0:', np.abs(d - d0).max(), 'mm')
    print('max error  :', np.abs(d - d2).max(), 'mm')
    # idx = np.argsort(np.abs(d - d0))
    cm2 = np.reshape(x[-4*6:], (4, 6)).copy()
    trans2, rot2 = cm2[:, :3], cm2[:, 3:]
    cm2 = trans_rotvec2frame(trans=trans2, rotvec=rot2)

    trans, rot = frame2trans_rotvec(f=cm)
    d_trans, d_rot = frame_difference(cm, cm2)
    print(trans - trans2)  # TODO there is redundancy!

    fig, ax = new_fig()
    ax.hist(d0, alpha=0.5, color='b', density=True, bins=20)
    ax.hist(d2, alpha=0.5, color='r', density=True, bins=20)

    f0 = cal_rob.f_static[1::2]
    from wzk.spatial import invert
    cm3 = invert(cm2[2]) @ cm2  #


    d3 = _cal_touch(f=f, cm=cm3, pairs=pairs, cal_rob=cal_rob)
    print(d0.mean(), d0.max())
    print(d2.mean(), d2.max())
    print(d3.mean(), d3.max())

    f3 = cm3 @ f0  #  normalize on fore finger [2]
    # cal_rob.f_static[1::2] = np.array([[[-0.06545762,  0.74404062,  0.66492018, -0.00578521],
    #                                     [-0.05458364, -0.66802143,  0.74213745, -0.06114935],
    #                                     [ 0.99636134,  0.01228479,  0.08433953,  0.1326214 ],
    #                                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                                    [[-0.21002886,  0.05765612,  0.97599367, -0.04815137],
    #                                     [-0.09710584, -0.99455386,  0.03785587, -0.01210181],
    #                                     [ 0.9728609 , -0.08682386,  0.21448376,  0.13330026],
    #                                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                                    [[-0.21002886,  0.05765612,  0.97599367, -0.04598232],
    #                                     [-0.09710584, -0.99455386,  0.03785587,  0.02769159],
    #                                     [ 0.9728609 , -0.08682386,  0.21448376,  0.13884357],
    #                                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                                    [[ 0.46476466,  0.104609  , -0.87923306,  0.05226424],
    #                                     [-0.30343889,  0.95168265, -0.04716964,  0.02698612],
    #                                     [ 0.83181648,  0.28871628,  0.4740509 ,  0.10262013],
    #                                     [ 0.        ,  0.        ,  0.        ,  1.        ]]])
    fig, ax = new_fig()
    i_t = np.hstack((np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)))
    i_f = np.hstack((np.arange(0, 10), np.arange(30, 40), np.arange(40, 50)))
    i_m = np.hstack((np.arange(10, 20), np.arange(30, 40), np.arange(50, 60)))
    i_r = np.hstack((np.arange(20, 30), np.arange(40, 50), np.arange(50, 60)))
    ax.plot(d0[i_t], d2[i_t],  ls='', marker=0, markersize=5, label='thumb', color='k')
    ax.plot(d0[i_f], d2[i_f],  ls='', marker=1, markersize=5, label='fore', color='r')
    ax.plot(d0[i_m], d2[i_m],  ls='', marker=2, markersize=5, label='middle', color='b')
    ax.plot(d0[i_r], d2[i_r],  ls='', marker=3, markersize=5, label='ring', color='m')

    # ax.plot(d0[0:10], d2[0:10],   ls='', marker='x', label='t_f', color='r')
    # ax.plot(d0[10:20], d2[10:20], ls='', marker='+', label='t_m', color='b')
    # ax.plot(d0[20:30], d2[20:30], ls='', marker='<', label='t_r', color='k')
    # ax.plot(d0[30:40], d2[30:40], ls='', marker='>', label='f_m', color='g')
    # ax.plot(d0[40:50], d2[40:50], ls='', marker='D', label='f_r', color='y')
    # ax.plot(d0[50:60], d2[50:60], ls='', marker='P', label='m_r', color='m')
    # ax.plot(np.linspace(d2.min(), d2.max()), np.linspace(d2.min(), d2.max()), color='b')
    # ax.plot(np.linspace(-d2.max(), -d2.min()), np.linspace(d2.max(), d2.min()), color='b')
    ax.set_xlabel('error before calibration [mm]')
    ax.set_ylabel('error after calibration [mm]')
    ax.legend()
    ax.grid()
    save_fig(fig=fig, filename='tfmr_calibration_error', formats='pdf')
    # ax.hlines(0, -0.01, +0.01, 'k')
    # ax.vlines(0, -0.01, +0.0025, 'k')
    print(repr(f3))

