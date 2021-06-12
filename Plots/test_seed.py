import numpy as np
import ray

from wzk import change_tuple_order, object2numeric_array

from rocal.calibration import calibrate
from rocal.Measurements.io2 import get_q

from rocal.definitions import ICHR20_CALIBRATION
from rocal.Robots.Justin19 import Justin19Cal


def test_seed():
    ray.init(address='auto')

    cal_rob = Justin19Cal(dkmc='cc0c', ma0=True, fr0=True, use_imu=False, el_loop=0)

    @ray.remote
    def calibrate_ray2(seed):
        (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=seed)
        return calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=0,
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

    futures = []
    for seed in range(100):
        futures.append(calibrate_ray2.remote(seed))

    x_all, stats_all = change_tuple_order(ray.get(futures))
    np.save(ICHR20_CALIBRATION + f'/600_seed_influence_rot{cal_par.sigma_rot}.npy',
            (object2numeric_array(x_all), object2numeric_array(stats_all), []))


def test_seed2():
    from wzk import new_fig

    x, stats, _ = np.load(ICHR20_CALIBRATION + '/600_seed_influence_rot0.npy', allow_pickle=True)

    a = stats[:, :, 0, :].mean(axis=1)

    i0 = np.argsort(a[:, 0])
    i2 = np.argsort(a[:, 2])
    i4 = np.argsort(a[:, 4])

    fig, ax = new_fig()
    a0 = a[i0]
    a2 = a[i2]
    a4 = a[i4]
    ax.plot(a0[:, 0], color='b', marker='o')
    ax.plot(a0[:, 2], color='b', marker='s')
    ax.plot(a0[:, 4], color='b', marker='x')
    ax.plot(a2[:, 0], color='r', marker='o')
    ax.plot(a2[:, 2], color='r', marker='s')
    ax.plot(a2[:, 4], color='r', marker='x')
    ax.plot(a4[:, 0], color='m', marker='o')
    ax.plot(a4[:, 2], color='m', marker='s')
    ax.plot(a4[:, 4], color='m', marker='x')
    print(a[i4[0]])
    print(i4[0], a[i4[0]])

    # a = np.rad2deg(stats[:, :, 1, :]).mean(axis=1)
    # print(a[0])
    # i0 = np.argsort(a[:, 0])
    # i2 = np.argsort(a[:, 2])
    # i4 = np.argsort(a[:, 4])
    #
    # fig, ax = new_fig()
    # a0 = a[i0]
    # a2 = a[i2]
    # a4 = a[i4]
    # ax.plot(a0[:, 0], color='b', marker='o')
    # ax.plot(a0[:, 2], color='b', marker='s')
    # ax.plot(a0[:, 4], color='b', marker='x')
    # ax.plot(a2[:, 0], color='r', marker='o')
    # ax.plot(a2[:, 2], color='r', marker='s')
    # ax.plot(a2[:, 4], color='r', marker='x')
    # ax.plot(a4[:, 0], color='m', marker='o')
    # ax.plot(a4[:, 2], color='m', marker='s')
    # ax.plot(a4[:, 4], color='m', marker='x')
    # print(a[i4[3]])
    # print(i4[3], a[i4[3]])
