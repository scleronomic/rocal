import ray
import numpy as np

from wzk import train_test_split, change_tuple_order, object2numeric_array

from rocal.Robots.Justin19 import Justin19Cal
from rocal.Measurements.io2 import get_q
from rocal.calibration import calibrate
from rocal.main import cal_par

from rocal.definitions import ICHR20_CALIBRATION


def test_n_influence(seed=75):
    ray.init(address='auto')

    cal_rob = Justin19Cal(dcmf='cc0c', ma0=True, fr0=True, use_imu=False, cp_loop=0)
    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=seed)

    n_list = np.hstack([np.arange(1, 21, 1),
                        np.arange(25, 50, 5),
                        np.arange(50, 100, 10),
                        np.arange(100, 301, 20)])
    n = len(n_list)
    m = 100
    x_all = np.zeros((n, m), dtype=object)
    stats_all = np.zeros((n, m), dtype=object)

    @ray.remote
    def calibrate_ray(_n):
        _, (q0_cal_i, q_cal_i, t_cal_i) = train_test_split(q0_cal, q_cal, t_cal, split=int(_n), shuffle=True)
        return calibrate(q_cal=q0_cal_i, t_cal=t_cal_i, q_test=q0_test, t_test=t_test, verbose=0,
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

    for i, nn in enumerate(n_list):
        print(nn)

        futures = []
        for j in range(m):
            futures.append(calibrate_ray.remote(nn))
        x, stats = change_tuple_order(ray.get(futures))
        x_all[i, :] = x
        stats_all[i, :] = stats

        np.save(ICHR20_CALIBRATION + f'/600_test_n_influence_rot0_seed{seed}.npy',
                (n_list, object2numeric_array(x_all), object2numeric_array(stats_all), []))
