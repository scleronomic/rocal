import os
import numpy as np


from rocal.Robots.Justin19 import Justin19Cal
from rocal.calibration import calibrate
from rocal.main import cal_par

from rocal.definitions import ICHR20_CALIBRATION


def test_static_equilibrium_truncation():
    directory = ICHR20_CALIBRATION + '/Justin19/f13_22/static_equ_test0/'

    cal_rob = Justin19Cal(dcmf='0c00', ma0=True, fr0=False, use_imu=False, cp_loop=0)
    np.random.seed(0)
    n = 100
    q = cal_rob.sample_q(1000)[:n]

    d = {}
    print(os.listdir(directory))
    for file in os.listdir(directory):
        print(file)
        _, t, t_noise = np.load(directory + file, allow_pickle=True)
        t = t[:n]
        t_noise = t_noise[:n]
        x_list = []
        stats_list = []
        for i in range(4):
            print(i)
            cal_rob = Justin19Cal(dcmf='cc0c', ma0=True, fr0=False, use_imu=False, cp_loop=i)
            cal_rob.ma[:, 1:] *= 1
            x, stats = calibrate(q_cal=q, t_cal=t, q_test=q, t_test=t, verbose=1,
                                 cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

            x_list.append(x)
            stats_list.append(stats)
        d[file.split('_')[0]] = (x_list, stats_list)

    np.save(directory + 'results.npy', d)

