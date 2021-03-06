import numpy as np


from rocal.Robots import Justin19Cal
from rocal.Measurements.io2 import get_q, get_parameter_identifier
from rocal.calibration import calibrate
from rocal.parameter import unwrap_x, Parameter

from rocal.definitions import ICHR20_CALIBRATION

directory = ICHR20_CALIBRATION + '/Measurements/600'
cal_par = Parameter()


def iterative_mass_compliance():

    cal_rob = Justin19Cal(dkmc='cc0c', use_imu=True, el_loop=0)

    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=75)

    x_list, stats_list = [], []
    for i in range(20):
        print(i)
        if i % 3 == 0:
            cal_rob = Justin19Cal(dkmc='cc0c', use_imu=True, el_loop=0)

        elif i % 3 == 1:
            cal_rob = Justin19Cal(dkmc='00p0', use_imu=True, el_loop=0)

        elif i % 3 == 2:
            cal_rob = Justin19Cal(dkmc='00m0', use_imu=True, el_loop=0)

        if i > 0:
            cal_rob.dh = x['dh']
            cal_rob.el = x['el']
            cal_rob.ma = x['ma']
            cal_rob.cm = x['cm']
        x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                             cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
        x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)

        x_list.append(x)
        stats_list.append(stats)

    save_file = get_parameter_identifier(cal_rob)
    save_file = f'{directory}/results/{save_file}_iterative_mass_compliance.npy'

    np.save(save_file, (x_list, stats_list, []))
