from wzk import spatial

from rocal.calibration import calibrate
from rocal.Measurements.io2 import get_q
from rocal.Robots.Justin19 import Justin19Cal
from rocal.Vis.plotting import print_stats2
from rocal.parameter import Parameter

from definitions import ICHR20_CALIBRATION

# TODO check_marker_occlusion(q)
# TODO check for duplicate indices
# TODO clip the 1000 bestworst because only a small portion is at the extremes
# FINDING 10000 test configurations is easily enough to have a small variance


cal_par = Parameter()


def search_world_frame(_cal_rob, q, t):
    """
    its hard / impossible to find the correct world frame if it the initial guess is far off
       -> try multi start and safe the good fits
    """
    n = 100
    threshold = 0.1  # m
    for i in range(n):
        _x, _stats = calibrate(q_cal=q, t_cal=t, q_test=q, t_test=t, x0_noise=0.1, verbose=0,
                               cal_rob=_cal_rob, cal_par=cal_par)
        if _stats[0, 0, 0] < threshold:
            print(_stats[:, 0, 0].meand(axis=0))
            print(spatial.trans_rotvec2frame(trans=_x[:3], rotvec=_x[3:6]))


if __name__ == '__main__':

    cal_rob = Justin19Cal(dkmc='cc0c', add_nominal_offsets=True, use_imu=False, el_loop=1)

    directory = ICHR20_CALIBRATION + '/Measurements/600'
    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=-1, seed=75)

    # import pandas as pd
    # df = pd.DataFrame([(q, t) for q, t in zip(q0_cal, t_cal[:, :, :3, -1])], columns=('joints', 'markers'))
    # print(df.shape)

    x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

    print_stats2(stats)

    from rocal.parameter import unwrap_x
    print(cal_rob.dh)
    print(cal_rob.ma)
    x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)
    print(x)
    # #
    # save_file = 'final_without_cp'
    # save_file = f'{directory}/results/{save_file}.npy'
    # x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)
    # np.save(save_file, (x, stats))
    # #
    # # x0 = x

    # f0 = kinematic(cal_rob=cal_rob, q=q0, **xn)

    # x = wrap_x(x=x, cal_rob=cal_rob)
    # print(evaluate_x(cal_rob=cal_rob, x_list=[x], squared=False, q=q0_test, t=t_test))
    # tic()
    # x, stats = change_tuple_order(calibrate2(q_cal=q_cal, t_cal=t_cal, q_test=q_test, t_test=t_test, c=c, verbose=1)
    #                               for _ in range(20))
    # toc()
