import numpy as np

from rocal.Robots.Justin19 import Justin19Cal
from rocal.Measurements.io2 import get_q
from rocal.calibration import calibrate
from rocal.main import cal_par


def leave_one_out_analysis_joints():

    cal_rob = Justin19Cal(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=1)
    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=75)

    d = np.zeros(20, dtype=object)
    print(len(cal_rob.dh_bool_c))
    for i in range(20):
        print(i)
        cal_rob = Justin19Cal(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=0)
        cal_rob.dh_bool_c[i, :] = False
        cal_rob.cp_bool_c[i, :] = False

        x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                             cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

        d[i] = (x, stats)
    np.save('leave_one_out_joints.test', d)

    # 0
    #                       mean       std    median       min       max
    # Translation [mm]   3.85569   1.89718   3.48282   0.54580   9.74722
    #   Rotation [deg]   0.42432   0.00442   0.42415   0.41215   0.43747
    # 1
    #                       mean       std    median       min       max
    # Translation [mm]   4.43106   2.17606   4.12763   0.69907  11.89570
    #   Rotation [deg]   0.42533   0.00439   0.42515   0.41540   0.43939
    # 2
    #                       mean       std    median       min       max
    # Translation [mm]   5.15282   2.63764   4.72278   0.67073  15.64670
    #   Rotation [deg]   0.42512   0.00469   0.42488   0.41211   0.44005
    # 3
    #                       mean       std    median       min       max
    # Translation [mm]   5.27316   2.49574   4.68187   0.91025  15.26738
    #   Rotation [deg]   0.42482   0.00525   0.42481   0.41128   0.43976
    # 4
    #                       mean       std    median       min       max
    # Translation [mm]   3.94134   1.97286   3.58729   0.51300   9.85564
    #   Rotation [deg]   0.42526   0.00471   0.42511   0.41291   0.44175
    # 5
    #                       mean       std    median       min       max
    # Translation [mm]   4.81101   2.19598   4.64103   0.64016  12.26936
    #   Rotation [deg]   0.42605   0.00569   0.42617   0.41017   0.44257
    # 6
    #                       mean       std    median       min       max
    # Translation [mm]   4.99399   2.12952   4.64639   0.92249  11.48371
    #   Rotation [deg]   0.42416   0.00704   0.42461   0.40773   0.44048
    # 7
    #                       mean       std    median       min       max
    # Translation [mm]   3.91401   1.78351   3.69883   0.61188   9.87544
    #   Rotation [deg]   0.42417   0.00422   0.42408   0.41166   0.44020
    # 8
    #                       mean       std    median       min       max
    # Translation [mm]   3.67848   1.70854   3.46260   0.62808   8.70999
    #   Rotation [deg]   0.42486   0.00409   0.42472   0.41334   0.43956
    # 9
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890
    # 10
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890
    # 11
    #                       mean       std    median       min       max
    # Translation [mm]   3.96121   1.87977   3.74557   0.67924   9.65674
    #   Rotation [deg]   0.42516   0.00476   0.42510   0.41127   0.43966
    # 12
    #                       mean       std    median       min       max
    # Translation [mm]   4.73212   2.13598   4.51412   0.93685  10.80704
    #   Rotation [deg]   0.42521   0.00577   0.42527   0.41211   0.44347
    # 13
    #                       mean       std    median       min       max
    # Translation [mm]   3.75693   1.75022   3.43782   0.47185   8.56264
    #   Rotation [deg]   0.42355   0.00423   0.42330   0.41316   0.43707
    # 14
    #                      mean       std    median       min       max
    # Translation [mm]   3.74451   1.74478   3.51743   0.59041   8.77585
    #   Rotation [deg]   0.42385   0.00421   0.42330   0.41277   0.43882
    # 15
    #                       mean       std    median       min       max
    # Translation [mm]   3.75783   1.77383   3.50608   0.58460  10.15400
    #   Rotation [deg]   0.42536   0.00561   0.42524   0.41016   0.43943
    # 16
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890
    # 17
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890
    # 18
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890
    # 19
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890


def leave_one_out_analysis():
    name_list = ['all', 'd', 'theta', 'a', 'alpha', 'cp_theta', 'cp_alpha']
    cal_rob = Justin19Cal(dcmf='ff0c', ma0=True, fr0=True, use_imu=True, cp_loop=1)

    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=75)

    d = {}
    for name in name_list:
        print(name)
        _cal_rob = Justin19Cal(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=0)
        if name == 'd':
            _cal_rob.dh_bool_c[:, 0] = False
        elif name == 'theta':
            _cal_rob.dh_bool_c[:, 1] = False
        elif name == 'a':
            _cal_rob.dh_bool_c[:, 2] = False
        elif name == 'alpha':
            _cal_rob.dh_bool_c[:, 3] = False
        elif name == 'cp_theta':
            _cal_rob.cp_bool_c[:, 2] = False
        elif name == 'cp_alpha':
            _cal_rob.cp_bool_c[:, 0] = False
        else:
            pass

        x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                             cal_par=cal_par, cal_rob=_cal_rob, x0_noise=0)

        d[name] = (x, stats)
    np.save('ddd.test', d)
