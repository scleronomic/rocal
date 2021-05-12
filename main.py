import os

import ray
import numpy as np

from Justin.Calibration.calibration import (calibrate, unwrap_x, kinematic, frame_difference)
from Justin.Calibration.Measurements.io2 import get_q, get_parameter_identifier


from Justin.Calibration.justin import Justin19Calib
from definitions import DLR_USERSTORE_PAPER_20CAL

from wzk import (print_table, train_test_split,
                 combine_iterative_indices, delete_args)
from wzk import change_tuple_order, object2numeric_array


# TODO check_marker_occlusion(q)
# TODO check for duplicate indices
# TODO clip the 1000 bestworst because only a small portion is at the extremes
# FINDING 10000 test configurations is easily enough to have a small variance
#


class CalibrationParameter:
    def __init__(self):
        self.sigma_trans, self.sigma_rot = 1000, 0  # was 100
        self.x_weighting = 0.01  # was  0.01 for dummy
        self.f_weighting = [1, 1]
        self.method = 'PyOpt - SLSQP'  # way faster
        self.options = {'maxiter': 200,
                        'disp': True,
                        'ftol': 1e-7}


cal_par = CalibrationParameter()


def search_world_frame(cal_rob, q, t):
    # FINDING its hard / impossible to find the correct world frame if it the initial guess is far off
    #   -> try multi start and safe the good fits
    n = 100
    threshold = 0.1  # m
    for i in range(n):
        x, stats = calibrate(q_cal=q, t_cal=t, q_test=q, t_test=t, x0_noise=0.1, verbose=0,
                             cal_rob=cal_rob, cal_par=cal_par)
        if stats[0, 0, 0] < threshold:
            print(stats[:, 0, 0].meand(axis=0))
            from Kinematic.frames import trans_rotvec2frame
            print(trans_rotvec2frame(trans=x[:3], rotvec=x[3:6]))


def test_n_influence(seed=75):
    ray.init(address='auto')

    cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=False, cp_loop=0)
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
    def calibrate_ray(nn):
        _, (q0_cal_i, q_cal_i, t_cal_i) = train_test_split(q0_cal, q_cal, t_cal, split=int(nn), shuffle=True)
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

        np.save(DLR_USERSTORE_PAPER_20CAL + f'/600_test_n_influence_rot0_seed{seed}.npy',
                (n_list, object2numeric_array(x_all), object2numeric_array(stats_all), []))

def test_seed():
    ray.init(address='auto')

    cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=False, cp_loop=0)

    @ray.remote
    def calibrate_ray2(seed):
        (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=seed)
        return calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=0,
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

    futures = []
    for seed in range(100):
        futures.append(calibrate_ray2.remote(seed))

    x_all, stats_all = change_tuple_order(ray.get(futures))
    np.save(DLR_USERSTORE_PAPER_20CAL + f'/600_seed_influence_rot{cal_par.sigma_rot}.npy',
            (object2numeric_array(x_all), object2numeric_array(stats_all), []))


def test_seed2():
    from wzk import new_fig

    x, stats, _ = np.load(DLR_USERSTORE_PAPER_20CAL + '/600_seed_influence_rot0.npy', allow_pickle=True)

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


def iterative_mass_compliance():

    cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=0)

    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=75)

    x_list, stats_list = [], []
    for i in range(20):
        print(i)
        if i % 3 == 0:
            cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=0)

        elif i % 3 == 1:
            cal_rob = Justin19Calib(dcmf='00p0', ma0=True, fr0=True, use_imu=True, cp_loop=0)

        elif i % 3 == 2:
            cal_rob = Justin19Calib(dcmf='00m0', ma0=True, fr0=True, use_imu=True, cp_loop=0)

        if i > 0:
            cal_rob.dh = x['dh']
            cal_rob.cp = x['cp']
            cal_rob.ma = x['ma']
            cal_rob.fr = x['fr']
        x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                             cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
        x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)

        x_list.append(x)
        stats_list.append(stats)

    save_file = get_parameter_identifier(cal_rob)
    directory = DLR_USERSTORE_PAPER_20CAL + '/Measurements/600'
    save_file = f'{directory}/results/{save_file}_iterative_mass_compliance.npy'

    np.save(save_file, (x_list, stats_list, []))


def test_static_equilibrium_truncation():
    directory = DLR_USERSTORE_PAPER_20CAL + '/Justin19/f13_22/static_equ_test0/'

    cal_rob = Justin19Calib(dcmf='0c00', ma0=True, fr0=False, use_imu=False, cp_loop=0)
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
            cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=False, use_imu=False, cp_loop=i)
            cal_rob.ma[:, 1:] *= 1
            x, stats = calibrate(q_cal=q, t_cal=t, q_test=q, t_test=t, verbose=1,
                                 cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
            # print(unwrap_x(cal_rob=cal_rob, x=x, add_nominal_offset=True)['cp'])
            x_list.append(x)
            stats_list.append(stats)
        d[file.split('_')[0]] = (x_list, stats_list)

    np.save(directory + 'results.npy', d)


def leave_one_out_analysis_joints():

    cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=1)
    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=75)

    d = np.zeros(20,dtype=object)
    print(len(cal_rob.dh_bool_c))
    for i in range(20):
        print(i)
        cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=0)
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
    dcmf = 'cc0c', 'jc0c', 'Xc0c', 'cj0c'
    name_list = ['all', 'd', 'theta', 'a', 'alpha', 'cp_theta', 'cp_alpha']
    cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=1)

    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=200, seed=75)

    d = {}
    for name in name_list:
        print(name)
        cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=True, cp_loop=0)
        if False:
            pass
        elif name == 'd':
            cal_rob.dh_bool_c[:, 0] = False
        elif name == 'theta':
            cal_rob.dh_bool_c[:, 1] = False
        elif name == 'a':
            cal_rob.dh_bool_c[:, 2] = False
        elif name == 'alpha':
            cal_rob.dh_bool_c[:, 3] = False
        elif name == 'cp_theta':
            cal_rob.cp_bool_c[:, 2] = False
        elif name == 'cp_alpha':
            cal_rob.cp_bool_c[:, 0] = False

        x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                             cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)

        d[name] = (x, stats)
    np.save('ddd.test', d)

    # all
    #                       mean       std    median       min       max
    # Translation [mm]   3.68196   1.73396   3.45258   0.45778   8.81959
    #   Rotation [deg]   0.42488   0.00391   0.42471   0.41537   0.43890
    # d
    #                       mean       std    median       min       max
    # Translation [mm]   3.71190   1.72964   3.44106   0.67895   8.82015
    #   Rotation [deg]   0.42528   0.00402   0.42496   0.41393   0.44042
    # theta
    #                       mean       std    median       min       max
    # Translation [mm]   6.35764   3.16010   5.81514   0.81279  18.89478
    #   Rotation [deg]   0.42522   0.00856   0.42537   0.40202   0.44466
    # a
    #                       mean       std    median       min       max
    # Translation [mm]   4.33373   2.03031   4.07281   0.69288  11.09058
    #   Rotation [deg]   0.42515   0.00449   0.42518   0.41218   0.43800
    # alpha


#                d      th     a     alp    cp_t  cp_a
# mean & 3.21 & 3.46 & 6.35 & 4.33 & 6.48 & 9.09 & 4.85 \\
# std  & 1.70 & 1.73 & 3.16 & 2.03 & 2.93 & 4.23 & 2.28 \\
# max  & 8.81 & 8.97 & 18.89 & 11.09 & 17.25 & 24.59 & 13.37


#               cp_t    alp     th     cp_a      a       d
# mean & 3.21 & 9.09  & 6.48  & 6.35  & 4.85  & 4.33  & 3.46 \\
# std  & 1.70 & 4.23  & 2.93  & 3.16  & 2.28  & 2.03  & 1.73 \\
# max  & 8.81 & 24.59 & 17.25 & 18.89 & 13.37 & 11.09 & 8.97


if __name__ == '__main__':
    pass
    # test_seed2()
    # iterative_mass_compliance()
    # evaluate_different_effects©()
    # leave_one_out_analysis_joints()
    # test_static_equilibrium_truncation()

    cal_rob = Justin19Calib(dcmf='cc0c', ma0=True, fr0=True, use_imu=False, cp_loop=0)

    directory = DLR_USERSTORE_PAPER_20CAL + '/Measurements/600'
    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=-1, seed=75)
    import pandas as pd
    df = pd.DataFrame([(q, t) for q, t in zip(q0_cal, t_cal[:, :, :3, -1])], columns=('joints', 'markers'))

    print(df.shape)
    # x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
    #                      cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
    # #
    # save_file = 'final_without_cp'
    # save_file = f'{directory}/results/{save_file}.npy'
    # x = unwrap_x(x=x, cal_rob=cal_rob, add_nominal_offset=True)
    # np.save(save_file, (x, stats))
    # #
    # # x0 = x

    # f0 = kinematic(cal_rob=cal_rob, q=q0, **xn)


    #
    # dcmf = 'cc0c'
    # cal_rob0 = Justin19Calib(dcmf=dcmf, ma0=True, fr0=True, use_imu=False, cp_loop=0)
    # cal_rob3 = Justin19Calib(dcmf=dcmf, ma0=True, fr0=True, use_imu=False, cp_loop=3)
    #
    # directory = DLR_USERSTORE_PAPER_2020_CALIB  # + '/Justin19'
    # q = load_q(directory=directory, cal_rob=cal_rob0)
    # m0 = np.load(file=directory + '/Justin19/f13_22/m__cc0c_0_11_ff.npy', allow_pickle=True)
    # m3 = np.load(file=directory + '/Justin19/f13_22/m__cc0c_3_11_ff.npy', allow_pickle=True)
    #
    # (x_dict_true0, t0, t_noise0, t_test0) = m0
    # (x_dict_true3, t3, t_noise3, t_test3) = m3
    #
    #
    # def calibrate_xxx(cal_rob, t_cal, t_test):
    #     return calibrate(q_cal=q[:500], t_cal=t_cal[:500], q_test=q, t_test=t_test, cal_rob=cal_rob, x0_noise=0,
    #                      verbose=1, cal_par=cal_par)
    #
    # x000, stats000 = calibrate_xxx(cal_rob=cal_rob0, t_cal=t_noise0[0], t_test=t0)
    # x003, stats003 = calibrate_xxx(cal_rob=cal_rob0, t_cal=t_noise0[0], t_test=t3)
    # x033, stats033 = calibrate_xxx(cal_rob=cal_rob0, t_cal=t_noise3[0], t_test=t3)
    # x333, stats333 = calibrate_xxx(cal_rob=cal_rob3, t_cal=t_noise3[0], t_test=t3)
    # x330, stats330 = calibrate_xxx(cal_rob=cal_rob3, t_cal=t_noise3[0], t_test=t0)
    # x300, stats300 = calibrate_xxx(cal_rob=cal_rob3, t_cal=t_noise0[0], t_test=t0)
    #
    # d = {'000': (unwrap_x(cal_rob=cal_rob0, x=x000), stats000),
    #      '003': (unwrap_x(cal_rob=cal_rob0, x=x003), stats003),
    #      '033': (unwrap_x(cal_rob=cal_rob0, x=x033), stats033),
    #      '333': (unwrap_x(cal_rob=cal_rob0, x=x333), stats333),
    #      '330': (unwrap_x(cal_rob=cal_rob0, x=x330), stats330),
    #      '300': (unwrap_x(cal_rob=cal_rob0, x=x300), stats300)
    #      }
    # save_file = f'{directory}/Justin19/f13_22/results/{"static_equ"}.npy'
    #
    # np.save(save_file, d)

    # x = wrap_x(x=x, cal_rob=cal_rob)
    # print(evaluate_x(cal_rob=cal_rob, x_list=[x], squared=False, q=q0_test, t=t_test))
    # tic()
    # x, stats = change_tuple_order(calibrate2(q_cal=q_cal, t_cal=t_cal, q_test=q_test, t_test=t_test, c=c, verbose=1)
    #                               for _ in range(20))
    # toc()



    # def calibrate_1_to_n(directory, idx_list):
    #
    #     q_cal = get_q(mode='unfiltered')
    #     q_test = get_q(mode='unfiltered-test')
    #
    #     _, _, t, t_noise, t_test = np.load(DLR_USERSTORE_PAPER_2020_CALIB + f"m_unfiltered_{name}.npy", allow_pickle=True)
    #
    #     fun = __calibrate_subset_wrapper(q_cal=q_cal, t_cal=t_noise[0], q_test=q_test, t_test=t_test,
    #                                      fun=calibrate2, verbose=0)
    #
    #     # fun(np.arange(50))
    #     n = len(idx_list)
    #     for i, idx in enumerate(idx_list):
    #         print_progress(i, n)
    #         calibrate_subsets(idx_list=idx, n_processes=10,
    #                           directory=f"{directory}/{i+1}/",
    #                           fun=fun,
    #                           verbose=0)
    #
    #
    # def calibrate_subsets_with_noises():
    #     # idx_list,
    #     #                               q_cal, t_cal_noises, q_test, t_test,
    #     #                               directory, n_processes=10):
    #     # _, q_cal, _, t_cal_noises = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'dummy_c_10000.npy', allow_pickle=True)
    #     # _, q_test, _, t_test = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'dummy_c_10000_test.npy', allow_pickle=True)
    #
    #     # idx_b, o, _ = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'Dummy_c/idx50_DetmaxBest_1000.npy', allow_pickle=True)
    #     # idx_w, o, _ = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'Dummy_/idx_DetmaxWorst_1000.npy', allow_pickle=True)
    #     # idx_r, o, _ = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'Dummy_c/idx50_Random_10000.npy', allow_pickle=True)
    #
    #     idx_list = np.load(DLR_USERSTORE_PAPER_2020_CALIB + 'idx_random_10000.npy')
    #     idx_list = idx_list[:100]
    #     directory = DLR_USERSTORE_PAPER_2020_CALIB + 'error_random_1000_noises1000/'
    #
    #     q_cal = get_q(mode='unfiltered')
    #     q_test = get_q(mode='unfiltered-test')
    #
    #     name = get_parameter_identifier(cal_rob=c)
    #     _, _, t, t_noise, t_test = np.load(DLR_USERSTORE_PAPER_2020_CALIB + f"m_unfiltered_{name}.npy", allow_pickle=True)
    #
    #     n_noises = len(t_noise)
    #     n_idx = len(idx_list)
    #
    #     for i in range(n_noises):
    #         fun = __calibrate_subset_wrapper(q_cal=q_cal, t_cal=t_noise[i], q_test=q_test, t_test=t_test,
    #                                          fun=calibrate2, verbose=0)
    #         print_progress(i, n_noises)
    #         calibrate_subsets(idx_list=idx_list, n_processes=10, verbose=0,
    #                           directory=f'{directory}/{i}/',
    #                           fun=fun)
    #
    #
    # def combine_calibrations_with_noises(directory=None, n=100):
    #
    #     directory = DLR_USERSTORE_PAPER_2020_CALIB + 'error_random_150/'
    #     n = 50
    #     dir_list = [directory + str(i+1) for i in range(n)]
    #
    #     # for dd in dir_list:
    #     #     combine_npy_files(dd, new_name='dummy')
    #
    #     idx, pars, stats, _ = change_tuple_order((np.load(f"{dd}/dummy.npy", allow_pickle=True) for dd in dir_list))
    #     # idx, pars, stats = np.atleast_1d(idx, pars, stats)
    #     from wzk import atleast_list
    #     idx, pars, stats = atleast_list(idx, pars, stats, convert=True)
    #
    #     for i in range(n):
    #         i_sort = np.lexsort(np.array(idx[i]).T[::-1])
    #         idx[i] = np.array(idx[i])[i_sort]
    #         pars[i] = np.array(pars[i])[i_sort]
    #         stats[i] = np.array(stats[i])[i_sort]
    #
    #     # Handle multiple same runs, should not be needed again
    #     for i, idx_i in enumerate(idx):
    #         assert np.allclose(idx[i][::2], idx[i][1::2])
    #
    #     print(np.max([stats[i][::2] - stats[i][1::2] for i in range(50)]))
    #     print(np.max([pars[i][::2] - pars[i][1::2] for i in range(50)]))
    #     #
    #     # idx = idx[::2]
    #     # stats = np.array(stats[::2])
    #     # pars = np.array(pars[::2])
    #
    #     np.save(directory + 'combined_noises.npy', (idx, pars, stats, []))


def filter_out_bad_samples():
    pass
    # TODO
