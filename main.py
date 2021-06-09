import numpy as np

from calibration import calibrate, unwrap_x
from Measurements.io2 import get_q, get_parameter_identifier

from justin import Justin19Calib
from definitions import ICHR20_CALIBRATION

from wzk.spatial import trans_rotvec2frame


# TODO check_marker_occlusion(q)
# TODO check for duplicate indices
# TODO clip the 1000 bestworst because only a small portion is at the extremes
# FINDING 10000 test configurations is easily enough to have a small variance


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


def search_world_frame(_cal_rob, q, t):
    # FINDING its hard / impossible to find the correct world frame if it the initial guess is far off
    #   -> try multi start and safe the good fits
    n = 100
    threshold = 0.1  # m
    for i in range(n):
        x, stats = calibrate(q_cal=q, t_cal=t, q_test=q, t_test=t, x0_noise=0.1, verbose=0,
                             cal_rob=_cal_rob, cal_par=cal_par)
        if stats[0, 0, 0] < threshold:
            print(stats[:, 0, 0].meand(axis=0))
            print(trans_rotvec2frame(trans=x[:3], rotvec=x[3:6]))


if __name__ == '__main__':
    pass
    # test_seed2()
    # iterative_mass_compliance()
    # evaluate_different_effects©()
    # leave_one_out_analysis_joints()
    # test_static_equilibrium_truncation()

    cal_rob = Justin19Calib(dcmf='000c', ma0=True, fr0=True, use_imu=False, cp_loop=1)

    directory = ICHR20_CALIBRATION + '/Measurements/600'
    (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test) = get_q(cal_rob=cal_rob, split=-1, seed=75)
    import pandas as pd
    df = pd.DataFrame([(q, t) for q, t in zip(q0_cal, t_cal[:, :, :3, -1])], columns=('joints', 'markers'))

    print(df.shape)
    x, stats = calibrate(q_cal=q0_cal, t_cal=t_cal, q_test=q0_test, t_test=t_test, verbose=1,
                         cal_par=cal_par, cal_rob=cal_rob, x0_noise=0)
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

