import numpy as np

from wzk.spatial import apply_noise, trans_rotvec2frame, frame_difference
from wzk.random2 import noise

from rocal.calibration import kinematic, set_bool_dict_false, get_active_parameters
# noinspection PyUnresolvedReferences
from rocal.Measurements.io2 import load_q, save_m
from rocal.Plots.util_plotting import hist_frame_difference


def get_dummy_parameters(cal_rob,
                         dh_scale=0.01, cp_scale=0.01, ma_scale=1., fr_scale=1.,
                         noise_mode='plusminus'):

    dh_bool, cp_bool, ma_bool, fr_bool = get_active_parameters(cal_rob=cal_rob)
    dh = np.zeros_like(dh_bool, dtype=float)
    cp = np.zeros_like(cp_bool, dtype=float)
    ma = np.zeros_like(ma_bool, dtype=float)
    fr = np.zeros_like(fr_bool, dtype=float)

    dh[dh_bool] = noise(scale=dh_scale, shape=int(dh_bool.sum()), mode=noise_mode)
    cp[cp_bool] = noise(scale=cp_scale, shape=int(cp_bool.sum()), mode=noise_mode)
    ma[ma_bool] = noise(scale=ma_scale, shape=int(ma_bool.sum()), mode=noise_mode)
    fr[fr_bool] = noise(scale=fr_scale, shape=int(fr_bool.sum()), mode=noise_mode)

    fr = trans_rotvec2frame(trans=fr[:, :3], rotvec=fr[:, 3:])

    return dict(dh=dh, cp=cp, ma=ma, fr=fr)


def create_dummy_measurements(*, cal_rob, x_dict, q,
                              trans_noise=0.0005, rot_noise=0.002, n_noise=100,
                              verbose=0):

    t = kinematic(cal_rob=cal_rob, q=q, **x_dict)

    if verbose > 0:
        x_dict0 = set_bool_dict_false(x_dict=x_dict)
        x_dict0['fr'] = x_dict['fr']  # or np.eye
        t0 = kinematic(cal_rob=cal_rob, q=q, **x_dict0)
        trans_diff0, rot_diff0 = frame_difference(t0, t)

        print(f"Difference to Nominal: "
              f"Trans={1000*trans_diff0.mean(axis=0)}mm, "
              f"Rot={np.rad2deg(rot_diff0.mean(axis=0))}deg")

        if verbose > 1:
            [hist_frame_difference(diff_trans=trans_diff0[:, i], diff_rot=rot_diff0[:, i],
                                   title=f'Frame_{i} vs. Nominal') for i in range(t0.shape[1])]

    # Create different measurement noises
    t_wo_noise = np.tile(t[np.newaxis, ...], reps=(n_noise, 1, 1, 1, 1))
    t_noise = apply_noise(frame=t_wo_noise, trans=trans_noise, rot=rot_noise, mode='normal')
    return t, t_noise


def value_analysis_cp():
    from rocal.definitions import ICHR20_CALIBRATION

    directory = ICHR20_CALIBRATION
    # x_dict0 = np.load(f"{directory}/Measurements/600/results/Justin19_cc0c_0_11_ff.npy", allow_pickle=True)[0]
    # x_dict3 = np.load(f"{directory}/Measurements/600/results/Justin19_cc0c_3_11_ff.npy", allow_pickle=True)[0]
    x_dict0_imu = np.load(f"{directory}/Measurements/600/results/Justin19_cc0c_0_11_ff_imu.npy", allow_pickle=True)[0]
    x_dict3_imu = np.load(f"{directory}/Measurements/600/results/Justin19_cc0c_3_11_ff_imu.npy", allow_pickle=True)[0]
    # cp0, cp3 = x_dict0['cp'], x_dict3['cp']
    cp0_imu, cp3_imu = x_dict0_imu['cp'], x_dict3_imu['cp']

    # print(np.round(cp0 / 100 * 1000, 3))
    # print(np.round(cp0_imu / 100 * 1000, 3))
    print(np.round(cp3_imu / 100 * 1000, 3))
    # print(np.round(cp3 / 100 * 1000, 3))

    # joint stiffness torso 1e-2, 1e-4, 1e-2
    # traversal 3e-3


def main_create_dummy_measurements():
    from rocal.definitions import ICHR20_CALIBRATION
    from rocal.Robots.Justin19 import Justin19Cal
    directory = ICHR20_CALIBRATION

    dcmf = 'cc0c'
    cal_rob_ff = Justin19Cal(dcmf=dcmf, cp_loop=30, config_filter='ff', use_imu=False, add_nominal_offsets=True,
                             fr0=False)
    # cal_rob_nf = Justin19Calib(dcmf=dcmf, cp_loop=30, config_filter='nf', use_imu=False, add_nominal_offsets=True,
    #                            fr0=False)
    x_dict = np.load(f"{directory}/Measurements/600/results/Justin19_cc0c_0_11_ff.npy", allow_pickle=True)[0]
    x_dict['dh'][:] = 0
    x_dict['ma'][:] = 0
    x_dict['fr'][:] = np.eye(4)
    print(x_dict['cp'])
    x_dict['cp'] *= 20
    # q_ff = load_q(directory=directory, cal_rob=cal_rob_ff)
    # q_nf = load_q(directory=directory, cal_rob=cal_rob_nf)
    # q_nf, q_ff
    np.random.seed(0)
    q_ff = cal_rob_ff.sample_q(1000)
    t, t_noise = create_dummy_measurements(cal_rob=cal_rob_ff, q=q_ff, x_dict=x_dict, n_noise=100, verbose=0)

    # tt = kinematic(cal_rob=cal_rob_ff, q=q_ff, **x_dict)
    # t_test, _ = create_dummy_measurements(cal_rob=cal_rob_nf, q=q_nf, x_dict=x_dict, n_noise=100, verbose=0)
    save_m(directory=directory, cal_rob=cal_rob_ff, arr=(x_dict, t, t_noise))
    # from wzk.mpl import new_fig
    # fig, ax = new_fig(n_dim=3)
    # ax.plot(*t[..., 0, :3, -1].reshape(-1, 3).T, ls='', marker='o', alpha=0.2)
    # ax.plot(*t[..., 1, :3, -1].reshape(-1, 3).T, ls='', marker='o', alpha=0.2)

    return x_dict, t, t_noise


if __name__ == '__main__':
    # value_analysis_cp()
    ttt = main_create_dummy_measurements()
    # r = np.linalg.norm((ttt[-1:] - ttt[:-1])[..., :, :3, -1], axis=-1)
    # r_mean = r.mean(axis=(-1, -2)) * 1000
    # r_max = r.max(axis=(-1, -2)) * 1000
    # r_std = r.std(axis=(-1, -2)) * 1000
    #
    # from wzk.mpl import new_fig
    # fig, ax = new_fig()
    # ax.semilogy(r_mean, label='mean')
    # ax.semilogy(r_max, label='max')
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Mean Difference to Converged Poses [mm]')
    # ax.legend()

# TODO check the difference if jac_sum_sum in a optimality is filtered with self collision or not
# TODO play with different noises / scales
# TODO sample more often and plot results

# Difference to Nominal: Trans=[0.04548804 0.04398301]m,  Rot=[2.47169378 2.62497576]deg  # uf
# Difference to Nominal: Trans=[0.0511411  0.04693086]m,  Rot=[2.25970183 2.84358699]deg  # f
# There is quite a large variance in the difference to the nominal robot when sampling the calibration parameters
# and those parameters are more prominent in different poses, ie one can also see an consistent difference to
# the filtered poses
