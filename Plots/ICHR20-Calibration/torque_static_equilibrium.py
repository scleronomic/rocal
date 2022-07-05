import numpy as np

from rocal.calibration import (kinematic, unwrap_x)
from rocal.Robots.Justin19 import Justin19Cal

from rocal.definitions import ICHR20_CALIBRATION
from wzk.mpl import new_fig, save_fig, set_style, set_borders

directory = ICHR20_CALIBRATION

directory_fig = ICHR20_CALIBRATION + '/Plots/Final/'
set_style(s=('ieee',))
set_borders(left=0.15, right=0.95, bottom=0.175, top=0.95)
font_size = 8
ha = 'left'
va = 'center'

# Main
fig, ax = new_fig(width='ieee1c',)


res = np.load(f"{directory}/Justin19/f13_22/static_equ_test0/results.npy", allow_pickle=True).item()
stats = np.array([res['softer20'][1], res['softer5'][1], res['m'][1], res['stiffer5'][1], res['stiffer20'][1]])

# med1 = stats[:, 0, :, 0, 2].mean(axis=-1) * 1000
med1 = np.array([1.5818e+02, 7.0572e+00, 2.7401e-01, 1.172e-02, 0.804e-03]) * 1.1


def plot():

    n_samples = 1000
    np.random.seed(0)
    n_iter = 10
    n_iter_show = 5

    cal_rob = Justin19Cal(dkmc='0c00', el_loop=0, config_filter='ff', use_imu=False, fr0=False,
                          add_nominal_offsets=True)
    x_dict = np.load(f"{directory}/Measurements/600/results/Justin19_cc0c_0_11_ff.npy", allow_pickle=True)[0]
    el = x_dict['el']
    el = el[el != 0]

    cp_list = [el*20, el*5, el, el/5, el/20]
    color_list = ['0.5', '0.25', 'b', '0.25', '0.5']
    name_list = ['x20 softer', 'x5 softer', 'real robot', 'x5 stiffer', 'x20 stiffer']
    marker_list = ['^', '^', 'o', 'v', 'v']

    # n_tests = len(cp_list)
    y_lim = (1e-4, 1e3)

    np.random.seed(0)
    q = cal_rob.sample_q(n_samples)
    x = np.zeros_like(el)

    t_list = []
    for i in cp_list:
        if isinstance(i, float):
            x[:] = i * np.sign(el)
        else:
            x[:] = i

        tt = []
        for j in range(n_iter):
            cal_rob.el_loop = j
            tt.append(kinematic(cal_rob=cal_rob, q=q, **unwrap_x(cal_rob=cal_rob, x=x)))
        t_list.append(tt)

    t_list = np.array(t_list)
    # order zero is to completely leave out the compliance part
    # x[:] = 0
    # t_list = np.concatenate([kinematic(cal_rob=cal_rob, q=q, **unwrap_x(cal_rob=cal_rob, x=x)
    #                                    )[np.newaxis, np.newaxis].repeat(n_tests, axis=0),
    #                          t_list], axis=1)

    r = np.linalg.norm((t_list[:, -1:] - t_list[:, :-1])[..., :3, -1], axis=-1)

    r *= 1000  # to mm
    r[r < 1e-6] = 1e-6
    r_med = np.percentile(r, q=50, axis=(-1, -2))
    r_std0 = np.percentile(r, q=15.9, axis=(-1, -2))
    r_std1 = np.percentile(r, q=84.1, axis=(-1, -2))

    for i in range(len(cp_list)):
        ax.semilogy(r_med[i], color=color_list[i], marker=marker_list[i], label=name_list[i], markersize=3)
        ax.fill_between(x=np.arange(n_iter-1), y1=r_std0[i], y2=r_std1[i], color=color_list[i], alpha=0.2)

    # ax.plot(100, 1, color='k', ls='', marker='x', markersize=3, label='calibrated')
    # for i in range(len(cp_list)):
    #     ax.plot(1, med1[i], color=color_list[i], ls='', marker='x', markersize=3)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Difference to Converged Poses [mm]')
    ax.set_xticks(np.arange(n_iter_show))
    # ax.set_yticks([1e-4, 1e-2, 1e+0, 1e2], minor=True)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlim((0, n_iter_show-1))
    ax.set_ylim(y_lim)
    ax.grid()
    ax.legend(loc='upper right', framealpha=0.5)

    save_fig(file=directory_fig + 'torque_equilibrium', fig=fig, formats='pdf', bbox=None)


plot()

