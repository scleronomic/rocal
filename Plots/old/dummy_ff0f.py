import numpy as np
from wzk.mpl import (new_fig, save_fig, set_style)

from rocal.definitions import ICHR20_CALIBRATION

from Justin.Calibration.OptimalDesign.get_calibration_sets import setup
# from Justin.Calibration import greedy
# from Justin.Calibration import load_error_stats
# from Justin.Calibration import plot_bins
# from A_Plots.Calibration2020.util import true_best_with_noises

set_style(s=('ieee',))
fig_width_inch = 10

dir_files = ICHR20_CALIBRATION + 'ff0f_11/'
dir_figures = ICHR20_CALIBRATION + 'Figures/ff0f_11/'


opt_ta, _, opt_d, _ = setup(model='j29', cal_set='dummy', test_set='100000')
# opt_ta_f, _, opt_d_f, _ = setup(model='j', cal_set='1000test', test_set='100000')


file_rand_120 = dir_files + 'error_random_150.npy'


# import os
# [__retrieve_idx_from_error_stats(directory+file, opt_ta) for file in os.listdir(directory) if '.npy' in file]


def unpack_stair(f, n_idx, n_stair):
    idx, pars, stats, [] = np.load(f, allow_pickle=True)

    # for i, idx_i in enumerate(idx):
    #     i_sort = np.lexsort(np.array(idx_i).T[::-1])
    #     idx[i] = np.array(idx_i)[i_sort]
    #     pars[i] = np.array(pars[i])[i_sort]
    #     stats[i] = np.array(stats[i])[i_sort]
    #
    # # Handle multiple same runs, should not be needed again
    # for i, idx_i in enumerate(idx):
    #     assert np.allclose(idx[i][::2], idx[i][1::2])
    #
    # print(np.max([stats[i][::2] - stats[i][1::2] for i in range(len(stats))]))
    # print(np.max([pars[i][::2] - pars[i][1::2] for i in range(len(pars))]))
    # #
    # idx = idx[::2]
    # stats = np.array(stats[::2])
    # pars = np.array(pars[::2])


    ota = np.empty((n_stair, n_idx))
    err = np.empty((n_stair, n_idx))
    for i in range(n_stair):
        err[i] = stats[i][:, :, 0, 0].mean(axis=1)
        ota[i] = opt_ta(idx[i])

    return idx, err, ota


idx_rand_stair, err_mean_rand_stair, ota_rand_stair = unpack_stair(file_rand_120, n_stair=25, n_idx=200)
# idx_det_b_120, err_mean_det_b_120, ota_det_b_120 = unpack_120(file_det_b_120, n=100)


def stair(n_stair):

    print('Final Error:', err_mean_rand_stair.mean(axis=-1)[-1])
    fig, ax = new_fig(width=fig_width_inch)
    ax.semilogy(np.arange(1, n_stair+1), err_mean_rand_stair.mean(axis=-1),
                color='b', marker='s', ls='-', label='Random: Mean TCP Error')
    ax.semilogy(np.arange(1, n_stair+1), ota_rand_stair.mean(axis=-1), color='r', marker='o', ls='-',
                label='Random: Task A-Optimality')

    ax.legend()
    ax.set_xlabel('|Calibration Set|')

    ax.set_xticks(np.arange(1, 21, 1))
    ax.set_xticklabels([i if i % 2 == 1 else '' for i in np.arange(1, 21, 1)])
    save_fig(fig=fig, filename=dir_figures + f'stair{1}->{n_stair}',
             bbox=None, formats=('png', 'pdf'))

stair(n_stair=25)


