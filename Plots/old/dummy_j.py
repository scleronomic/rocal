import numpy as np
from wzk.mpl import (new_fig, save_fig, set_style, plt,
                     correlation_plot)

from rocal.definitions import ICHR20_CALIBRATION
from Justin.Calibration import greedy


from Justin.Calibration import setup
from Justin.Calibration import load_error_stats
from Justin.Calibration import plot_bins
from A_Plots.Calibration2020.util import true_best_with_noises

set_style(s=('ieee',))
fig_width_inch = 10

dir_files = ICHR20_CALIBRATION + 'Dummy_j/'
dir_figures = ICHR20_CALIBRATION + 'Figures/Dummy_j/'


opt_ta, _, opt_d, _ = setup(model='j29', cal_set='dummy', test_set='100000')
# opt_ta_f, _, opt_d_f, _ = setup(model='j', cal_set='1000test', test_set='100000')


file_rand_120 = dir_files + 'error_Random_1000_120.npy'
file_det_b_120 = dir_files + 'error_DetmaxBest_100_120.npy'
file_bins = dir_files + 'error_100Bins_10000.npy'
file_rand = dir_files + 'error_Random_10000.npy'
file_rand_f = dir_files + 'error_Random_filtered_10000.npy'
file_rand_b = dir_files + 'error_RandomBest_1000.npy'
file_rand_w = dir_files + 'error_RandomWorst_1000.npy'
file_det_b = dir_files + 'error_DetmaxBest_1000.npy'
file_det_b_f = dir_files + 'error_DetmaxBest_filtered_1000.npy'
file_det_w = dir_files + 'error_DetmaxWorst_1000.npy'
file_det_w_f = dir_files + 'error_DetmaxWorst_filtered_1000.npy'

file_rand_n = dir_files + 'error_Random_100_50noises.npy'
file_det_b_n = dir_files + 'error_DetmaxBest_100_50noises.npy'
file_det_w_n = dir_files + 'error_DetmaxWorst_100_50noises.npy'

# import os
# [__retrieve_idx_from_error_stats(directory+file, opt_ta) for file in os.listdir(directory) if '.npy' in file]

idx_bins, par_bins, err_mean_bins, err_max_bins = load_error_stats(file=file_bins)
idx_rand, par_rand, err_mean_rand, err_max_rand = load_error_stats(file=file_rand)
idx_rand_f, par_rand_f, err_mean_rand_f, err_max_rand_f = load_error_stats(file=file_rand_f)
idx_rand_b, par_rand_b, err_mean_rand_b, err_max_rand_b = load_error_stats(file=file_rand_b)
idx_rand_w, par_rand_w, err_mean_rand_w, err_max_rand_w = load_error_stats(file=file_rand_w)
idx_det_b, par_det_b, err_mean_det_b, err_max_det_b = load_error_stats(file=file_det_b)
idx_det_b_f, par_det_b_f, err_mean_det_b_f, err_max_det_b_f = load_error_stats(file=file_det_b_f)
idx_det_w, par_det_w, err_mean_det_w, err_max_det_w = load_error_stats(file=file_det_w)
idx_det_w_f, par_det_w_f, err_mean_det_w_f, err_max_det_w_f = load_error_stats(file=file_det_w_f)

# idx_, par_, err_mean_, err_max_ = load_error_stats(file=file_)
idx_rand_n, par_rand_n, err_mean_rand_n, err_max_rand_n = load_error_stats(file=file_rand_n)
idx_det_b_n, par_det_b_n, err_mean_det_b_n, err_max_det_b_n = load_error_stats(file=file_det_b_n)
idx_det_w_n, par_det_w_n, err_mean_det_w_n, err_max_det_w_n = load_error_stats(file=file_det_w_n)

ota_rand_1e8 = np.load(dir_files + 'A-Opt_1e7.npy', allow_pickle=True)[1]
ota_bins = opt_ta(idx_bins)
ota_rand = opt_ta(idx_rand)
ota_rand_b = opt_ta(idx_rand_b)
ota_rand_w = opt_ta(idx_rand_w)
ota_det_b = opt_ta(idx_det_b)
ota_det_w = opt_ta(idx_det_w)

ota_rand_n = opt_ta(idx_rand_n[0])
ota_det_b_n = opt_ta(idx_det_b_n[0])
ota_det_w_n = opt_ta(idx_det_w_n[0])

# ota_det_b_f = opt_ta_f(idx_det_b_f)
# ota_det_w_f = opt_ta_f(idx_det_w_f)
# ota_rand_f = opt_ta_f(idx_rand_f)


def unpack_120(f, n):
    d = np.load(f, allow_pickle=True)
    ota = np.empty((20, n))
    err = np.empty((20, n))
    for i in range(20):
        err[i] = d[2][i][:, 0, 0, 0]
        ota[i] = opt_ta(d[0][i])

    return d[0], err, ota


idx_rand_120, err_mean_rand_120, ota_rand_120 = unpack_120(file_rand_120, n=1000)
idx_det_b_120, err_mean_det_b_120, ota_det_b_120 = unpack_120(file_det_b_120, n=100)


def noises50_1oversqrtn():
    n = 50
    rm = np.array([err_mean_rand_n[:i+1].mean(axis=0) for i in range(50)])

    fig, ax = new_fig()

    rm_diff = rm - rm[-1]
    rm_diff_abs = np.abs(rm - rm[-1])
    print(rm.shape)
    print(rm[-1].shape)
    ax.plot(np.arange(1, n+1)[:, np.newaxis].repeat(100, axis=1), rm_diff_abs, alpha=0.2, color='k',
            label='Absolute Difference to N50')
    ax.plot(np.arange(1, n+1), rm_diff_abs.mean(axis=-1), color='b', label='Mean over 100')
    ax.plot(np.arange(1, n+1), 1 / np.sqrt(np.arange(1, n+1))*rm_diff_abs.mean(axis=-1)[0], color='r',
            label='1 / n^(1/2) * alpha')

    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    save_fig(fig=fig, filename=dir_figures + '50noises_1_over_sqrtn',
             bbox=None, formats=('png', 'pdf'))


def noises50():

    rm = err_mean_rand_n.mean(axis=0)
    bm = err_mean_det_b_n.mean(axis=0)
    wm = err_mean_det_w_n.mean(axis=0)

    print(rm.mean())
    print(bm.mean())
    print(wm.mean())
    rstd = err_mean_rand_n.std(axis=0)
    bstd = err_mean_det_b_n.std(axis=0)
    wstd = err_mean_det_w_n.std(axis=0)

    fig, ax = new_fig()
    ax.plot(np.sqrt(ota_rand_n), rm, ls='', marker='o', color='k', alpha=0.8, label='Random')
    ax.plot(np.sqrt(ota_det_b_n), bm, ls='', marker='o', color='g', alpha=0.8, label='Detmax')
    # ax.plot(ota_det_w_n, wm, ls='', marker='o', color='r')

    ax.plot(np.vstack((np.sqrt(ota_rand_n),)*2), np.vstack((rm-rstd, rm+rstd)), color='k', alpha=0.2)
    ax.plot(np.vstack((np.sqrt(ota_det_b_n),)*2), np.vstack((bm-bstd, bm+bstd)), color='g', alpha=0.2)
    # ax.plot(np.vstack((ota_det_w_n,)*2), np.vstack((wm-wstd, wm+wstd)), color='r')

    ax.hlines(rm.mean(), xmin=np.sqrt(ota_rand_n.min()), xmax=np.sqrt(ota_rand_n.max()), color='k', zorder=10)
    ax.hlines(bm.mean(), xmin=np.sqrt(ota_det_b_n.min()), xmax=np.sqrt(ota_det_b_n.max()), color='k', zorder=10)

    ax.vlines(np.sqrt(ota_rand_n.min()/2+ota_rand_n.max()/2), ymin=rm.mean()-rstd.mean(), ymax=rm.mean()+rstd.mean(),
              color='k', zorder=10)
    ax.vlines(np.sqrt(ota_det_b_n.min()/2+ota_det_b_n.max()/2), ymin=bm.mean()-bstd.mean(), ymax=bm.mean()+bstd.mean(),
              color='k', zorder=10)
    # a + (b - a) / 2
    # Mark optimum
    i_r = np.argmin(rm)
    ax.plot(np.sqrt(ota_rand_n[i_r]), rm[i_r], ls='', marker='o', color='b', alpha=0.8, label='Minimum')
    ax.plot(np.vstack((np.sqrt(ota_rand_n[i_r]),)*2), np.vstack((rm[i_r]-rstd[i_r], rm[i_r]+rstd[i_r])),
            color='b', alpha=0.8)

    i_b = np.argmin(bm)
    ax.plot(np.sqrt(ota_det_b_n[i_b]), bm[i_b], ls='', marker='o', color='b', alpha=0.8)
    ax.plot(np.vstack((np.sqrt(ota_det_b_n[i_b]),)*2), np.vstack((bm[i_b]-bstd[i_b], bm[i_b]+bstd[i_b])),
            color='b', alpha=0.8)

    ax.set_xlabel('Task A-Optimality')
    ax.set_ylabel('Mean TCP Error')
    ax.legend()
    save_fig(fig=fig, filename=dir_figures + '50noises',
             bbox=None, formats=('png', 'pdf'))


def bins():
    fig, ax = new_fig(width=fig_width_inch)
    plot_bins(ax=ax, o_bins=np.sqrt(ota_bins), err=err_mean_bins, kwargs=dict(color='k', marker='o'),
              kwargs_std=dict(alpha=0.2, color='k'))
    save_fig(file=dir_figures + 'dummy_j_bins', formats=('png', 'pdf'),
             bbox=None)


def onetwenty():
    fig, ax = new_fig(width=fig_width_inch)
    ax.semilogy(np.arange(1, 21), err_mean_rand_120.mean(axis=-1) * 100,
                color='b', marker='s', ls='-', label='Random: Mean TCP Error x 100')
    ax.semilogy(np.arange(1, 21), ota_rand_120.mean(axis=-1), color='r', marker='o', ls='-',
                label='Random: Task A-Optimality')

    ax.semilogy(np.arange(1, 21), err_mean_det_b_120.mean(axis=-1) * 100,
                color='matrix', marker='s', ls='-', label='Detmax: Mean TCP Error x 100')
    ax.semilogy(np.arange(1, 21), ota_det_b_120.mean(axis=-1), color='c', marker='o', ls='-',
                label='Detmax: Task A-Optimality')

    ax.set_ylim(0.5, 200)
    ax.set_xlim(0.5, 20.5)
    ax.legend()
    ax.set_xlabel('|Calibration Set|')

    ax.set_xticks(np.arange(1, 21, 1))
    ax.set_xticklabels([i if i % 2 == 1 else '' for i in np.arange(1, 21, 1)])
    save_fig(fig=fig, filename=dir_figures + 'dummy_j_120',
             bbox=None, formats=('png', 'pdf'))


def onetwenty_animation():
    fig, ax = new_fig()
    for i in range(20):
        ax.clear()
        ax.set_xlim(0, 0.1)
        ax.hist(err_mean_rand_120[i].ravel(), bins=50, density=True)
        plt.pause(0.1)


def ota_hist():

    ota_greedy = greedy(n=10000, k=10, fun=opt_ta)[1]
    nota_greedy = -greedy(n=10000, k=10, fun=lambda idx: -opt_ta(idx))[1]

    fig, ax = new_fig(width=fig_width_inch)
    # fig, ax = new_fig(fig_width_inch='ieee1c')
    ax.hist(np.sqrt(ota_rand_1e8), bins=1000, density=True, label='Random (1e8)')
    ax.hist(np.sqrt(ota_det_b), bins=50, density=True, label='Detmax Best (1e3)')
    ax.hist(np.sqrt(ota_det_w), bins=50, density=True, label='Detmax Worst (1e3)')
    ax.vlines(np.sqrt(ota_greedy), ymin=0, ymax=5, color='g', lw=2, label='Greedy Best (1)')
    ax.vlines(np.sqrt(nota_greedy), ymin=0, ymax=5, color='matrix', lw=2, label='Greedy Worst (1)')
    # ax.vlines(nota_greedy, ymin=0, ymax=5, color='r', lw=2, label='Greedy worst')
    ax.set_xlabel('Task- A-Optimality')
    ax.legend()

    save_fig(fig=fig, filename=dir_figures + 'dummy_j_ota_hist',
             bbox=None, formats=('png', 'pdf'))


def correlation():

    labels = ['Random', 'DetmaxBest', 'DetmaxWorst', 'RandomBest', 'RandomWorst', 'Regression Line (R={:.4})']
    colors = ['gray', 'g', 'r', 'y', 'matrix', 'k']
    markers = ['x', 'o', 'o', 's', 's', '-']
    a = (ota_rand, ota_det_b, ota_det_w, ota_rand_b, ota_rand_w)
    a = (np.sqrt(ota_rand),
         np.sqrt(ota_det_b),
         np.sqrt(ota_det_w),
         np.sqrt(ota_rand_b),
         np.sqrt(ota_rand_w))
    b = (err_mean_rand, err_mean_det_b, err_mean_det_w, err_mean_rand_b, err_mean_rand_w)

    fig, ax = new_fig(width=fig_width_inch)
    ax = correlation_plot(ax=ax, a=a, b=b, name_a='Task A-Optimality', name_b='Mean TCP Error [matrix]',
                          labels=labels, colors=colors, markers=markers,

                          regression_line=True)
    ax.set_xlim(0.8, 5)
    ax.set_ylim(0.005, 0.08)
    save_fig(file=dir_figures + 'dummy_j_correlation', fig=fig,
             bbox=None, formats=('png', 'pdf'))


def full_vs_filtered():
    fig, ax = new_fig(width=fig_width_inch)

    labels = ['Random', 'DetmaxBest', 'Random Filtered', 'DetmaxBest Filtered']
    colors = ['b', 'b', 'r', 'r']
    markers = ['x', 'x', 'x', 'x']
    alphas = [0.2, 0.2, 0.2, 0.2]

    a = (ota_rand, ota_det_b, ota_det_w, ota_rand_b, ota_rand_w)
    a = (np.sqrt(ota_rand),
         np.sqrt(ota_det_b),
         np.sqrt(ota_rand_f),
         np.sqrt(ota_det_b_f))
    b = (err_mean_rand, err_mean_det_b, err_mean_rand_f, err_mean_det_b_f)

    fig, ax = new_fig(width=fig_width_inch)
    ax = correlation_plot(ax=ax, a=a, b=b, name_a='Task A-Optimality', name_b='Mean TCP Error [matrix]',
                          labels=labels, colors=colors, markers=markers, alphas=alphas,
                          regression_line=False)
    ax.set_xlim(0.8, 2.5)
    ax.set_ylim(0.005, 0.08)
    save_fig(file=dir_figures + 'dummy_j_filtered_correlation', fig=fig,
             bbox=None, formats=('png', 'pdf'))



# bins()
# onetwenty()
# ota_hist()
correlation()
# full_vs_filtered()
# noises50()
true_best_with_noises(err_r=err_mean_rand_n, err_b=err_mean_det_b_n, obj_r=ota_rand_n, obj_b=ota_det_b_n,
                      save_dir=dir_figures)
# noises50_1oversqrtn()
