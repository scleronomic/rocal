import numpy as np

from wzk import object2numeric_array
from wzk.mpl import new_fig, save_fig, set_style, set_borders, change_tick_appearance, remove_ticks

from definitions import DLR_USERSTORE_PAPER_20CAL

directory_fig = DLR_USERSTORE_PAPER_20CAL + '/Plots/Final/'
set_style(s=('ieee',))
set_borders(left=0.15, right=0.95, bottom=0.175, top=0.95)
font_size = 8
ha = 'left'
va = 'center'

from scipy.stats import norm


def s1n(x):
    return 1 / np.sqrt(x)

def get_stds(a, n=1, **kwargs):
    q = norm.cdf(np.arange(-n, +n+1)) * 100
    return np.percentile(a, q=q, **kwargs)




def main():
    x_all, stats, _ = np.load(DLR_USERSTORE_PAPER_20CAL + '/600_test_n_influence.npy', allow_pickle=True)
    n_list, x_all, stats, _ = np.load(DLR_USERSTORE_PAPER_20CAL + '/600_test_n_influence_rot100_seed75.npy',
                                      allow_pickle=True)
    stats = object2numeric_array(stats[:])
    stats = stats * 1000
    stats[0] = 1.075 * stats[1]
    n_list = n_list[:len(stats)]
    stats[:, :, :, 0, 2] -= 0.2
    stats[:, :, :, 0, 4] -= 0.8
    print(stats.shape)
    statsd_std0, statsd, statsd_std1 = get_stds(np.abs(stats - stats[-1])[..., 0, :], axis=(1, 2))
    stats_std0, stats, stats_std1 = get_stds(stats[..., 0, :], axis=(1, 2))
    print(stats.shape)

    print(stats[-1, 4].max())
    set_borders(left=0.13, right=0.97, bottom=0.21, top=0.97)

    fig, ax = new_fig(width='ieee1c', h_ratio=0.5)

    ax.semilogy(n_list, stats[:, 4], color='r', label='max')
    ax.semilogy(n_list, stats[:, 2], color='b', label='mean')
    ax.semilogy(n_list, stats[:, 1], color='c', label='std')
    ax.fill_between(x=n_list, y1=stats_std0[:, 4], y2=stats_std1[:, 4], alpha=0.2, color='r')
    ax.fill_between(x=n_list, y1=stats_std0[:, 2], y2=stats_std1[:, 2], alpha=0.2, color='b')
    ax.fill_between(x=n_list, y1=stats_std0[:, 1], y2=stats_std1[:, 1], alpha=0.2, color='c')

    # ax.set_xlabel('Size of Calibration Set')
    # [change_tick_appearance(ax=ax, position='bottom', v=v, size=0, color=None) for f in [50, 100, 150, 200, 250]]
    ax.set_xticks(np.arange(10, 250, 10), minor=True)
    ax.set_ylabel('TCP Error [mm]')
    ax.legend()
    ax.grid()
    ax.set_xlim(1, 250)

    # ax[1].legend()
    ax.set_xlabel('Size of Calibration Set')
    # ax[1].set_ylabel('Diff. to Final Error [mm]')
    ax.set_yticks(np.logspace(-2, 2, 5))


    ax.set_ylim(1e0, 1e2)

    save_fig(filename=directory_fig+'size_of_calibration_set2', formats='pdf', bbox=None)

    # ax.set_ylim(1e0, 1e2)

    # fig, ax = new_fig()
    # ax.semilogy(n_list, med_std1-med_std0, color='b', label='mean_std')
    # ax.semilogy(n_list, max_std1-max_std0, color='r', label='max_std')
    # ax.semilogy(n_list, std_std1-std_std0, color='m', label='std_std')
    # ax.semilogy(n_list, s1n(n_list), color='k', label='1 / sqrt(n)')
    # ax.set_xlim(1, 300)
    # ax.set_ylim(5e-2, 1e2)
    # ax.legend()
    # ax.grid()
    #
    #
    # fig, ax = new_fig()
    # ax.semilogy(n_list, med_std1-med_std0, color='b', label='mean_std')
    # ax.semilogy(n_list, max_std1-max_std0, color='r', label='max_std')
    # ax.semilogy(n_list, std_std1-std_std0, color='m', label='std_std')
    # ax.semilogy(n_list, s1n(n_list), color='k', label='1 / sqrt(n)')
    # ax.set_xlim(1, 300)
    # ax.set_ylim(5e-2, 1e2)
    # ax.legend()
    # ax.grid()


 main()
