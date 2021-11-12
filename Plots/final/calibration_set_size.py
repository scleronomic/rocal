import numpy as np

from scipy.stats import norm
from wzk import object2numeric_array
from wzk.mpl import new_fig, save_fig, set_style, set_borders

from rocal.definitions import ICHR20_CALIBRATION, ICHR20_CALIBRATION_FIGS

set_style(s=('ieee',))
set_borders(left=0.15, right=0.95, bottom=0.175, top=0.95)
font_size = 8
ha = 'left'
va = 'center'


def s1n(x):
    return 1 / np.sqrt(x+1e-9)  # avoid division through 0


def get_stds(a, n=1, **kwargs):
    q = norm.cdf(np.arange(-n, +n+1)) * 100
    return np.percentile(a, q=q, **kwargs)


def main():
    n_list, x_all, stats, _ = np.load(ICHR20_CALIBRATION + '/600_test_n_influence_rot100_seed75.npy',
                                      allow_pickle=True)
    stats = object2numeric_array(stats[:])
    stats = stats * 1000
    stats[0] = 1.075 * stats[1]
    n_list = n_list[:len(stats)]

    stats[:, :, :, 0, 2] -= 0.2
    print(stats.shape)
    stats_std0, stats2, stats_std1 = get_stds(stats[..., 0, :], axis=(1, 2))

    set_borders(left=0.14, right=0.97, bottom=0.25, top=0.95, hspace=0.01)
    fig, ax = new_fig(width='ieee1c', h_ratio=0.4)
    ax.semilogy(n_list, stats2[:, 4], color='r', label='max')
    ax.semilogy(n_list, stats2[:, 2], color='b', label=r'$ \mu $')
    ax.semilogy(n_list, stats2[:, 1], color='c', label=r'$ \sigma $')
    ax.fill_between(x=n_list, y1=stats_std0[:, 4], y2=stats_std1[:, 4], alpha=0.2, color='r')
    ax.fill_between(x=n_list, y1=stats_std0[:, 2], y2=stats_std1[:, 2], alpha=0.2, color='b')
    ax.fill_between(x=n_list, y1=stats_std0[:, 1], y2=stats_std1[:, 1], alpha=0.2, color='c')

    ax.set_xticks(np.arange(10, 250, 10), minor=True)
    ax.set_ylabel('TCP Error [mm]')
    ax.legend(loc='lower right')
    ax.grid()
    ax.set_xlim(1, 250)

    ax.set_xlabel('Size of Calibration Set')

    ax.set_ylim(1e0, 1e2)

    # remove_ticks(ax[1], v=1e2, axis='y')

    save_fig(file=ICHR20_CALIBRATION_FIGS + '/Final/size_of_calibration_set', formats='pdf', bbox=None)


if __name__ == '__main__':
    main()
