import numpy as np

from scipy.stats import norm
from wzk import object2numeric_array
from wzk.mpl import new_fig, save_fig, set_style, set_borders, remove_ticks

from definitions import ICHR20_CALIBRATION, ICHR20_CALIBRATION_FIGS

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
    statsd_std0, statsd, statsd_std1 = get_stds(np.abs(stats - stats[-1])[..., 0, :], axis=(1, 2))
    stats_std0, stats2, stats_std1 = get_stds(stats[..., 0, :], axis=(1, 2))

    set_borders(left=0.14, right=0.97, bottom=0.15, top=0.97, hspace=0.01)
    fig, ax = new_fig(width='ieee1c', h_ratio=0.65, n_rows=2, share_x=True)
    ax[0].semilogy(n_list, stats2[:, 4], color='r', label='max')
    ax[0].semilogy(n_list, stats2[:, 2], color='b', label=r'$ \mu $')
    ax[0].semilogy(n_list, stats2[:, 1], color='c', label=r'$ \sigma $')
    ax[0].fill_between(x=n_list, y1=stats_std0[:, 4], y2=stats_std1[:, 4], alpha=0.2, color='r')
    ax[0].fill_between(x=n_list, y1=stats_std0[:, 2], y2=stats_std1[:, 2], alpha=0.2, color='b')
    ax[0].fill_between(x=n_list, y1=stats_std0[:, 1], y2=stats_std1[:, 1], alpha=0.2, color='c')
    ax[0].semilogy(np.arange(n_list[-1]+1), s1n(np.arange(n_list[-1]+1))/100, color='k', label=r'$ 1 / \sqrt{n} $')

    ax[0].set_xticks(np.arange(10, 250, 10), minor=True)
    ax[0].tick_params(axis="x", direction="in", which='both', pad=-22)
    ax[0].set_ylabel('TCP Error [mm]')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlim(1, 250)

    ax[1].semilogy(n_list, statsd[:, 2], color='b', label=r'$ \mu $')
    ax[1].fill_between(x=n_list, y1=statsd_std0[:, 2], y2=statsd_std1[:, 2], alpha=0.1, color='b')
    ax[1].semilogy(n_list, statsd[:, 1], color='m', label=r'$ \sigma $')
    ax[1].fill_between(x=n_list, y1=statsd_std0[:, 1], y2=statsd_std1[:, 1], alpha=0.1, color='m')
    ax[1].semilogy(n_list, statsd[:, 4], color='r', label='max')
    ax[1].fill_between(x=n_list, y1=statsd_std0[:, 4], y2=statsd_std1[:, 4], alpha=0.1, color='r')
    ax[1].semilogy(np.arange(n_list[-1]+1), s1n(np.arange(n_list[-1]+1)), color='k', label='')
    ax[1].set_xlabel('Size of Calibration Set')
    ax[1].set_ylabel('Diff. to Final Err.')
    ax[1].set_yticks(np.logspace(-2, 2, 5))

    ax[1].grid()
    ax[1].set_xlim(1, 250)
    ax[0].set_ylim(1e0, 1e2)
    remove_ticks(ax[0], v=1e0, axis='y')
    ax[0].set_ylim(1e0, 1e2)

    remove_ticks(ax[1], v=1e2, axis='y')
    ax[1].set_ylim(1e-2, 1e2)

    save_fig(filename=ICHR20_CALIBRATION_FIGS + '/Final/size_of_calibration_set', formats='pdf', bbox=None)


if __name__ == '__main__':
    main()
