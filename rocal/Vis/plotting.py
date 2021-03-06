import numpy as np


from wzk import str0_to_n, get_stats, print_stats, print_table
from wzk.spatial import frame2trans_rotvec, frame_difference

from wzk.mpl import new_fig, save_all, error_area, plot_projections_2d


def print_stats2(stats):
    print_table(data=stats.mean(axis=0) * [[1000], [1]], rows=[f'Translation [mm]', 'Rotation [deg]'],
                columns=['mean', 'std', 'median', 'min', 'max'])


# Visualization
def plot_frame_difference(f0, f1, frame_names=None, verbose=2):
    """
    Return an array (n_frames, 2, 5) where the stats (wzk.mathematics.get_stats())
    for translational and rotational differences are combined
    """
    n_samples, n_frames, _, _ = f0.shape
    if frame_names is None:
        frame_names = str0_to_n(s='Frame_', n=n_frames)
    else:
        assert n_frames == len(frame_names)

    stats = np.empty((n_frames, 2, 5))
    for i, fname in enumerate(frame_names):
        d_trans, d_rot = frame_difference(f0[:, i, :, :], f1[:, i, :, :])
        stats[i, 0, :] = get_stats(x=d_trans, return_array=True)
        stats[i, 1, :] = get_stats(x=d_rot, return_array=True)

        if verbose > 1:
            print_frame_difference(d_trans=d_trans, d_rot=d_rot, title=fname)

        if verbose > 1:
            n = 10
            idx_worst = np.argsort(d_trans)[::-1]
            print('Worst Configurations')
            print(repr(idx_worst[:n]))
            print(d_trans[idx_worst[:n]])

        if verbose > 2:
            hist_frame_difference(d_trans=d_trans, d_rot=d_rot, title='Difference - ' + fname)

        if verbose > 3:
            pos, rot = frame2trans_rotvec(f=f0[:, i, :, :])
            pos2, rot2 = frame2trans_rotvec(f=f1[:, i, :, :])

            scatter_measurements_3d(x0=pos, x1=pos2, title='Positions - ' + fname,)
            scatter_measurements_3d(x0=rot, x1=rot2, title='Rotations - ' + fname,)



    if verbose > 0:
        print_stats2(stats=stats)

    return stats


def mm_wrapper(x, mm):
    if mm:
        return x*1000, 'mm'
    else:
        return x, 'm'


def print_frame_difference(f1=None, f2=None,
                           d_trans=None, d_rot=None,
                           title=None, mm=True):
    if d_trans is None or d_rot is None:
        d_trans, d_rot = frame_difference(f1, f2)

    if title is not None:
        print(title)

    d_trans, mmm = mm_wrapper(x=d_trans, mm=mm)
    d_rot = np.rad2deg(d_rot)

    stats = print_stats(d_trans, d_rot, names=[f'Translation [{mmm}]', 'Rotation [deg]'])
    return (d_trans, d_rot), stats


def hist_frame_difference(f1=None, f2=None,
                          d_trans=None, d_rot=None,
                          title='', mm=True, bins=20, **kwargs):

    if d_trans is None or d_rot is None:
        d_trans, d_rot = frame_difference(f1, f2)

    d_trans, mmm = mm_wrapper(d_trans, mm)
    d_rot = np.rad2deg(d_rot)

    fig, ax = new_fig(n_cols=3, title=title, width=10)
    ax[0].hist(d_trans, bins=bins, **kwargs)
    ax[0].set_xlabel(mmm)
    ax[1].hist(d_rot, bins=bins, **kwargs)
    ax[1].set_xlabel('deg')
    ax[2].plot(d_trans, d_rot, ls='', marker='o', alpha=0.7, **kwargs)
    ax[2].set_xlabel(mmm)
    ax[2].set_ylabel('deg')
    return fig


def scatter_measurements_3d(x0, x1, title):
    style_dict = dict(marker='o',
                      markersize=3,
                      alpha=0.8,
                      ls='',
                      dim_labels='xyz')

    fig, ax = new_fig(n_rows=1, n_cols=3, title=title, width=10)
    fig, ax_diff = new_fig(n_rows=1, n_cols=3, title=title, width=10, aspect=1)

    plot_projections_2d(x=x0, ax=ax, color='b', label='measured', **style_dict)
    plot_projections_2d(x=x1, ax=ax, color='r', label='calibrated', **style_dict)
    ax[-1].legend()

    diff = x0 - x1
    max_diff = np.abs(diff).max()
    plot_projections_2d(x=diff, ax=ax_diff, color='k', label='difference',
                        limits=np.array([[-max_diff * 1.02, +max_diff * 1.02]] * 3), **style_dict)
    ax_diff[-1].legend()


def save_plots(file_res):
    import matplotlib.pyplot as plt
    plt.pause(5)
    save = input('If all figures should be saved, type yes')
    if save == 'yes' or save == 'y':
        save_all(directory=file_res[:-4]+'/', close=False)


def plot_bins(o_bins, err, n_bins=100, ax=None,
              x_label='Task A-Optimality', y_label='Mean TCP Error [matrix]', kwargs=None, kwargs_std=None):

    if ax is None:
        fig, ax = new_fig(title='Bins')

    o_bins = o_bins.reshape(n_bins, -1).mean(axis=1)
    err_mean_bins = err.reshape(n_bins, -1).mean(axis=1)
    err_std_bins = err.reshape(n_bins, -1).std(axis=1)
    idx_sorted = np.argsort(o_bins)

    error_area(ax=ax, x=o_bins[idx_sorted], y=err_mean_bins[idx_sorted], y_std=err_std_bins[idx_sorted],
               kwargs=kwargs, kwargs_std=kwargs_std)
    ax.set_xlim(o_bins.min(), o_bins.max())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax
