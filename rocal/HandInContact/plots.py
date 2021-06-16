from wzk.mpl import new_fig, save_fig

# finger_dict = {'ring': 0,
#                'middle': 1,
#                'fore': 2,
#                'thumb': 3}
# finger_dict2 = {0: 'ring',
#                 1: 'middle',
#                 2: 'fore',
#                 3: 'thumb'}
labels = ('thumb', 'fore', 'middle', 'ring')
color_dict = dict(thumb='k', fore='r', middle='b', ring='m')
marker_dict = dict(thumb=0, fore=1, middle=2, ring=3)


def hist_before_after(d0, d1):
    fig, ax = new_fig()
    ax.hist(d0, density=True, alpha=0.5, color='b', bins=20, label='before')
    ax.hist(d1, density=True, alpha=0.5, color='r', bins=20, label='after')
    ax.set_xlabel('Capsule Distance [mm]')


def finger_before_after(d0, d1, pairs):
    i_list = pairs2finger(pairs)

    fig, ax = new_fig(aspect=1)
    for ii, ll in zip(i_list, labels):
        ax.plot(d0[ii], d1[ii],  ls='', markeredgewidth=5, markersize=10, alpha=0.5,
                label=ll, marker=marker_dict[ll], color=color_dict[ll])
    ax.set_xlabel('Capsule Distance before Calibration [mm]')
    ax.set_ylabel('Capsule Distance after Calibration [mm]')
    ax.legend()
    ax.grid()
    # save_fig(fig=fig, filename='tfmr_calibration_error', formats='pdf')


def pairs2finger(pairs):
    thumb = (pairs == 3).sum(axis=-1) > 0
    fore = (pairs == 2).sum(axis=-1) > 0
    middle = (pairs == 1).sum(axis=-1) > 0
    ring = (pairs == 0).sum(axis=-1) > 0
    return thumb, fore, middle, ring


def finger_hist(d0, d1, pairs):
    i_list = pairs2finger(pairs)

    fig, ax = new_fig(n_rows=4, share_x=True, share_y=True)
    for ax_i, ii, ll in zip(ax, i_list, labels):
        ax_i.hist(d0[ii], alpha=2/3, bins=30, density=True,
                  label=ll, color=color_dict[ll],)
        if d1 is not None:
            ax_i.hist(d1[ii], alpha=1/3, bins=30, density=True,
                      label='after', color='green', zorder=10)
        ax_i.legend()

    ax[-1].set_xlabel('Capsule Distance [mm]')

