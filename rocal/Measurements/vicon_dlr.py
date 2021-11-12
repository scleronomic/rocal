import numpy as np
from wzk.mpl import new_fig, hvlines_grid, save_fig


tracking_system_pos_sketch = np.array([[+6., -5., 4.],
                                       [+6., +0., 4.],
                                       [+6., +5., 4.],
                                       [-6., -5., 4.],
                                       [-6., +0., 4.],
                                       [-6., +5., 4.]])

tracking_system_pos = np.array([[6077.15865599469, 6865.63869380712, 3459.09046758155],
                                [6459.22910380056, -2223.04094947304, 3536.0336791628],
                                [6290.30342915698, 2523.45970996789, 3479.58439388084],
                                [-4471.83982524364, 6605.68129326139, 3535.35709930714],
                                [-5848.87908174607, 1865.37341366668, 3612.22699719695],
                                [-4129.2376834596, -2428.24745395319, 3642.60795853699]]) / 1000

# as measured at 2020-08-17
f_world_base = np.array([[1, 0, 0, 0.92],
                         [0, 1, 0, 1.93],
                         [0, 0, 1, 0.07],
                         [0, 0, 0, 1.]])

f_world_base[:3, -1] += 0.1

tracking_system_pos = tracking_system_pos - f_world_base[:3, -1]


if __name__ == '__main__':
    fig, ax = new_fig(aspect=1)
    ax.plot(*tracking_system_pos[:, :2].T, ls='', c='k', marker='o', label='cameras')
    ax.plot(0, 0, ls='', c='b', marker='o', label='robot')
    hvlines_grid(ax=ax, x=tracking_system_pos)
    ax.legend()

    save_fig(file='vicon_camera_poss_2020-08-17', fig=fig, formats='pdf')
