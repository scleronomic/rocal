import numpy as np

from mopla.Kinematic.Robots import Justin19
from mopla.Kinematic.forward import get_x_spheres
from mopla.Justin.parameter_torso import joint_frame_idx_dh
from mopla.Justin.masses import MASSES, MASS_POS, MASS_F_IDX

from wzk.mpl import new_fig, save_fig, plt

from rocal.definitions import ICHR20_CALIBRATION_FIGS


def mass_model():

    robot = Justin19()
    robot.spheres_f_idx = MASS_F_IDX
    robot.spheres_pos = MASS_POS
    robot.spheres_rad = MASSES

    q = np.deg2rad([0, -30, 40,
                    -20, -100, -25, 70, 20, -40, 20,
                    -35, -105, -5, 90, 70, 30, 30,
                    -20, 10])
    m = get_x_spheres(q=q, robot=robot)
    f = robot.get_frames(q)
    fx = f[..., :3, -1]
    fig, ax = new_fig(n_dim=3, width=6, height=6.2)
    ax.scatter(*m.T, s=MASSES*20, alpha=0.5, zorder=-100, color='xkcd:cerulean')
    ms = 5
    lw = 2
    ax.plot(*fx[np.concatenate([range(0, 5), range(23, 27)])].T, marker='o', color='k', lw=lw, markersize=ms)
    ax.plot(*fx[range(4, 14)].T, marker='o', color='k', markersize=ms, lw=lw)
    ax.plot(*fx[np.concatenate([range(4, 5), range(14, 23)])].T, marker='o', color='k', markersize=ms, lw=lw)

    count = 0
    for i, (fxx, ff) in enumerate(zip(fx, f)):
        if i not in joint_frame_idx_dh:
            continue
        a = 0.05
        xxx = np.vstack([fxx-ff[:3, 2]*a, fxx--ff[:3, 2]*a])
        ax.plot(*xxx.T, color='r')
        count += 1

    # Create cubic bounding box to simulate equal aspect ratio
    x, y, z = fx.T
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max(initial=0)
    xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(xb, yb, zb):
        ax.plot([xb], [yb], [zb], color='w', alpha=0)
    ax.elev = 5
    ax.azim = 10
    plt.axis('off')

    directory_fig = ICHR20_CALIBRATION_FIGS + '/Final/'
    save_fig(fig=fig, filename=directory_fig + 'strich_justin', formats='pdf', bbox='tight', transparent=True)


if __name__ == '__main__':
    mass_model()
