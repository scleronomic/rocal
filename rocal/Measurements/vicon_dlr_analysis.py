import numpy as np
from itertools import combinations

from wzk.mpl import new_fig, save_fig, remove_duplicate_labels
from wzk.spatial import invert, Rotation
from rocal.Measurements import io2
from rocal.Plots.util_plotting import scatter_measurements_3d, plot_projections_2d

from Calibration.use import load_calibrated_kinematic

from mopla.Kinematic.forward import get_frames_x
from mopla.Kinematic.Robots.Justin19 import Justin19

from rocal.definitions import ICHR20_CALIBRATION

robot = Justin19()

kinematic2 = load_calibrated_kinematic()

# f_base_imu = np.array([[0, 0, -1, -0.25],
#                        [0, -1, 0, 0.007],
#                        [-1, 0, 0, -0.34],
#                        [0, 0, 0, 1]])

f_base_imu = np.array([[0, 0, -1, 0.0],
                       [0, -1, 0, 0.0],
                       [-1, 0, 0, -0.3],
                       [0, 0, 0, 1]])


def get_date_img(date_list):
    date_img = np.ones(len(date_list)) * -1
    c = -1
    for i, d_i in enumerate(date_list):
        if date_img[i] == -1:
            c += 1
            date_img[i] = c

        for j, d_j in enumerate(date_list[i + 1:], start=i + 1):
            if d_i.date() == d_j.date():
                date_img[j] = c

    return date_img


# Cal

def poses20_over_time():
    from matplotlib.ticker import MultipleLocator, MaxNLocator
    directory = ICHR20_CALIBRATION + '/TorsoRightLeft/0/m20/'

    n = 20

    p = 1000*(norm_lr - norm_lr_mean)
    p0 = 1000*(norm0_lr - norm0_lr_mean)
    p0b = 1000*(norm0b_lr - norm0b_lr_mean)

    date_img = get_date_img(date) + 1

    p_sum = np.abs(p).sum(axis=0)
    p = p[:, np.argsort(p_sum)]
    p0 = p0[:, np.argsort(p_sum)]
    p0b = p0b[:, np.argsort(p_sum)]

    fig, ax = new_fig(width=10, height=20, n_rows=20, share_x=True)
    fig.subplots_adjust(wspace=0.0075)
    xlim = (0, len(p)-1)
    ylim = (-1, 1)
    # ylim = (-5, 5)

    for i in range(n):
        ax[i].plot(p[:, i], color='k', marker='x', label='measurements')
        ax[i].plot(p0[:, i], ls=':', color='b', marker='o', alpha=0.5, label='nominal kinematic', markersize=3)
        ax[i].plot(p0b[:, i], ls='-.', color='r', marker='s', alpha=0.5, label='calibrated kinematic', markersize=2)
        ax[i].yaxis.set_minor_locator(MultipleLocator(0.5))
        ax[i].grid(which='minor', axis='y')
        ax[i].imshow(date_img[np.newaxis, :], extent=(xlim + ylim), alpha=0.5, zorder=-10,
                     cmap='Pastel2')
        ax[i].set_xlim(xlim)
        ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))

        if i == 0:
            ax[0].legend()

    save_fig(fig=fig, filename='vicon_calibration_over_time', formats='pdf')
    return


def plot_measurements():

    # directory = DLR_USERSTORE_PAPER_2020_CALIB + '/Measurements/0_3_5.0/'
    # directory = ICHR20_CALIBRATION + '/Measurements/0_20_4.0/'
    directory = ICHR20_CALIBRATION + '/TorsoRightLeft/0/m20/'

    n = 20
    q, t, imu, date = io2.load_multiple_measurements(directory=directory, target_order=[2, 1])

    # q0 = np.load(directory + f'../ordered_poses_{n}.npy')[1:-1, 0]
    t = t[:, :-1]
    q = q[:, :-1]
    imu = imu[:, :-1]

    t0 = robot.get_frames(q=q)[:, :, [13, 22], :, :]
    t0b = kinematic2(q=q.reshape(-1, 19)).reshape(t.shape)
    imu = imu[-1]

    imu_normalized = imu / np.linalg.norm(imu, axis=-1, keepdims=True)
    imu_mean = imu.mean(axis=0)
    # imu
    imu_mean_normalized = imu_mean / np.linalg.norm(imu_mean, axis=-1, keepdims=True)
    imu_cos = np.arccos((imu_mean_normalized[np.newaxis, ...] * imu_normalized).sum(axis=-1)).mean(axis=-1)

    print(np.rad2deg(np.pi-np.arccos(imu_normalized[:, 0])).max())
    # All the test sets have values < 1 deg (seems a little bit to small. Did we set up the robot so precise,
    # shock-locks)

    print(np.round(1000*imu_cos, decimals=5), "mm")

    f_imu = np.zeros_like(t)
    f_imu[..., -1, -1] = 1
    f_imu[..., :-1, :-1] = np.array(
        [[Rotation.align_vectors(a=np.array([[-1, 0, 0]]), b=imu_normalized[i:i + 1, j])[0].as_matrix()
          for j in range(imu.shape[1])] for i in range(imu.shape[0])])[:, :, np.newaxis, :, :]

    a = np.zeros_like(imu_normalized)
    a[..., 0] = -1
    f_imu2 = np.zeros_like(t)
    f_imu2[..., -1, -1] = 1
    f_imu2[..., :-1, :-1] = rot(a=a, b=imu_normalized)[:, :, np.newaxis, :, :]

    t = invert(f_imu2) @ invert(f_base_imu) @ t
    t0 = f_imu2 @ invert(f_base_imu) @ t0
    t = f_imu @ invert(f_base_imu) @ t

    tx = t[:, :, :, :3, -1]
    t0x = t0[:, :, :, :3, -1]
    t0bx = t0b[:, :, :, :3, -1]

    def print_diff_mat(mat, title=''):
        print(title)
        mat = np.round(mat * 1000, decimals=3)
        mat[np.tri(len(mat), dtype=bool)] = 0
        print(mat, 'mm')

    def plot_diff_from_mean(_t, title=""):
        fig, ax = new_fig(title=title)
        for i in range(_t.shape[1]):
            ax.plot(1000 * (_t.mean(axis=0)[i] - _t[:, i]), np.zeros_like(norm_lr[:, 0]) + i*1, marker='o', alpha=0.7)

    norm_lr = np.linalg.norm(tx[:, :, 1] - tx[:, :, 0], axis=-1)
    norm0_lr = np.linalg.norm(t0x[:, :, 1] - t0x[:, :, 0], axis=-1)
    norm0b_lr = np.linalg.norm(t0bx[:, :, 1] - t0bx[:, :, 0], axis=-1)

    diff_lr = tx[:, :, 1] - tx[:, :, 0]

    # i = list(range(8, 19)) + list(range(28, 37))
    i = list(range(0, 37))
    norm_lr = norm_lr[i]
    norm0_lr = norm0_lr[i]
    norm0b_lr = norm0b_lr[i]
    date = date[i]

    norm_lr_mean = norm_lr.mean(axis=0)
    norm0_lr_mean = norm0_lr.mean(axis=0)
    norm0b_lr_mean = norm0b_lr.mean(axis=0)

    diff_mat_norm_lr = (norm_lr[:, np.newaxis] - norm_lr[np.newaxis, :]).mean((-1))
    diff_mat_norm0_lr = (norm0_lr[:, np.newaxis] - norm0_lr[np.newaxis, :]).mean((-1))
    diff_mat_norm0b_lr = (norm0b_lr[:, np.newaxis] - norm0b_lr[np.newaxis, :]).mean((-1))

    print_diff_mat(title="Norm Left - Right", mat=diff_mat_norm_lr)
    print_diff_mat(title="Norm Left - Right - Nominal", mat=diff_mat_norm0_lr)
    print_diff_mat(title="Norm Left - Right - Nominal cal", mat=diff_mat_norm0b_lr)

    print("Norm Left - Right")
    print(np.round(1000 * np.abs((norm_lr.mean(axis=0, keepdims=True) - norm_lr)).mean(axis=-1), 4), "mm")

    frame_lr = invert(t[:, :, 1]) @ t[:, :, 0]
    frame_rl = invert(t[:, :, 0]) @ t[:, :, 1]
    x_rl = frame_rl[:, :, :3, -1]
    x_lr = frame_lr[:, :, :3, -1]
    diff_mat_rl = np.linalg.norm(x_rl[:, np.newaxis] - x_rl[np.newaxis, :], axis=-1).mean((-1))
    diff_mat_lr = np.linalg.norm(x_lr[:, np.newaxis] - x_lr[np.newaxis, :], axis=-1).mean((-1))
    print("Relative Frame Left - Right")
    print_diff_mat(mat=diff_mat_lr)
    print("Relative Frame Right - Left")
    print_diff_mat(mat=diff_mat_rl)

    print("Relative Vectors")
    x_rl3 = tx[:, :, 1] - tx[:, :, 0]
    diff_mat_rel_lr2 = np.linalg.norm(x_rl3[:, np.newaxis] - x_rl3[np.newaxis, :], axis=-1).mean((-1))
    print_diff_mat(mat=diff_mat_rel_lr2)

    diff_mat_abs = np.linalg.norm(tx[:, np.newaxis, ...] - tx[np.newaxis, :, ...], axis=-1)
    diff_mat_abs0 = np.linalg.norm(t0x[:, np.newaxis, ...] - t0x[np.newaxis, :, ...], axis=-1)
    diff_mat_abs0b = np.linalg.norm(t0bx[:, np.newaxis, ...] - t0bx[np.newaxis, :, ...], axis=-1)

    print_diff_mat(title='Absolute Position MAX', mat=diff_mat_abs.max(axis=(-1, -2), initial=0))
    print_diff_mat(title='Absolute Position MEAN', mat=diff_mat_abs.mean(axis=(-1, -2)))
    print_diff_mat(title='Absolute Position - Nominal MAX', mat=diff_mat_abs0.max(axis=(-1, -2), initial=0))
    print_diff_mat(title='Absolute Position - Nominal MEAN', mat=diff_mat_abs0.mean(axis=(-1, -2)))
    print_diff_mat(title='Absolute Position - Nominal cal MAX', mat=diff_mat_abs0b.max(axis=(-1, -2), initial=0))
    print_diff_mat(title='Absolute Position - Nominal cal MEAN', mat=diff_mat_abs0b.mean(axis=(-1, -2)))

    print('Absolute Position - Nominal')
    print(np.round(1000 * np.linalg.norm(t0x.mean(axis=0, keepdims=True) - t0x, axis=-1).mean(axis=(-2, -1)), 3), "mm")
    print('Absolute Position')
    print(np.round(1000 * np.linalg.norm(tx.mean(axis=0, keepdims=True) - tx, axis=-1).mean(axis=(-2, -1)), 3), "mm")

    colors = "rbgrgbrgbrgb"
    markers = "oooxxxsssddd"

    fig, ax_2d_diff = new_fig(n_rows=1, n_cols=3, width=10, aspect=1)
    fig, ax = new_fig(n_dim=3)

    for i, tx_i in enumerate(tx):
        print(tx_i.shape)
        ax.plot(*tx_i[:, 0].T, marker='o', ls='', color=colors[i])
        ax.plot(*tx_i[:, 1].T, marker='x', ls='', color=colors[i])

    max_diff = 0

    mean = tx[:, :, 1, :].mean(axis=0)
    mean = t0x[:, :, 1, :].mean(axis=0)

    # for i, c in enumerate(np.arange(len(tx))):
    for i, c in enumerate(combinations(np.arange(len(tx)), 2)):

        # diff = mean - tx[c, :, 1, :]
        diff = tx[c[0], :, 0, :] - tx[c[1], :, 0, :]
        diff *= 1000  # mm
        max_diff = max(max_diff, np.abs(diff).max())
        if len(diff) == 3:
            for j, m in enumerate(['x', 'o', 's']):
                plot_projections_2d(x=diff[j:j+1], ax=ax_2d_diff, dim_labels=dim_labels, color=None, alpha=0.8, s=20,
                                    label=f'({c[0]}-{c[1]})', marker=m,
                                    limits=np.array([[-max_diff * 1.02, +max_diff * 1.02]] * 3))

        else:
            plot_projections_2d(x=diff, ax=ax_2d_diff, dim_labels=dim_labels, color=None, alpha=0.8, s=20,
                                # label=f'({c[0]}-{c[1]})', marker="o",
                                label=f'({c})', marker="o",
                                limits=np.array([[-max_diff * 1.02, +max_diff * 1.02]] * 3))
            # ax
    ax_2d_diff[0].legend()
    remove_duplicate_labels(ax_2d_diff[0])
    # scatter_measurements_3d(x0=tx[0, :, 0, :], x1=tx[1, :, 0, :], title="right")


def test_world_frame():

    directory = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/0/'
    # q, t = io2.load_measurements_right_left_head(directory + 'random_poses_smooth_3.measurements')
    q, t = io2.load_measurements_right_left_head(directory + 'random_poses_smooth_20-1603969427.measurements')

    # robot.f_world_robot = io2.f_world_base

    tx = t[:, :, :3, -1]
    tx2 = get_frames_x(q=q[:, np.newaxis, :], robot=robot)[:, 0, [13, 22, 26]]

    fig, ax = new_fig(n_dim=3)

    colors = "rbg"

    for i in range(2):
        ax.plot(*tx[:, i].T, c=colors[i], ls='', marker='o', label='measured')
        ax.plot(*tx2[:, i].T, c=colors[i], ls='', marker='x', label='assumption')

    ax.legend()

    scatter_measurements_3d(x0=tx[:, 0], x1=tx2[:, 0], title="right")
    scatter_measurements_3d(x0=tx[:, 1], x1=tx2[:, 1], title="left")

    print((tx2 - tx).mean(axis=0)[:2, :])


def joint_comparison(q, q0):
    # FINDING only torso 0 is this bad, can easily be filtered out if range of motion is limit [-120, +120]

    dq = np.abs(q - q0)

    sort_i = dq.argsort(axis=0)

    q_torso0_worst = q[sort_i[:, 0], 0]
    q0_torso0_worst = q0[sort_i[:, 0], 0]

    fig, ax = new_fig()
    ax.plot(q_torso0_worst)
    ax.plot(q0_torso0_worst)

    fig, ax = new_fig()
    ax.plot(np.sort(dq, axis=0)[::-1])


def plot_joints(q, q_c):
    dq = np.abs(q_c[:, 0, :] - q)
    print(np.abs(dq).max(axis=(0, 1)))
    print(np.abs(dq).mean(axis=(0, 1)))
    print(np.round(dq[..., 0], 4))

    fig, ax = new_fig()
    ax.hist(np.rad2deg(dq[..., 0].ravel()))


def __plot_tcp_nullspace(directory, n, cal, i_rmv_pose=None, j_rmv_measurement=None, ax=None):

    directory = directory.format(n=n, cal=cal)
    q, t, imu, date = io2.load_multiple_measurements(directory=directory, target_order=[2, 1])
    q_c = np.load(directory + f'ordered_poses_{n}.npy')[1:-1]

    t = t[:, :-1]
    q = q[:, :-1]

    if j_rmv_measurement is not None:
        q = np.delete(q, j_rmv_measurement, axis=0)
        t = np.delete(t, j_rmv_measurement, axis=0)

    if i_rmv_pose is not None:
        q = np.delete(q, i_rmv_pose, axis=1)
        q_c = np.delete(q_c, i_rmv_pose, axis=0)
        t = np.delete(t, i_rmv_pose, axis=1)

    # q - measured
    # q0c = nominal commanded
    # q1c = calibrated commanded
    # q0m = nominal measured
    # q1m = calibrated measured

    # t - measured
    # t_0c - commanded q, nominal kinematic
    # t_1c - commanded q, calibrated kinematic
    # t_0m - measured q, nominal kinematic
    # t_1m - measured q, calibrated kinematic

    t_0m = robot.get_frames(q=q)[:, :, [13, 22], :, :]
    t_0c = robot.get_frames(q=q_c)[:, :, [13, 22], :, :].reshape((t.shape[1:]))[np.newaxis, ...]
    t_1m = kinematic2(q=q.reshape(-1, 19)).reshape(t.shape)
    t_1c = kinematic2(q=q_c.reshape(-1, 19)).reshape((t.shape[1:]))[np.newaxis, ...]

    def t2tx(*t_list):
        return tuple(_t[:, :, 0, :3, -1] for _t in t_list)

    tx, tx_0m, tx_0c, tx_1m, tx_1c = t2tx(t, t_0m, t_0c, t_1m, t_1c)

    def tx_mean(*args):
        return tuple(tt.mean(axis=(0, 1)) for tt in args)

    tx_mean, tx_mean_0m, tx_mean_0c, tx_mean_1m, tx_mean_1c = tx_mean(tx, tx_0m, tx_0c, tx_1m, tx_1c)

    def dx_mean(a, b):
        return tuple(np.linalg.norm(aa - bb, axis=-1) for aa, bb in zip(a, b))

    dx, dx_0m, dx_0c, dx_1m, dx_1c = dx_mean(a=(tx, tx_0m, tx_0c, tx_1m, tx_1c),
                                             b=(tx_mean, tx_mean_0m, tx_mean_0c, tx_mean_1m, tx_mean_1c))

    def ddxa(*tx_list):
        # txa_list = tuple(tx[0] for tx in tx_list)
        # return tuple(np.linalg.norm(txa[:, np.newaxis, :] - txa[np.newaxis, :, :], axis=-1) for txa in txa_list)
        return tuple(np.linalg.norm(_tx[:, :, np.newaxis, :] - _tx[:, np.newaxis, :, :], axis=-1) for _tx in tx_list)

    ddxa, ddxa_0m, ddxa_0c, ddxa_1m, ddxa_1c = ddxa(tx, tx_0m, tx_0c, tx_1m, tx_1c)

    mm = q.shape[1]
    pred_0m = np.abs(ddxa - ddxa_0m)[..., np.tri(mm, mm, -1, dtype=bool)]
    pred_0c = np.abs(ddxa - ddxa_0c)[..., np.tri(mm, mm, -1, dtype=bool)]
    pred_1m = np.abs(ddxa - ddxa_1m)[..., np.tri(mm, mm, -1, dtype=bool)]
    pred_1c = np.abs(ddxa - ddxa_1c)[..., np.tri(mm, mm, -1, dtype=bool)]

    from wzk import print_table

    print('Difference to the mean position')
    print_table(data=1000*np.array([[dx.mean(), dx.max(initial=0)],
                                    [dx_0m.mean(), dx_0m.max(initial=0)],
                                    [dx_0c.mean(), dx_0c.max(initial=0)],
                                    [dx_1m.mean(), dx_1m.max(initial=0)],
                                    [dx_1c.mean(), dx_1c.max(initial=0)]]).T,
                columns=['', '0m', '0c', '1m', '1c'], rows=['mean', 'max'], cell_format='.3f')

    print('Pairwise Distance Matrix')
    print_table(data=1000*np.array([[ddxa.mean(), ddxa.max(initial=0)],
                                    [ddxa_0m.mean(), ddxa_0m.max(initial=0)],
                                    [ddxa_0c.mean(), ddxa_0c.max(initial=0)],
                                    [ddxa_1m.mean(), ddxa_1m.max(initial=0)],
                                    [ddxa_1c.mean(), ddxa_1c.max(initial=0)]]).T,
                columns=['', '0m', '0c', '1m', '1c'], rows=['mean', 'max'], cell_format='.3f')

    print('Difference between the pairwise Matrix - prediction vs. measurement')
    print_table(data=1000*np.array([[pred_0m.mean(), pred_0m.max()],
                                    [pred_0c.mean(), pred_0c.max()],
                                    [pred_1m.mean(), pred_1m.max()],
                                    [pred_1c.mean(), pred_1c.max()]]).T,
                columns=['0m', '0c', '1m', '1c'], rows=['mean', 'max'], cell_format='.3f')

    fig, ax1 = new_fig()
    ddxa_list = [ddxa, ddxa_0m, ddxa_0c, ddxa_1m, ddxa_1c]
    names = ['measured', 'nominal - measured', 'nominal - commanded', 'calibrated - measured', 'calibrated - commanded']
    for _ddxa, name in zip(ddxa_list, names):
        ax1.hist(1000 * _ddxa[..., np.tri(mm, mm, -1, dtype=bool)].ravel(),
                 bins=50, range=(0, 50), alpha=0.5, label=name)

    ax1.set_xlabel("mm")
    ax1.legend()

    if ax is None:
        fig, ax = new_fig(n_rows=2)

    ax[0].hist(1000 * dx.ravel(), bins=25, range=(0, 20), alpha=0.5, label=f'{cal}measured')
    ax[0].set_xlabel("Distance to Mean TCP [mm]")
    ax[0].legend()

    ax[1].hist(1000 * ddxa[..., np.tri(mm, mm, -1, dtype=bool)].ravel(),  bins=50,  range=(0, 40), alpha=0.5,
               label=f'{cal}measured')
    ax[1].set_xlabel("Pairwise Distance between Poses TCP [mm]")
    ax[0].set_xlim(ax[1].get_xlim())

    repeatability_error = np.linalg.norm(np.abs(tx - tx.mean(axis=0)).mean(axis=0), axis=-1)
    print('Mean Abs Repeatability Error {:.5} mm'.format(repeatability_error.mean()*1000))


def plot_tcps_nullspace():

    n = 20
    cal = ''
    # directory_ = DLR_USERSTORE_PAPER_2020_CALIB + f'/Measurements/TCP_right_{cal}{n}/'
    directory = ICHR20_CALIBRATION + f'/Measurements/TCP_right3_{cal}{n}/'
    # directory = DLR_USERSTORE_PAPER_2020_CALIB + f'/Measurements/TCP_right03_{cal}{n}/'
    # directory = DLR_USERSTORE_PAPER_2020_CALIB + f'/Measurements/TCP_right_left3_{cal}{n}/'
    # # right3_5
    # right3_20
    i_rmv_pose = [2]
    j_rmv_measurement = [0, 1, 2, 3, 4, 5, 6]

    fig, ax = new_fig(n_rows=2)
    __plot_tcp_nullspace(directory=directory, n=n, i_rmv_pose=i_rmv_pose, j_rmv_measurement=j_rmv_measurement, ax=ax,
                         cal='')
    __plot_tcp_nullspace(directory=directory, n=n, i_rmv_pose=i_rmv_pose, j_rmv_measurement=j_rmv_measurement, ax=ax,
                         cal='cal_', )


if __name__ == '__main__':
    plot_measurements()
    # main_plot_tcps_nullspace()
