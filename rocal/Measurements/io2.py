import numpy as np
import os

from wzk import (read_msgpack, write_msgpack, change_tuple_order, print_stats,
                 combine_iterative_indices, delete_args, train_test_split)
from datetime import datetime

from rokin.Robots.Justin19.justin19_primitives import justin_primitives

__order_rh = [1, 0]
__order_rlh = [2, 1, 0]


def imu2gravity(imu):
    imu = imu / np.linalg.norm(imu, axis=-1, keepdims=True) * 9.81
    imu = imu[:, [2, 1, 0]]
    return imu


def remove_poses_form_measurements(file, target_order, q_list=None, verbose=0):

    threshold = 0.1
    if q_list is None:
        q_list = np.concatenate((justin_primitives(justin='getready_left_side_down')[0],
                                 justin_primitives(justin='getready')[0]))

    q, t, imu = load_measurements(file=file, target_order=target_order)

    idx = []
    for q_remove in q_list:
        idx = np.concatenate([idx, np.nonzero((np.linalg.norm(q - q_remove, axis=-1) < threshold))[0]])

    idx = np.unique(idx).astype(int)
    if verbose:
        print(idx)
    q = np.delete(q, idx, axis=0)
    t = np.delete(t, idx, axis=0)
    save_measurements(q=q, t=t, file=file, target_order=target_order)


def load_measurements(file, target_order, verbose=0):
    measurements = read_msgpack(file)
    q = np.array([m[0] for m in measurements])

    if target_order is None:
        target_order = np.arange(len(measurements[0])-1)

    t = np.concatenate([np.array([m[1+i] for m in measurements])[:, np.newaxis, :, :]
                        for i in target_order], axis=1)

    try:
        imu = np.concatenate([np.array([m[i] for m in measurements])
                              for i in range(1, len(measurements[0])) if np.size(measurements[0][i]) == 3])
    except ValueError:
        imu = None

    if verbose > 0:
        print("Measurements q:\n", q.shape)
        print("Measurements Targets:\n", t.shape)
    return q, t, imu


def load_measurements_right_head(file, verbose=0):
    # [joint_angles[19], f_vicon_head[4][4], f_vicon_hand[4][4]]
    return load_measurements(file=file, target_order=__order_rh, verbose=verbose)


def load_measurements_right_left_head(file, verbose=0):
    # [joint_angles[19], f_vicon_head[4][4], f_vicon_left[4][4]], f_vicon_right[4][4]]
    return load_measurements(file=file, target_order=__order_rlh, verbose=verbose)


def load_multiple_measurements(directory, target_order, verbose=0):
    files = sorted([f for f in os.listdir(directory) if 'measurements' in f])

    q, t, imu, d = change_tuple_order((load_measurements(file=directory + f, target_order=target_order, verbose=verbose)
                                       + (measurement2datetime(f),) for f in files))

    if imu is not None:
        # Attention, imu can be mixed, was not always recorded
        res = np.array(q), np.array(t), np.array(imu), np.array(d)
    else:
        res = np.array(q), np.array(t), np.array(d)

    return res


def measurement2datetime(m):
    m = os.path.splitext(m)[0]
    d = int(m[-10:])
    return datetime.fromtimestamp(d)


def save_measurements(q, t, file, target_order):
    m = [[q.tolist()] + [t[target_order[i]].tolist() for i in range(t.shape[0])] for q, t in zip(q, t)]
    write_msgpack(file=file, nested_list=m)


def combine_measurements(file_list, new_file, load_fun, target_order):
    """

    Example:
    directory = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRight/Validation/Measurements/'
    # file_list = [directory + '/m_random_poses_smooth50_A.msgpk',
    #              directory + '/m_random_poses_smooth50_C.msgpk']
    # new_file = directory + '/m_random_poses_smooth100_AC.msgpk'
    # load_fun = load_measurements_right_head
    # combine_measurements(file_list, new_file, load_fun)
    """

    q = []
    t = []
    for file in file_list:
        q_i, t_i = load_fun(file=file, verbose=0)
        q = np.concatenate((q, q_i), axis=0)
        t = np.concatenate((t, t_i), axis=0)
        save_measurements(q=q, t=t, file=new_file, target_order=target_order)


def combine_measurements_justin2(directory, save=False):

    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1] == '.measurements']

    q_all = np.zeros((0, 19))
    t_all = np.zeros((0, 2, 4, 4))
    imu_all = np.zeros((0, 3))
    for f in files:
        q, t, imu = load_measurements_right_left_head(directory + f)
        q, t, imu = q[:-1, :], t[:-1, :2, :, :], imu[:-1, :]
        q_all = np.concatenate((q_all, q), axis=0)
        t_all = np.concatenate((t_all, t), axis=0)
        imu_all = np.concatenate((imu_all, imu), axis=0)

    if save:
        np.save(directory + f'measurements_{len(q_all)}.npy', (q_all, t_all, imu_all, []))


def save_results(*, x, x_dict, file=None):
    np.save(file, dict(x=x, x_dict=x_dict))


# noinspection PyUnresolvedReferences
def load_results(file):
    data = np.load(file, allow_pickle=True).item()
    x = data['x']
    x_dict = data['d']
    return x, x_dict


def load_error_stats(file, verbose=0):

    idx, pars, x_dict, stats = np.load(file, allow_pickle=True)  # To be compatible with old format
    # idx, par_list, stats_list = np.load(file, allow_pickle=True)

    if np.size(stats) == 0:  # b
        stats = x_dict

    if verbose > 0:
        print('idx:', idx.shape)
        print('pars:', pars.shape)
        print('x_dict:', x_dict.shape)
        print('stats:', stats.shape)

        print_stats(*stats[:, 0, 0, :].T, names=['mean', 'std', 'median', 'min', 'max'])

    err_mean = stats[..., 0, 0, 0]
    # err_std = stats[..., 0, 0, 1]
    # err_med = stats[..., 0, 0, 2]
    err_max = stats[..., 0, 0, 4]

    return idx, pars, err_mean, err_max


def __retrieve_idx_from_error_stats(file, fun=None):
    d, f = os.path.split(file)
    file2 = f"{d}/idx_{f[len('error_'):]}"
    idx = load_error_stats(file=file)[0]

    if fun is not None:
        o = fun(idx)
        np.save(file2, [idx, o, []])
    else:
        np.save(file2, idx)


# Dummy data / oed analysis
def get_parameter_identifier(cal_rob, with_robot=True):
    # config_filter: nf - no filter, ff - full filter , rf - real filter
    mafr0 = f"{int(cal_rob.ma0)}{int(cal_rob.fr0)}"
    imu = '_imu' if cal_rob.use_imu else ''
    if with_robot:
        name = f"{cal_rob.id}_{cal_rob.dkmc}_{cal_rob.el_loop}_{mafr0}_{cal_rob.config_filter}{imu}"
    else:
        name = f"{cal_rob.dkmc}_{cal_rob.el_loop}_{mafr0}_{cal_rob.config_filter}{imu}"

    return name


def __get_path_identifier(directory, cal_rob, name='', full=False):
    if full:
        file = f"{os.path.normpath(directory)}/" \
               f"{cal_rob.id}/" \
               f"f{'_'.join([str(i) for i in cal_rob.cm_f_idx])}/" \
               f"{name}__{get_parameter_identifier(cal_rob=cal_rob, with_robot=False)}.npy"
        return file

    else:
        file = f"{os.path.normpath(directory)}/" \
               f"{cal_rob.id}/" \
               f"q__{cal_rob.config_filter}.npy"
        return file


def load_q(directory, cal_rob):
    file = __get_path_identifier(directory=directory, cal_rob=cal_rob, name='q', full=False)
    return np.load(file)


def load_j(directory, cal_rob,
           prior):
    print(prior, 'prior is missing')
    file = __get_path_identifier(directory=directory, cal_rob=cal_rob, name='j', full=True)
    jac = np.load(file)

    def __get_sub_jac(_jac, mode):
        if mode == 'p':
            _jac = _jac[:, :, :3, -1, :].reshape((_jac.shape[0], -1, _jac.shape[-1]))

        elif mode == 'r':
            raise NotImplementedError

        elif mode == 'pr':
            raise NotImplementedError

        else:
            raise AttributeError

        return _jac

    jac = __get_sub_jac(_jac=jac, mode=cal_rob.target_mode)

    return jac


def load_m(directory, cal_rob):
    file = __get_path_identifier(directory=directory, cal_rob=cal_rob, name='m', full=True)
    return np.load(file, allow_pickle=True)


def save_m(directory, cal_rob, arr):
    file = __get_path_identifier(directory=directory, cal_rob=cal_rob, name='m', full=True)
    return np.save(file, arr)


########################################################################################################################
def get_q(cal_rob, split, seed=0):
    from rocal.definitions import ICHR20_CALIBRATION

    directory = ICHR20_CALIBRATION + '/Measurements/600'
    measurement_file = directory + '/measurements_600.npy'
    q, t, imu, _ = np.load(measurement_file, allow_pickle=True)
    q0 = np.load(directory + '/q0.npy')

    if cal_rob.use_imu:
        imu = imu2gravity(imu=imu)
        q = np.concatenate((imu, q), axis=-1)
        q0 = np.concatenate((imu, q0), axis=-1)

    remove_idx = [[64, 174, 183, 241, 557,
                   270, 320, 306, 307, 183, 430, 425, 173, 517, 512, 172, 513],  # q
                  [263, 414, 377, 259, 314, 370, 409, 10, 205, 412],  # bad
                  [5,  13,  18,  73, 102, 125, 220, 291, 319, 330, 362, 369,  # bad2
                   374, 376, 422, 459, 470, 522, 526, 528, 538, 543, 556, 559],
                  ]
    # [18, 41, 52, 54, 68, 71, 79, 83, 84, 85, 134, 141,  # bad3
    #  166, 178, 194, 199, 209, 215, 224, 226, 240, 253,
    #  257, 259, 280, 281, 306, 313, 321, 332, 341, 342,
    #  347, 386, 394, 419, 428, 429, 440, 449, 453, 467,
    #  482, 487, 492, 498, 500, 501, 517, 534]]
    remove_idx = combine_iterative_indices(n=len(q), idx_list=remove_idx)
    q, q0, t, imu = delete_args(q, q0, t, imu, obj=remove_idx, axis=0)

    # (q0_cal, q_cal, t_cal), (q0_test, q_test, t_test)
    return train_test_split(q0, q, t, split=split, shuffle=True, seed=seed)


if __name__ == '__main__':
    pass
    # file = "/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/0/random_poses_smooth_3-1637234441.measurements"
    # q, t, imu = load_measurements_right_left_head(file=file, verbose=3)

