import numpy as np

from wzk import pyOpt2, spatial, files, math2, printing, strings, multiprocessing2

from rokin import forward

from rocal.measurment_functions import meas_fun_dict, build_objective_cal_marker
from rocal.parameter import offset_nominal_parameters, create_x_unwrapper, get_x_bool_dict
from scipy.optimize import least_squares


# Kinematic
def get_torque(f, cal_rob,
               ma, gravity=None):
    # Finding
    #  f_dh(..., q, theta+X, ...) == f_dh(..., q, theta, ...) @ Rot_z(X)
    # # rotation around z (theta) after dh
    # np.allclose(frame_from_dh2(q=1, d=2, theta=3+1, a=4, alpha=5),
    #             frame_from_dh2(q=1, d=2, theta=3, a=4, alpha=5) @ trans_rotvec2frame(rotvec=np.array([0, 0, 1])))
    # # rotation around x (alpha) before dh
    # np.allclose(frame_from_dh2(q=4, d=3, theta=2, a=1, alpha=1+1),
    #             trans_rotvec2frame(rotvec=np.array([1, 0, 0])) @ frame_from_dh2(q=4, d=3, theta=2, a=1, alpha=1))
    mass, mass_pos = ma[:, -1], ma.copy()
    mass_pos[:, -1] = 1
    torque = forward.get_torques(f=f, mass=mass, mass_pos=mass_pos, mass_f_idx=cal_rob.masses_f_idx,
                                 torque_f_idx=cal_rob.joint_f_idx_dh,
                                 frame_frame_influence=cal_rob.frame_frame_influence,
                                 gravity=gravity,
                                 mode='dh')[1]
    return torque


def get_torque_dh(f, cal_rob,
                  dh, el, ma, gravity=None):
    torque = get_torque(f=f, cal_rob=cal_rob, ma=ma, gravity=gravity)
    return torque_compliance2dh(torque=torque, dh=dh,  el=el, include_beta=cal_rob.include_beta)


def torque_compliance2dh(torque, dh, el, include_beta=False):
    if torque is None or el is None:
        return None

    dh_trq = torque * el  # x, y, z
    if include_beta:
        dh2 = np.zeros((dh_trq.shape[0], dh.shape[0], 5))
        dh2[..., :dh.shape[-1]] = dh
        dh2[..., [3, 4, 1]] += dh_trq  # x, y, z
    else:
        dh2 = np.zeros((dh_trq.shape[0], dh.shape[0], 5))
        dh2[..., :dh.shape[-1]] = dh
        dh2[..., [3, 1]] += dh_trq[..., [0, 2]]  # x, z
    return dh2


def kinematic(cal_rob,
              q, dh, el, ma, cm, ad):

    if cal_rob.add_nominal_offsets:
        dh, el, ma, cm, ad = offset_nominal_parameters(cal_rob=cal_rob, dh=dh, el=el, ma=ma, cm=cm, ad=ad)

    # Gravity hack, encoded in the elasticities
    gravity_shift = spatial.rotvec2matrix(rotvec=el[:3, 1])
    gravity = np.array([0, 0, -1])
    gravity = gravity_shift @ gravity[:, np.newaxis]
    # print('gravity', gravity)

    if cal_rob.use_imu:
        imu, q = np.split(q, [3], axis=-1)
    else:
        imu = None  # noqa TODO NotImplemented

    # Forward Kinematic with compliance in the joints
    f = cal_rob.get_frames_dh(q=q, dh=dh)
    dh_trq = None
    for i in range(cal_rob.el_loop):
        dh_trq = get_torque_dh(f=f, cal_rob=cal_rob, dh=dh, el=el, ma=ma, gravity=gravity)
        f = cal_rob.get_frames_dh(q=q, dh=dh_trq)

    return (f,
            (dh, el, ma, cm, ad),
            dh_trq)


def create_wrapper_kinematic(cal_rob, x_wrapper=None,
                             q=None, x=None):

    if x_wrapper is None:
        n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
        x_wrapper = create_x_unwrapper(**x_bool_dict)

    if q is None and x is None:
        # noinspection PyShadowingNames
        def kinematic2(q, x):
            return kinematic(cal_rob=cal_rob, q=q, **x_wrapper(x))
    elif x is None:
        # noinspection PyShadowingNames
        def kinematic2(x):
            return kinematic(cal_rob=cal_rob, q=q, **x_wrapper(x))
    elif q is None:
        x_dict2 = x_wrapper(x)

        # noinspection PyShadowingNames
        def kinematic2(q):
            return kinematic(cal_rob=cal_rob, q=q, **x_dict2)
    else:
        raise ValueError

    return kinematic2


def wrapper_numeric_kin_jac(q, cal_rob):
    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
    x_wrapper = create_x_unwrapper(**x_bool_dict)
    calc_targets_cal = create_wrapper_kinematic(x_wrapper=x_wrapper, cal_rob=cal_rob, q=q)

    def jac(x):
        return math2.numeric_derivative(fun=calc_targets_cal, x=x)

    return jac, n


def calibrate_wrapper(cal_rob, cal_par, x0_noise):

    def calibrate2(q_cal, t_cal, q_test, t_test, verbose):
        return calibrate(cal_rob=cal_rob, cal_par=cal_par, x0_noise=x0_noise,
                         q_cal=q_cal, t_cal=t_cal, q_test=q_test, t_test=t_test, verbose=verbose)

    return calibrate2


def obj_wrapper(cal_rob, cal_par, obj_fun):

    if isinstance(obj_fun, str):
        obj_fun = meas_fun_dict[obj_fun]

    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
    x_wrapper = create_x_unwrapper(**x_bool_dict)

    kinematic2 = create_wrapper_kinematic(x_wrapper=x_wrapper, cal_rob=cal_rob)

    def obj_fun2(q, t):
        return obj_fun(q=q, t=t, kin_fun=kinematic2, cal_rob=cal_rob, cal_par=cal_par)

    return obj_fun2


def calibrate(cal_rob, cal_par, x0_noise,
              q_cal, t_cal, q_test, t_test,
              obj_fun=build_objective_cal_marker,
              verbose=1):
    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)

    if x0_noise is None or x0_noise == 0:
        x0 = np.zeros(n)
    else:
        x0 = np.random.normal(0, scale=x0_noise, size=n)

    # Pre
    obj_fun = obj_wrapper(cal_rob=cal_rob, cal_par=cal_par, obj_fun=obj_fun)
    obj_fun_cal = obj_fun(q=q_cal, t=t_cal)

    # Main
    cal_par.options['disp'] = verbose > 2
    x = pyOpt2.minimize_slsqp(fun=obj_fun_cal, x0=x0, options=cal_par.options, verbose=verbose-1)
    # x = pyOpt2.minimize_cobyla(fun=obj_fun_cal, x0=x0, options=cal_par.options, verbose=verbose-1)
    # x = least_squares(fun=obj_fun_cal, x0=x0, method="lm").x

    # Post
    if q_test is None or t_test is None:
        print('Attention! No test set given -> Show trainings error')
        q_test = q_cal
        t_test = t_cal

    obj_fun_test = obj_fun(q=q_test, t=t_test)
    stats, _ = obj_fun_test(x, verbose=max(1, verbose))

    return x, stats


# Parallel
def __calibrate_subset_wrapper(q_cal, t_cal, q_test, t_test, fun, verbose=0):

    def calibrate_subset(idx):
        return fun(q_cal=q_cal[idx], t_cal=t_cal[idx], q_test=q_test, t_test=t_test,
                   verbose=verbose)

    return calibrate_subset


def calibrate_subsets(idx_list, fun, n_processes=1, directory=None, verbose=1):

    def __fun(idx_list2):
        __uuid4 = strings.uuid4()
        x_list = []
        stats_list = []
        for i, idx in enumerate(idx_list2):
            if verbose > 0:
                printing.print_progress(i, len(idx_list2))
            x, stats = fun(idx)
            x_list.append(x)
            stats_list.append(stats)
            if directory:
                np.save(directory + f"{__uuid4}.npy",
                        (idx_list2[:i + 1], np.array(x_list), np.array(stats_list), []))

        return np.array(x_list), np.array(stats_list)

    if isinstance(idx_list, tuple) and len(idx_list) == 3:
        idx_list = math2.random_subset(n=idx_list[0], k=[idx_list[1]], m=idx_list[2])

    if directory:
        files.safe_mkdir(directory=directory)

    return multiprocessing2.mp_wrapper(idx_list, fun=__fun, n_processes=n_processes)


# Post Processing
def evaluate_x(x_list, cal_rob, q, t, squared=True):

    fun = create_wrapper_kinematic(cal_rob=cal_rob, q=q)

    t1 = np.array([fun(x=x) for x in x_list])

    diff_trans, diff_rot = spatial.frame_difference(t[np.newaxis, ...], t1)

    if squared:
        diff_trans = np.sqrt(np.mean(np.power(diff_trans, 2), axis=1))
        diff_rot = np.sqrt(np.mean(np.power(diff_rot, 2), axis=1))

    else:
        diff_trans = np.mean(diff_trans, axis=(1, 2))
        diff_rot = np.mean(diff_rot, axis=(1, 2))

    return diff_trans, diff_rot
