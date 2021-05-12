import copy
import numpy as np
from scipy.optimize import minimize as minimize_scipy
import pyOpt

from Kinematic.frames import invert, frame2trans_rotvec, trans_rotvec2frame  # noqa: F401 unused import
from Kinematic.dh import frame_from_dh2
from Kinematic.frames_diff import frame_difference_cost, frame_difference
from Kinematic import forward

from Justin.Calibration.util_plotting import print_frame_difference, plot_frame_difference, hist_frame_difference
from wzk import (get_stats, get_timestamp, mp_wrapper, random_subset, safe_create_dir,
                 numeric_derivative,
                 print_stats, print_table, print_dict, print_progress, uuid4,
                 str0_to_n)  # noqa: F401 unused import


# Calibration Parameters
def get_active_parameters(cal_rob):

    """
    TODO think of nomenclature to include different body parts (ie torso + arms)
    dh: '0' - zero   ( turn all dh parameters off)
        'j' - joint  ( offset )
        'f' - full   ( all 4 DH parameters per joint)
        'c' - custom ( subsection of n x 4 dh parameters based on prior studies )

    cp:
        '0' - zero   ( turn all cp off)
        'j' - joint  ( cp )
        'f' - full   ( joint cp + both lateral compliances n x 3)
        'c' - custom ( subsection of n x 3 compliances based on prior studies )

    ma:
        '0' - zero     ( )
        'm' - mass     ( )
        'p' - position ( )
        'f' - full     ( )
        'c' - custom   ( )
    """
    dh, cp, ma, fr = cal_rob.dcmf
    n_dh, n_cp, n_fr, n_ma = cal_rob.n_dh, cal_rob.n_cp, cal_rob.n_fr, cal_rob.n_ma
    m_dh = 4
    m_cp = 3
    m_ma = 4
    m_fr = 6

    __error_string = ("Unknown {}-mode '{}'. Use one of the following: "
                      " ['j' (joint offsets), 'f' (full), 'c' (custom)]")

    if dh == '0':
        dh_bool = np.zeros((n_dh, m_dh), dtype=bool)
    elif dh == 'j':
        dh_bool = np.zeros((n_dh, m_dh), dtype=bool)
        dh_bool[:, 1] = True
    elif dh == 'f':
        dh_bool = np.ones((n_dh, m_dh), dtype=bool)
    elif dh == 'c':
        dh_bool = cal_rob.dh_bool_c
        assert dh_bool.shape == (n_dh, m_dh)

    else:
        raise ValueError(__error_string.format('dh', dh))

    if cp == '0':
        cp_bool = np.zeros((n_cp, m_cp), dtype=bool)
    elif cp == 'j':
        cp_bool = np.zeros((n_cp, m_cp), dtype=bool)
        cp_bool[:, -1] = True
    elif cp == 'f':
        cp_bool = np.ones((n_cp, m_cp), dtype=bool)
    elif cp == 'c':
        cp_bool = cal_rob.cp_bool_c
        assert cp_bool.shape == (n_cp, m_cp)

    else:
        raise ValueError(__error_string.format('cp', cp))

    if ma == '0':
        ma_bool = np.zeros((n_ma, m_ma), dtype=bool)
    elif ma == 'm':
        ma_bool = np.zeros((n_ma, m_ma), dtype=bool)
        ma_bool[:, -1] = True
    elif ma == 'p':
        ma_bool = np.zeros((n_ma, m_ma), dtype=bool)
        ma_bool[:, :3] = True
    elif ma == 'f':
        ma_bool = np.ones((n_ma, m_ma), dtype=bool)
    elif ma == 'c':
        ma_bool = cal_rob.ma_c
        assert ma_bool.shape == (n_ma, m_ma)
    else:
        raise ValueError(__error_string.format('ma', ma))

    if fr == '0':
        fr_bool = np.zeros((n_fr, m_fr), dtype=bool)
    elif fr == 'p':
        fr_bool = np.zeros((n_fr, m_fr), dtype=bool)
        fr_bool[:, :3] = True
    elif fr == 'o':
        fr_bool = np.zeros((n_fr, m_fr), dtype=bool)
        fr_bool[:, 3:] = True
    elif fr == 'f':
        fr_bool = np.ones((n_fr, m_fr), dtype=bool)
    elif fr == 'c':
        fr_bool = cal_rob.fr_c
        assert fr_bool.shape == (n_fr, m_fr)
    else:
        raise ValueError(__error_string.format('ma', ma))

    excluded_joints = cal_rob.frame_frame_influence[:, cal_rob.idx_fr].sum(axis=-1) == 0
    excluded_joints = excluded_joints[cal_rob.joint_frame_idx_dh]
    dh_bool[excluded_joints] = False
    cp_bool[excluded_joints] = False

    return dh_bool, cp_bool, ma_bool, fr_bool


def get_x_bool_dict(cal_rob):
    dh_bool, cp_bool, ma_bool, fr_bool = get_active_parameters(cal_rob)
    n = dh_bool.sum() + cp_bool.sum() + ma_bool.sum() + fr_bool.sum()
    x_bool_dict = dict(dh_bool=dh_bool, cp_bool=cp_bool, ma_bool=ma_bool, fr_bool=fr_bool)
    return n, x_bool_dict


def __calibration_bool2number(cal_bool, idx0, x):
    cal = np.zeros_like(cal_bool, dtype=float)
    idx1 = idx0 + cal_bool.sum()
    cal[cal_bool] = x[idx0:idx1]
    return cal, idx1


def set_bool_dict_false(x_dict, x=None):
    x_dict0 = copy.deepcopy(x_dict)
    c = 0
    for key in x_dict0:
        c += int(np.sum(x_dict0[key][:]))
        x_dict0[key][:] = False

    if x is None:
        return x_dict0
    else:
        x0 = x[:len(x)-c]
        return x_dict0, x0


def update_dict_wrapper(d2):
    def update_dict(d):
        d.update(d2)
        return d

    return update_dict


def wrap_x(x, cal_rob):
    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)

    if isinstance(x, dict):
        dh, cp, ma, fr = tuple([x[key] for key in x])

    elif isinstance(x, (tuple, list)):
        dh, cp, ma, fr = tuple([xx for xx in x])

    else:
        raise ValueError

    fr = np.concatenate(frame2trans_rotvec(f=fr), axis=-1)
    x = np.hstack((dh[x_bool_dict['dh_bool']].ravel(),
                   cp[x_bool_dict['cp_bool']].ravel(),
                   ma[x_bool_dict['ma_bool']].ravel(),
                   fr[x_bool_dict['fr_bool']].ravel()))
    return x


def unwrap_x(cal_rob, x, add_nominal_offset=False):
    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
    x_unwrapper = create_x_unwrapper(**x_bool_dict)
    x = x_unwrapper(x)
    if add_nominal_offset:
        x = offset_nominal_parameters(cal_rob=cal_rob, **x)
        x = dict(dh=x[0], cp=x[1], ma=x[2], fr=x[3])
    return x


def create_x_unwrapper(fr_bool, dh_bool, cp_bool, ma_bool,
                       update_dict=None):

    def x_unwrapper(x):
        dh, j = __calibration_bool2number(cal_bool=dh_bool, idx0=0, x=x)
        cp, j = __calibration_bool2number(cal_bool=cp_bool, idx0=j, x=x)
        ma, j = __calibration_bool2number(cal_bool=ma_bool, idx0=j, x=x)
        fr, j = __calibration_bool2number(cal_bool=fr_bool, idx0=j, x=x)
        fr = trans_rotvec2frame(trans=fr[:, :3], rotvec=fr[:, 3:])

        d = dict(dh=dh, cp=cp, ma=ma, fr=fr)
        if update_dict:
            d = update_dict(d)

        return d

    return x_unwrapper


# Kinematic
def get_torque_dh(*, f, cal_rob,
                  dh, cp, ma):

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
    torque = forward.get_torques(f=f, mass=mass, mass_pos=mass_pos, mass_frame_idx=cal_rob.masses_frame_idx,
                                 torque_frame_idx=cal_rob.joint_frame_idx_dh,
                                 frame_frame_influence=cal_rob.frame_frame_influence,
                                 mode='dh')[1]

    return torque_compliance2dh(torque=torque, dh=dh,  cp=cp, include_beta=cal_rob.include_beta)


def torque_compliance2dh(torque, dh,  cp, include_beta=False):
    if torque is None or cp is None:
        return None

    dh_trq = torque * cp  # x, y, z
    if include_beta:
        dh2 = np.zeros((dh_trq.shape[0], dh.shape[0], 5))
        dh2[..., :dh.shape[-1]] = dh
        dh2[..., [3, 4, 1]] += dh_trq  # x, y, z
    else:
        dh2 = np.zeros((dh_trq.shape[0], dh.shape[0], 5))
        dh2[..., :dh.shape[-1]] = dh
        dh2[..., [3, 1]] += dh_trq[..., [0, 2]]  # x, z
    return dh2


def offset_nominal_parameters(cal_rob,
                              dh, cp, ma, fr):
    # Update nominal values of robot
    # dh2 = dh
    # dh2[:, :4] += cal_rob.dh
    dh2 = cal_rob.dh + dh
    cp2 = cal_rob.cp + cp
    ma2 = cal_rob.ma + ma
    fr2 = cal_rob.fr @ fr

    return dh2, cp2, ma2, fr2


def kinematic(cal_rob,
              q, dh, cp, ma, fr):

    if cal_rob.add_nominal_offsets:
        dh, cp, ma, fr = offset_nominal_parameters(cal_rob=cal_rob, dh=dh, cp=cp, ma=ma, fr=fr)

    if cal_rob.use_imu:
        imu, q = np.split(q, [3], axis=-1)
    else:
        imu = None

    # Forward Kinematic with compliance in the joints
    f = cal_rob.get_frames_dh(q=q, dh=dh)
    dh_trq = get_torque_dh(f=f, cal_rob=cal_rob, dh=dh, cp=cp, ma=ma)
    f = cal_rob.get_frames_dh(q=q, dh=dh_trq)
    t = fr[0] @ f[:, cal_rob.idx_fr, :, :] @ fr[1:]

    t_list = [t]
    for i in range(cal_rob.cp_loop):
        dh_trq = get_torque_dh(f=f, cal_rob=cal_rob, dh=dh, cp=cp, ma=ma)
        f = cal_rob.get_frames_dh(q=q, dh=dh_trq)
        t = fr[0] @ f[:, cal_rob.idx_fr, :, :] @ fr[1:]
        t_list = [t]

    # Visualize
    # t_list = np.array(t_list)
    # txr_list = t_list[:, :, 0, :3, -1]
    # d2last = np.linalg.norm(txr_list[-1] - txr_list, axis=-1)
    # d2_last_stats = get_stats(d2last, axis=-1, return_array=True,)
    # print(np.round(d2_last_stats*1000, 1), 'mm')
    # print(cp2)
    # fig, ax = new_fig()
    # ax.plot(d2last*1000)

    return t


def create_wrapper_kinematic(cal_rob, x_wrapper=None,
                             q=None, x=None):

    if x_wrapper is None:
        n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
        print(n)
        x_wrapper = create_x_unwrapper(**x_bool_dict)

    if q is None and x is None:
        def kinematic2(q, x):
            return kinematic(cal_rob=cal_rob, q=q, **x_wrapper(x))
    elif x is None:
        def kinematic2(x):
            return kinematic(cal_rob=cal_rob, q=q, **x_wrapper(x))
    elif q is None:
        x_dict2 = x_wrapper(x)

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
        return numeric_derivative(fun=calc_targets_cal, x=x)

    return jac, n


# Optimize
def measure_frame_difference(frame_a, frame_b, weighting=None,
                             sigma_trans=1., sigma_rot=1.):

    n_samples, n_frames, _, _ = frame_a.shape

    if weighting is None:
        weighting = np.full((n_samples, n_frames), 1/(n_samples*n_frames))

    cost_loc, cost_rot = frame_difference_cost(f_a=frame_a, f_b=frame_b)

    cost_loc = np.sum(cost_loc * weighting)
    cost_rot = np.sum(cost_rot * weighting)

    objective = cost_loc * sigma_trans + cost_rot * sigma_rot
    return objective


def build_objective_calibration(q, t,
                                kin_fun,
                                f_weighting=None, sigma_trans=1., sigma_rot=100.,
                                x_weighting=0, x_nominal=0):

    def objective(x):
        targets = kin_fun(q=q, x=x)
        obj = measure_frame_difference(frame_a=targets, frame_b=t, weighting=f_weighting,
                                       sigma_trans=sigma_trans, sigma_rot=sigma_rot)

        if x_weighting != 0:
            obj += (x_weighting*(x - x_nominal)**2).mean()

        return obj

    return objective


def build_objective_correction(frame, kin_fun,
                               q0, q_active_bool=None,
                               sigma_trans=1000., sigma_rot=1000.,
                               f_weighting=None,
                               x_weighting=0, x_nominal=0):

    def objective(x):

        q = q0.copy()
        q[..., q_active_bool] += x

        frame2 = kin_fun(q=q)

        obj = measure_frame_difference(frame_a=frame, frame_b=frame2, weighting=f_weighting,
                                       sigma_rot=sigma_rot, sigma_trans=sigma_trans)

        if x_weighting != 0:
            obj += (x_weighting*(x - x_nominal)**2).mean()
        return obj

    return objective


def minimize(fun, x0, method, options, verbose=0):
    """
    # FINDING SLSQP is way faster and seems as accurate as L-BFGS-B
    """
    if 'PyOpt' in method:
        slsqp = pyOpt.SLSQP()
        slsqp.setOption('IPRINT', -1)
        # slsqp.setOption('fileout', 0)
        slsqp.setOption('MAXIT', options['maxiter'])
        try:
            slsqp.setOption('ACC', options['ftol'])
        except KeyError:
            pass

        def fun2(_x):
            f = fun(_x)
            g = []
            fail = 0
            return f, g, fail

        opt_prob = pyOpt.Optimization('', fun2)
        for i, x0_i in enumerate(x0):
            opt_prob.addVar(f"x{i+1}", 'c', lower=-10, upper=10, value=x0_i)
        opt_prob.addObj('f')
        slsqp(opt_prob, sens_type='FD')
        res = opt_prob.solution(0)
        if verbose > 1:
            print('------ Result ------')
            print(res)

        vs = res.getVarSet()
        x = np.array([vs[key].value for key in range(len(x0))])
        return x

    else:
        res = minimize_scipy(fun=fun, x0=x0, method=method, options=options)
        if verbose > 0:
            print('------ Result ------')
            print('Message: ', res.message)
            print('Success: ', res.success)
            print('Value:',    res.fun)

    return res.x


def calibrate_wrapper(*, cal_rob, cal_par, x0_noise):

    def calibrate2(q_cal, t_cal, q_test, t_test, verbose):
        return calibrate(cal_rob=cal_rob, cal_par=cal_par, x0_noise=x0_noise,
                         q_cal=q_cal, t_cal=t_cal, q_test=q_test, t_test=t_test, verbose=verbose)

    return calibrate2


def calibrate(*, cal_rob, cal_par, x0_noise,
              q_cal, t_cal, q_test, t_test,
              verbose=1):

    n, x_bool_dict = get_x_bool_dict(cal_rob=cal_rob)
    if x0_noise is None or x0_noise == 0:
        x0 = np.zeros(n)
    else:
        x0 = np.random.normal(0, scale=x0_noise, size=n)

    x_wrapper = create_x_unwrapper(**x_bool_dict)

    # Pre
    kinematic2 = create_wrapper_kinematic(x_wrapper=x_wrapper, cal_rob=cal_rob)

    obj = build_objective_calibration(q=q_cal, t=t_cal, kin_fun=kinematic2,
                                      sigma_rot=cal_par.sigma_rot, sigma_trans=cal_par.sigma_trans,
                                      f_weighting=cal_par.f_weighting, x_weighting=cal_par.x_weighting)

    # Main
    cal_par.options['disp'] = verbose > 2
    x = minimize(fun=obj, x0=x0, method=cal_par.method, options=cal_par.options, verbose=verbose-1)

    # Post
    if verbose > 1:
        print_dict(x_wrapper(x))

    if q_test is None or t_test is None:
        print('Attention! No test set given -> Show trainings error')
        q_test = q_cal
        t_test = t_cal

    f1 = kinematic2(q=q_test, x=x)
    stats = plot_frame_difference(f0=t_test, f1=f1, frame_names=None, verbose=verbose)

    return x, stats


def get_corrected_q_opt_wrapper(kin_fun, kin_fun_n=None,
                                q_active_bool=None, x_weighting=0, f_weighting=None):

    def get_corrected_q_opt2(q, f=None, verbose=0):
        if f is None:
            f = kin_fun_n(q)
        return get_corrected_q_opt(q_list=q, f_list=f,
                                   q_active_bool=q_active_bool, kin_fun=kin_fun,
                                   x_weighting=x_weighting, f_weighting=f_weighting,
                                   verbose=verbose)

    return get_corrected_q_opt2


def get_corrected_q_opt(q_list, f_list,
                        q_active_bool, kin_fun,
                        x_weighting=0, f_weighting=None,
                        verbose=1):

    method = 'PyOpt - SLSQP'
    options = {'maxiter': 100, 'disp': False, 'ftol': 1e-8}
    q_list_new = q_list.copy()

    if q_active_bool is None:
        q_active_bool = np.ones(q_list.shape[-1], dtype=bool)

    for i, (q, f) in enumerate(zip(q_list, f_list)):
        if verbose > 1:
            print_progress(i, len(q_list))

        obj = build_objective_correction(frame=f[np.newaxis, :, :, :], kin_fun=kin_fun,
                                         q0=q[np.newaxis, :], q_active_bool=q_active_bool,
                                         x_weighting=x_weighting, f_weighting=f_weighting)
        q_res = minimize(fun=obj, x0=np.zeros(q_active_bool.sum()), method=method, options=options, verbose=verbose-1)

        q_list_new[i, q_active_bool] += q_res

    if verbose > 0:
        print_frame_difference(f1=f_list, f2=kin_fun(q_list_new), mm=True)

    return q_list_new


def get_corrected_q_lin_wrapper(kin_fun, kin_fun_n=None):

    def get_corrected_q_lin2(q, f=None, verbose=0):
        if f is None:
            f = kin_fun_n(q)
        return get_corrected_q_lin(q=q, f=f,
                                   kin_fun=kin_fun, verbose=verbose)

    return get_corrected_q_lin2


def get_corrected_q_lin(q, f, kin_fun,
                        verbose=0):

    threshold = (0.0001, 0.001)  # m, rad
    eps_fd = 1e-5
    mode_fd = 'forward'

    def __f2flat(_f):
        n_samples, n_frames, _, _ = _f.shape
        return np.concatenate(frame2trans_rotvec(_f), axis=-1).reshape((n_samples, n_frames * 6))

    def __j2flat(j):
        n_samples, n_frames, _, _, n_dof = j.shape
        return np.concatenate(frame2trans_rotvec(j.transpose(0, 4, 1, 2, 3)), axis=-1
                              ).reshape((n_samples, n_dof, n_frames * 6)).transpose(0, 2, 1)

    def __diff_frame(a, b):
        # finite differences: eps can only be used in the complete representation, not in 4x4 -> apply later
        return (invert(a) @ b) * eps_fd

    def solve_lin(qc, fn):
        dfc_dq = 1/eps_fd * __j2flat(numeric_derivative(fun=kin_fun, x=qc, diff=__diff_frame, eps=eps_fd, mode=mode_fd))
        dq_dfc = np.linalg.pinv(dfc_dq)

        df = -__f2flat(invert(kin_fun(qc)) @ fn)[:, :, np.newaxis]
        dq = (dq_dfc @ df)[..., 0]

        qc += dq
        return qc

    q1 = q.copy()
    i_keep = np.ones(len(q), dtype=bool)
    for i in range(10):

        q1[i_keep] = solve_lin(qc=q1[i_keep], fn=f[i_keep])  # fn=fn[i_keep])
        d_trans, d_rot = frame_difference(f_a=f, f_b=kin_fun(q1))
        i_keep = np.logical_or(d_trans.mean(axis=-1) > threshold[0], d_rot.mean(axis=-1) > threshold[1])

        if i_keep.sum() == 0:
            break
    else:
        # 2 - 3 iterations were sufficient (most of) my tests
        raise RuntimeError('It was not possible to find the corrected q, '
                           'check if you are using the correct calibration model '
                           'and make sure that the world frame is zero')

    if verbose == 1:
        print_frame_difference(f1=f, f2=kin_fun(q1), title=None, mm=True)

    return q1


# Parallel
def __calibrate_subset_wrapper(q_cal, t_cal, q_test, t_test, fun, verbose=0):

    def calibrate_subset(idx):
        return fun(q_cal=q_cal[idx], t_cal=t_cal[idx], q_test=q_test, t_test=t_test,
                   verbose=verbose)

    return calibrate_subset


def calibrate_subsets(idx_list, fun, n_processes=1, directory=None, verbose=1):

    def __fun(idx_list2):
        __uuid4 = uuid4()
        x_list = []
        stats_list = []
        for i, idx in enumerate(idx_list2):
            if verbose > 0:
                print_progress(i, len(idx_list2))
            x, stats = fun(idx)
            x_list.append(x)
            stats_list.append(stats)
            if directory:
                np.save(directory + f"{__uuid4}.npy",
                        (idx_list2[:i + 1], np.array(x_list), np.array(stats_list), []))

        return np.array(x_list), np.array(stats_list)

    if isinstance(idx_list, tuple) and len(idx_list) == 3:
        idx_list = random_subset(n=idx_list[0], k=[idx_list[1]], m=idx_list[2])

    if directory:
        safe_create_dir(directory=directory)

    return mp_wrapper(idx_list, fun=__fun, n_processes=n_processes)


# Post Processing
def evaluate_x(x_list, cal_rob, q, t, squared=True):

    fun = create_wrapper_kinematic(cal_rob=cal_rob, q=q)

    t1 = np.array([fun(x=x) for x in x_list])

    diff_trans, diff_rot = frame_difference(t[np.newaxis, ...], t1)

    if squared:
        diff_trans = np.sqrt(np.mean(np.power(diff_trans, 2), axis=1))
        diff_rot = np.sqrt(np.mean(np.power(diff_rot, 2), axis=1))

    else:
        diff_trans = np.mean(diff_trans, axis=(1, 2))
        diff_rot = np.mean(diff_rot, axis=(1, 2))

    return diff_trans, diff_rot
