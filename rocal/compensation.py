import numpy as np

from wzk.spatial import invert, frame2trans_rotvec, frame_difference
from wzk.math2 import numeric_derivative
from wzk.printing import print_progress

from rocal.calibration import minimize_slsqp
from rocal.measurment_functions import build_objective_compensation
from rocal.Plots.plotting import print_frame_difference


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

    options = {'maxiter': 100, 'disp': False, 'ftol': 1e-8}
    q_list_new = q_list.copy()

    if q_active_bool is None:
        q_active_bool = np.ones(q_list.shape[-1], dtype=bool)

    for i, (q, f) in enumerate(zip(q_list, f_list)):
        if verbose > 1:
            print_progress(i, len(q_list))

        obj = build_objective_compensation(frame=f[np.newaxis, :, :, :], kin_fun=kin_fun,
                                           q0=q[np.newaxis, :], q_active_bool=q_active_bool,
                                           x_weighting=x_weighting, f_weighting=f_weighting)
        q_res = minimize_slsqp(fun=obj, x0=np.zeros(q_active_bool.sum()), options=options, verbose=verbose-1)

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


if __name__ == '__main__':
    import numpy as np

    from wzk.spatial import frame_difference
    from rocal.calibration import create_wrapper_kinematic
    from rocal.Robots.Justin19 import Justin19Cal
    from rocal.definitions import ICHR20_CALIBRATION

    cal_rob = Justin19Cal(dkmc='ff0c', use_imu=True, el_loop=1, add_nominal_offsets=False)
    x, _ = np.load(ICHR20_CALIBRATION + '/final_all.npy', allow_pickle=True)

    kinematic = create_wrapper_kinematic(cal_rob=cal_rob, x=x)
