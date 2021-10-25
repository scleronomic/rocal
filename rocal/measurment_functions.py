import numpy as np

from wzk.spatial import trans_rotvec2frame, frame_difference_cost
from wzk.geometry import capsule_capsule
# from wzk import get_stats

from mopla.Optimizer.cost_functions import lqql_basin
from rocal.Plots.plotting import plot_frame_difference


def measure_frame_difference(frame_a, frame_b, weighting=None,
                             lambda_trans=1., lambda_rot=1.):

    n_samples, n_frames, _, _ = frame_a.shape

    if weighting is None:
        weighting = np.full((n_samples, n_frames), 1/(n_samples*n_frames))

    cost_loc, cost_rot = frame_difference_cost(f_a=frame_a, f_b=frame_b)

    cost_loc = np.sum(cost_loc * weighting)
    cost_rot = np.sum(cost_rot * weighting)

    objective = cost_loc * lambda_trans + cost_rot * lambda_rot
    return objective


def __prior_objective(x, prior_mu, prior_sigma):
    return (1/prior_sigma**2 * (x - prior_mu) ** 2).sum()


def build_objective_cal_marker(q, t,
                               kin_fun,
                               cal_rob, cal_par):

    def objective(x, verbose=0):
        f, cm = kin_fun(q=q, x=x)

        #
        cm = trans_rotvec2frame(trans=cm[:, :3], rotvec=cm[:, 3:])
        cm = cal_rob.cm @ cm
        t2 = cm[0] @ f[:, cal_rob.cm_f_idx, :, :] @ cm[1:]

        obj = measure_frame_difference(frame_a=t2, frame_b=t, weighting=cal_par.f_weighting,
                                       lambda_trans=cal_par.lambda_trans, lambda_rot=cal_par.lambda_rot)

        if cal_par.x_weighting != 0:
            obj += (cal_par.x_weighting*(x - cal_par.x_nominal)**2).mean()

        if verbose > 0:
            stats = plot_frame_difference(f0=t, f1=2, frame_names=None, verbose=verbose-1)
            return stats, obj

        return obj

    return objective


def build_objective_compensation(frame,
                                 kin_fun,
                                 q0, q_active_bool=None,
                                 f_weighting=None, lambda_trans=1000., lambda_rot=1000.,
                                 x_weighting=0, x_nominal=0):

    def objective(x):

        q = q0.copy()
        q[..., q_active_bool] += x

        frame2 = kin_fun(q=q)

        obj = measure_frame_difference(frame_a=frame, frame_b=frame2, weighting=f_weighting,
                                       lambda_rot=lambda_rot, lambda_trans=lambda_trans)

        if x_weighting != 0:
            obj += (x_weighting*(x - x_nominal)**2).mean()
        return obj

    return objective


def _cal_touch(f, cm, pairs, cal_rob):
    """
    Not quite right / intuitive to make it in this particular order but should not matter in the end
    # from wzk.spatial import sample_frames, invert
    # a, b, c = sample_frames(shape=3)
    # d1 = a @ b @ c
    # d2 = a @ b @ invert(a) @ a @ c
    # assert np.allclose(d1, d2)

    """
    n = len(f)
    i, j = cal_rob.cm_f_idx[pairs.T]
    capsule_i = np.array([np.nonzero(cal_rob.capsules_f_idx == ii)[0].item() for ii in i])
    capsule_j = np.array([np.nonzero(cal_rob.capsules_f_idx == jj)[0].item() for jj in j])
    t_dict = {p: i for i, p in enumerate(cal_rob.cm_f_idx)}

    t_i = np.array([t_dict[ii] for ii in i])
    t_j = np.array([t_dict[jj] for jj in j])
    f_i = cm[t_i] @ f[np.arange(n), i, :, :]
    f_j = cm[t_j] @ f[np.arange(n), j, :, :]

    x_i = (f_i[..., np.newaxis, :, :] * cal_rob.capsules_pos[capsule_i][:, :, np.newaxis, :]).sum(axis=-1)[..., :-1]
    x_j = (f_j[..., np.newaxis, :, :] * cal_rob.capsules_pos[capsule_j][:, :, np.newaxis, :]).sum(axis=-1)[..., :-1]

    xa, xb, n = capsule_capsule(line_a=x_i.transpose(1, 0, 2), line_b=x_j.transpose(1, 0, 2),
                                radius_a=cal_rob.capsules_rad[capsule_i], radius_b=cal_rob.capsules_rad[capsule_j])
    return n


def build_objective_cal_touch(q, t,
                              kin_fun,
                              cal_rob, cal_par):

    pairs, t = t

    def objective(x, verbose=0):
        f, cm = kin_fun(q=q, x=x)
        cm = trans_rotvec2frame(trans=cm[:, :3], rotvec=cm[:, 3:])
        cm = cal_rob.cm @ cm

        d = _cal_touch(f=f, pairs=pairs, cm=cm, cal_rob=cal_rob)

        obj = lqql_basin(x=d * 1000, a=-2.5, b=-0.5, eps=0.5)  # [mm]
        # obj = 100*(d - t)**2

        obj = obj.sum()

        obj_prior = __prior_objective(x=x, prior_mu=np.zeros_like(x), prior_sigma=cal_par.prior_sigma)
        print(obj, obj_prior)

        obj += obj_prior
        if verbose > 0:
            # dd = np.abs(d-t)
            return d, obj
            # stats = get_stats(np.abs(d-t))
            # return stats, obj

        return obj

    return objective


meas_fun_dict = dict(marker=build_objective_cal_marker,
                     touch=build_objective_cal_touch)
