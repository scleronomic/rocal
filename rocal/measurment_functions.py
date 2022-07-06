import numpy as np

from wzk import spatial, geometry

from mopla.Optimizer.cost_functions import lqql_basin
from rocal.Vis.plotting import plot_frame_difference


def measure_frame_difference(frame_a, frame_b, weighting=None,
                             lambda_trans=1., lambda_rot=1.):

    n_samples, n_frames, _, _ = frame_a.shape

    if weighting is None:
        weighting = np.full((n_samples, n_frames), 1/(n_samples*n_frames))

    cost_loc, cost_rot = spatial.frame_difference_cost(f_a=frame_a, f_b=frame_b)

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

        f, cm, dh_torque = kin_fun(q=q, x=x)

        t2 = cm[0] @ f[:, cal_rob.cm_f_idx, :, :] @ cm[1:]

        obj = measure_frame_difference(frame_a=t2, frame_b=t, weighting=cal_par.f_weighting,
                                       lambda_trans=cal_par.lambda_trans, lambda_rot=cal_par.lambda_rot)

        if cal_par.x_weighting != 0:
            obj += (cal_par.x_weighting*(x - cal_par.x_nominal)**2).mean()
        if verbose > 0:
            # dd = (t2[:, 0, :-1, -1] - t[:, 0, :-1, -1])
            # from wzk.mpl import new_fig
            # fig, ax = new_fig()
            # ax.hist(dd[:, 0], bins=20, color='red')
            # ax.set_xlabel('x : nominal - measured')
            #
            # fig, ax = new_fig()
            # ax.hist(dd[:, 1], bins=20, color='green')
            # ax.set_xlabel('y : nominal - measured')
            #
            # fig, ax = new_fig()
            # ax.hist(dd[:, 2], bins=20, color='blue')
            # ax.set_xlabel('z : nominal - measured')
            #
            #
            # print('dd', dd.mean())
            stats = plot_frame_difference(f0=t, f1=t2, frame_names=None, verbose=verbose-1)  # verbose-1)
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

    xa, xb, n = geometry.capsule_capsule(line_a=x_i.transpose(1, 0, 2), line_b=x_j.transpose(1, 0, 2),
                                         radius_a=cal_rob.capsules_rad[capsule_i], radius_b=cal_rob.capsules_rad[capsule_j])
    return n


def build_objective_cal_touch(q, t,
                              kin_fun,
                              cal_rob, cal_par):

    pairs, t = t

    def objective(x, verbose=0):
        f, cm = kin_fun(q=q, x=x)

        d = _cal_touch(f=f, pairs=pairs, cm=cm, cal_rob=cal_rob)

        obj = lqql_basin(x=d * 1000, a=-2.5, b=-0.5, eps=0.5)  # [mm]
        obj = obj.sum()

        obj_prior = __prior_objective(x=x, prior_mu=np.zeros_like(x), prior_sigma=cal_par.prior_sigma)

        obj += obj_prior
        if verbose > 0:
            return d, obj

        return obj

    return objective


def build_objective_cal_joints(q, t,
                               kin_fun,
                               cal_rob, cal_par):

    q_c = q
    q_m = t

    def objective(x, verbose=0):

        f, cm, dh_trq = kin_fun(q=q_c, x=x)
        q_delta = dh_trq[:, :, 1] - cal_rob.dh[np.newaxis, :, 1]
        q_delta = np.delete(q_delta, 3, axis=1)

        d = q_m - (q_c + q_delta)
        d = (d ** 2)
        obj = d.sum() * 1000
        obj_prior = __prior_objective(x=x, prior_mu=np.zeros_like(x), prior_sigma=cal_par.prior_sigma)
        obj += obj_prior

        if verbose > 0:
            return d, obj

        return obj

    return objective


def build_objective_cal_marker_image(q, t,
                                     kin_fun,
                                     cal_rob, cal_par):

    t_pole = t[:, 0]
    t_right = t[:, 1]
    t_left = t[:, 2]

    i_pole = np.nonzero(t_pole.sum(axis=1) != 0)
    i_right = np.nonzero(t_right.sum(axis=1) != 0)
    i_left = np.nonzero(t_left.sum(axis=1) != 0)

    t_pole = t_pole[i_pole]
    t_right = t_right[i_right]
    t_left = t_left[i_left]

    def objective(x, verbose=0):
        # x = np.random.uniform(-0.001, +0.001, size=x.shape)
        # print(x)
        f, cm, dh_torque = kin_fun(q=q, x=x)

        (cal_rob.marker_pole.f_robot_marker,
         cal_rob.marker_right.f_robot_marker,
         cal_rob.marker_left.f_robot_marker,
         cal_rob.kinect.f_robot_camera) = cm

        distort = True
        u_pole = cal_rob.kinect.project_marker2image(marker=cal_rob.marker_pole, f=f[i_pole], distort=distort)[:, ::-1]
        u_right = cal_rob.kinect.project_marker2image(marker=cal_rob.marker_right, f=f[i_right], distort=distort)[:, ::-1]
        u_left = cal_rob.kinect.project_marker2image(marker=cal_rob.marker_left, f=f[i_left], distort=distort)[:, ::-1]

        d_pole = t_pole - u_pole
        d_right = t_right - u_right
        d_left = t_left - u_left
        d = np.concatenate((d_pole, d_right, d_left), axis=0)
        # obj = np.sum(d ** 2)
        # obj = np.sum(d_right**2) + np.sum(d_left**2)
        obj = np.sum(d_left**2)
        print(obj)

        if cal_par.x_weighting != 0:
            obj += (cal_par.x_weighting*(x - cal_par.x_nominal)**2).mean()

        if verbose > 0:
            # stats = plot_frame_difference(f0=t, f1=t2, frame_names=None, verbose=verbose-1)  # verbose-1)
            return d, obj
            # return stats, obj

        return obj

    return objective


meas_fun_dict = dict(marker=build_objective_cal_marker,
                     marker_image=build_objective_cal_marker_image,
                     touch=build_objective_cal_touch,
                     joints=build_objective_cal_joints)