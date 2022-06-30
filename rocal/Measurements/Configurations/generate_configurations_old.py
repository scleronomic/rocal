import numpy as np
from wzk.mpl import new_fig, subplot_grid
from wzk import spatial, get_timestamp, print_correlation, print_progress

from rokin.Robots.Justin19 import Justin19, justin19_par, justin19_primitives
from rokin.Vis.configurations import plot_q_distribution
from mopla.Optimizer import feasibility_check
from mopla.Optimizer.self_collision import sc_check
from mopla.Parameter import get_par_justin19

from rocal.Tools.vicon_dlr import TRACKING_SYSTEM_POS
from rocal.definitions import ICHR20_CALIBRATION
from rocal.Measurements.Configurations import configurations

# Parameter
par, gd, _ = get_par_justin19()


world_limits = np.array([[-2, 2],
                         [-2, 2],
                         [-0.5, 3.5]])
par.update_oc(img=None, limits=world_limits)

par.size.n_wp = 22
par.size.n_dof = par.robot.n_dof

par.check.self_collision = True
par.sc.dist_threshold = 0.015  # m
par.com.dist_threshold = 0.2   # m

#  - Marker rotation (x - axis [=0] should point in negative z direction in order to be visible for the cameras)
x_axis_marker0 = (0, np.array([0, 0, -1]))


marker_rotation_threshold = np.deg2rad(20)


#  - Initial configuration
q_getready = justin19_primitives.justin19_primitives(justin='getready')

# - Traveling Salesman Time Limit
tsp_time_limit_sec = 300


def check_marker_rotation_right(frames):
    return check_marker_rotation(frames=frames, marker_frame_idx=justin19_par.IDX_F_RIGHT_TCP, axis_marker0=x_axis_marker0,
                                 threshold=marker_rotation_threshold)


def check_marker_rotation_left(frames):
    return check_marker_rotation(frames=frames, marker_frame_idx=justin19_par.IDX_F_LEFT_TCP, axis_marker0=x_axis_marker0,
                                 threshold=marker_rotation_threshold)


def check_marker_occlusion_right(x_spheres):
    return check_marker_occlusion(x_spheres=x_spheres, x_cameras=TRACKING_SYSTEM_POS,
                                  marker_sphere_idx=target_right_sphere_idx, radius=sphere_radius_occlusion_right,
                                  threshold=marker_occlusion_n_camera_threshold)


def check_marker_occlusion_left(x_spheres):
    return check_marker_occlusion(x_spheres=x_spheres, x_cameras=TRACKING_SYSTEM_POS,
                                  marker_sphere_idx=target_left_sphere_idx, radius=sphere_radius_occlusion_left,
                                  threshold=marker_occlusion_n_camera_threshold)


def check_marker_occlusion_head(x_spheres):
    return check_marker_occlusion(x_spheres=x_spheres, x_cameras=TRACKING_SYSTEM_POS,
                                  marker_sphere_idx=target_head_sphere_idx, radius=sphere_radius_occlusion_head,
                                  threshold=marker_occlusion_n_camera_threshold)


def print_check_marker_occlusion_rlh(robot, q):
    x_spheres = robot.get_spheres(q=q)
    f_r = check_marker_occlusion_right(x_spheres=x_spheres)
    f_l = check_marker_occlusion_left(x_spheres=x_spheres)
    f_h = check_marker_occlusion_head(x_spheres=x_spheres)
    print('Right', np.nonzero(~f_r)[0])
    print('Left', np.nonzero(~f_l)[0])
    print('Head', np.nonzero(~f_h)[0])


def filter_body_part(robot, q,
                     torso=False, right=False, left=False, head=False,
                     return_frames=False):

    assert torso + right + left + head == 1

    variable_joints = pp.get_free_joints(torso=torso, right=right, left=left, head=head)

    frames = robot.get_frames(q=q, robot=par.robot)

    if torso:
        feasible = fc.center_of_mass_check(frames=frames, par=par.com)
        feasible_z = frames[:, 0, :, 2, -1].min(axis=-1) > min_z
        feasible = np.logical_and(feasible, feasible_z)

    elif right:
        feasible = check_marker_rotation_right(frames=frames)

    elif left:
        feasible = check_marker_rotation_left(frames=frames)

    elif head:
        raise NotImplementedError

    else:
        raise ValueError

    if return_frames:
        return q[feasible], frames[feasible]
    else:
        return q[feasible]


def main_torso_right_left(n_samples=100, verbose=1, safe=False):

    variable_joints = pp.get_free_joints(torso=True, right=True, left=True, head=False)

    # Create the samples
    q = par.robot.sample_q(n_samples)
    q[:, :, ~variable_joints] = q_getready[:, :, ~variable_joints]
    frames, x_spheres = par.robot.get_x_spheres(q=q, return_frames2=True)

    # Check the feasibility of the poses
    feasible_marker_rotation_right = check_marker_rotation_right(frames=frames)
    feasible_marker_rotation_left = check_marker_rotation_left(frames=frames)

    feasible_marker_occlusion_right = configurations.check_marker_occlusion()
    feasible_marker_occlusion_left = check_marker_occlusion_left(x_spheres=x_spheres)
    # feasible_marker_occlusion_head = check_marker_occlusion_head(x_spheres=x_spheres)

    feasible_self_collision = Optimizer.self_collision.sc_check(x_spheres=x_spheres, par=par.sc)
    feasible_joint_limits = fc.justin_coupled_torso_constraints_limits_check(q, limits=par.robot.limits)
    feasible_center_of_mass = fc.center_of_mass_check(frames=frames, par=par.com)

    feasible = print_correlation(bool_lists=[feasible_marker_rotation_right,
                                             feasible_marker_rotation_left,
                                             feasible_marker_occlusion_right,
                                             feasible_marker_occlusion_left,
                                             # feasible_marker_occlusion_head,
                                             feasible_self_collision,
                                             feasible_center_of_mass],
                                 names=['Rot Right', 'Rot Left',
                                        'Occ Right', 'Occ Left',  # 'Occ Head',
                                        'Collision', 'COM'])
    # -> Total: 23/100000 feasible

    q = q[feasible]
    show_joint_value_distribution(q=q, variable_joints=variable_joints)
    return q


def main_torso_right_left_filter(n_samples=1000, q=None, verbose=0):

    variable_joints = pp.get_free_joints(torso=True, right=True, left=True, head=False)

    # Create the samples
    if q is None:
        q = sample.sample_q(robot=par.robot, n_samples=n_samples)
    q[:, :, ~variable_joints] = q_getready[:, :, ~variable_joints]

    q = filter_body_part(q=q, torso=True)
    q = filter_body_part(q=q, right=True)
    q, frames = filter_body_part(q=q, left=True, return_frames=True)
    x_spheres = forward.frames2pos_spheres(f=frames, robot=par.robot)

    feasible_marker_occlusion_right = check_marker_occlusion_right(x_spheres=x_spheres)
    feasible_marker_occlusion_left = check_marker_occlusion_left(x_spheres=x_spheres)
    # feasible_marker_occlusion_head = check_marker_occlusion_head(x_spheres=x_spheres)

    feasible_self_collision = Optimizer.self_collision.sc_check(x_spheres=x_spheres, par=par.sc)
    # feasible_joint_limits = fc.justin_coupled_torso_constraints_limits_check(q, limits=par.robot.limits)

    feasible = print_correlation(bool_lists=[feasible_marker_occlusion_right,
                                             feasible_marker_occlusion_left,
                                             # feasible_marker_occlusion_head,
                                             feasible_self_collision],
                                 names=['Occ Right', 'Occ Left',  # 'Occ Head',
                                        'Collision'])

    q = q[feasible]

    # show_joint_value_distribution(q=q, variable_joints=variable_joints)
    return q


def save_loop(n):
    q = np.zeros((0, 1, 19))
    while len(q) < n:
        print_progress(len(q), n)
        q = np.concatenate((q, main_torso_right_left_filter(100000)), axis=0)

    q = q[:n]
    np.save(f"{ICHR20_CALIBRATION_DATA}random_poses_two_arms{get_timestamp()}_{n}.npy", q)

#
def order(q, q2):
    n = len(q)
    q_o, route = order_poses(q=q, q0=q_getready, variable_joints=pp.get_free_joints(torso=True, right=True, left=True),
                             time_limit=max(n, 600), verbose=0)
    q2_o = q_o.copy()
    q2_o[1:-1] = q2[route[1:]-1]

    np.save(ICHR20_CALIBRATION_DATA + f'TorsoRightLeft/TCP_right3/ordered_poses_{n}.npy', q_o)
    np.save(ICHR20_CALIBRATION_DATA + f'TorsoRightLeft/TCP_right3_cal/ordered_poses_{n}.npy', q2_o)

    return q_o, q2_o
