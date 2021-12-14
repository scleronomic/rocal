# Start with torso in get ready, left arm, pointing to bottom left, and just moving the right arm
# the makers need to point towards the ceiling

import numpy as np
from wzk.mpl import new_fig, subplot_grid
from wzk import tsp, get_timestamp, print_correlation, ray_sphere_intersection_2, print_progress

import Optimizer.self_collision
from Kinematic import forward, sample_configurations as sample, frames as cm
from Kinematic.Robots import Justin19, com
import Justin.parameter_torso as jtp
import Justin.primitives_torso as pp
from Justin.Calibration.Measurements.vicon_dlr import tracking_system_pos
import Optimizer.feasibility_check as fc

import Util.Visualization.justin_mayavi as ja
import Util.Visualization.plotting_2 as plt2

from rocal.definitions import ICHR20_CALIBRATION_DATA
import parameter

# Parameter
par = parameter.initialize_par()
par.robot = Justin19()
parameter.initialize_sc(sc=par.sc, robot=par.robot)
par.check.self_collision = True

par.com = com
world_limits = np.array([[-2, 2],
                         [-2, 2],
                         [-0.5, 3.5]])

par.size.n_waypoints = 22
par.size.n_dof = par.robot.n_dof
par.world = parameter.World(n_dim=par.robot.n_dim, limits=world_limits)

parameter.initialize_oc(oc=par.oc, robot=par.robot, world=par.world, obstacle_img=None)
par.sc.dist_threshold = 0.015  # m
par.com.dist_threshold = 0.2   # m

#  - Marker rotation (x - axis [=0] should point in negative z direction in order to be visible for the cameras)
x_axis_marker0 = (0, np.array([0, 0, -1]))
marker_rotation_threshold = np.deg2rad(20)

#  - Marker occlusion


def show_justin_spheres(r):
    par.robot.spheres_rad = r
    par.robot.spheres_rad[r == -1] = 0.01
    ja.justin_interactive(robot=par.robot)
    par.robot.spheres_rad = jtp.SPHERES_RAD


marker_occlusion_n_camera_threshold = 4

camera_pos_uncertainty = 0.1  # m -> add this value to the spheres

target_right_sphere_idx = 28
sphere_radius_occlusion_right = jtp.SPHERES_RAD.copy()
for i, sfi in enumerate(jtp.SPHERES_F_IDX):
    if sfi not in jtp.IDX_F_RIGHT:
        sphere_radius_occlusion_right[i] += camera_pos_uncertainty
sphere_radius_occlusion_right[jtp.SPHERES_F_IDX == 12] = -1
sphere_radius_occlusion_right[jtp.SPHERES_F_IDX == 13] = -1
# show_justin_spheres(r=sphere_radius_occlusion_right)

target_left_sphere_idx = 42
sphere_radius_occlusion_left = jtp.SPHERES_RAD.copy()
for i, sfi in enumerate(jtp.SPHERES_F_IDX):
    if sfi not in jtp.IDX_F_LEFT:
        sphere_radius_occlusion_left[i] += camera_pos_uncertainty
sphere_radius_occlusion_left[jtp.SPHERES_F_IDX == 21] = -1
sphere_radius_occlusion_left[jtp.SPHERES_F_IDX == 22] = -1
# show_justin_spheres(r=sphere_radius_occlusion_left)

target_head_sphere_idx = 46
sphere_radius_occlusion_head = jtp.SPHERES_RAD.copy()
for i, sfi in enumerate(jtp.SPHERES_F_IDX):
    if sfi not in jtp.IDX_F_HEAD and sfi not in jtp.F_TORSO_BASE:
        sphere_radius_occlusion_head[i] += camera_pos_uncertainty
for i in jtp.IDX_F_HEAD:
    sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == i] = -1
for i in jtp.IDX_F_TORSO:
    sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == i] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_RIGHT_BASE] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_RIGHT_BASE + 1] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_RIGHT_BASE + 2] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_RIGHT_BASE + 3] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_LEFT_BASE] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_LEFT_BASE + 1] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_LEFT_BASE + 2] = -1
sphere_radius_occlusion_head[jtp.SPHERES_F_IDX == jtp.IDX_F_LEFT_BASE + 3] = -1
# show_justin_spheres(r=sphere_radius_occlusion_head)

# No collision with floor
min_z = 0.3  # TODO apply for all

#  - Initial configuration
q_getready = pp.justin_primitives(justin='getready')

# - Traveling Salesman Time Limit
tsp_time_limit_sec = 300


def check_marker_rotation(*, frames,
                          marker_frame_idx, axis_marker0,
                          threshold, verbose=0):
    """
    The rotation of marker is close to the desired pose (pointing towards the ceiling)
     -> justin_torso_right_10: around 3% fulfil this criterion
    """

    axis_idx, axis_marker0 = axis_marker0

    axis_marker = frames[:, 0, marker_frame_idx, :3, axis_idx]
    angle_difference = np.arccos((axis_marker * axis_marker0).sum(axis=1))

    if verbose > 0:
        fig, ax = new_fig()
        n_bins = 50
        ax.hist(angle_difference, bins=n_bins)
        ax.vlines(x=threshold, ymin=0, ymax=frames.shape[0] // n_bins)

    feasible_marker_rotation = angle_difference < threshold
    return feasible_marker_rotation


def check_marker_occlusion(*, x_spheres, x_cameras,
                           marker_sphere_idx, radius,
                           threshold,  # How many of the cameras need to see the marker
                           ):
    n, _, n_spheres, _ = x_spheres.shape
    rays = np.zeros((n, len(x_cameras), 2, 3))
    rays[:, :, 0, :] = x_cameras[np.newaxis, :, :]
    rays[:, :, 1, :] = x_spheres[:, :, marker_sphere_idx, :]

    intersection = ray_sphere_intersection_2(rays=rays, spheres=x_spheres[:, 0, radius != -1, :],
                                             r=radius[radius != -1])

    feasible = intersection.sum(axis=-1) == 0      # is the view of a camera blocked?
    feasible = feasible.sum(axis=-1) >= threshold  # do enough cameras have a free view?
    return feasible


def check_marker_rotation_right(frames):
    return check_marker_rotation(frames=frames, marker_frame_idx=jtp.IDX_F_RIGHT_TCP, axis_marker0=x_axis_marker0,
                                 threshold=marker_rotation_threshold)


def check_marker_rotation_left(frames):
    return check_marker_rotation(frames=frames, marker_frame_idx=jtp.IDX_F_LEFT_TCP, axis_marker0=x_axis_marker0,
                                 threshold=marker_rotation_threshold)


def check_marker_occlusion_right(x_spheres):
    return check_marker_occlusion(x_spheres=x_spheres, x_cameras=tracking_system_pos,
                                  marker_sphere_idx=target_right_sphere_idx, radius=sphere_radius_occlusion_right,
                                  threshold=marker_occlusion_n_camera_threshold)


def check_marker_occlusion_left(x_spheres):
    return check_marker_occlusion(x_spheres=x_spheres, x_cameras=tracking_system_pos,
                                  marker_sphere_idx=target_left_sphere_idx, radius=sphere_radius_occlusion_left,
                                  threshold=marker_occlusion_n_camera_threshold)


def check_marker_occlusion_head(x_spheres):
    return check_marker_occlusion(x_spheres=x_spheres, x_cameras=tracking_system_pos,
                                  marker_sphere_idx=target_head_sphere_idx, radius=sphere_radius_occlusion_head,
                                  threshold=marker_occlusion_n_camera_threshold)


def print_check_marker_occlusion_rlh(q):
    from Kinematic.forward import get_x_spheres
    robot = None
    x_spheres = get_x_spheres(q=q, robot=robot)
    f_r = check_marker_occlusion_right(x_spheres=x_spheres)
    f_l = check_marker_occlusion_left(x_spheres=x_spheres)
    f_h = check_marker_occlusion_head(x_spheres=x_spheres)
    print('Right', np.nonzero(~f_r)[0])
    print('Left', np.nonzero(~f_l)[0])
    print('Head', np.nonzero(~f_h)[0])


def filter_body_part(q,
                     torso=False, right=False, left=False, head=False,
                     return_frames=False):

    assert torso + right + left + head == 1

    variable_joints = pp.get_free_joints(torso=torso, right=right, left=left, head=head)

    frames = forward.get_frames(q=q, robot=par.robot)

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


def show_joint_value_distribution(q, variable_joints):
    n = int(variable_joints.sum())
    ax = subplot_grid(n=n)
    for i, j in enumerate(np.nonzero(variable_joints)[0]):
        ax[np.unravel_index(i, ax.shape)].hist(q[..., j].flatten(), bins=50)


def plot_tcp_poss(frames, marker_frame_idx, q0, robot):
    limits = np.array([[-2, 2],
                       [-2, 2],
                       [0, 3]])
    robot.f_world_robot = cm.trans_rot2frame(trans=[2, 2, 0.5], rot=None)
    frames = robot.f_world_robot @ frames
    tcp_pos = frames[:, marker_frame_idx, 0, :3, -1]
    x_spheres0 = forward.get_x_spheres(q=q0[np.newaxis, np.newaxis], robot=robot)
    x_spheres0 = x_spheres0[0, :, 0, :]

    fig, ax = plt2.new_world_fig(limits=limits, n_dim=3)
    ax.scatter(xs=x_spheres0[:, 0],
               ys=x_spheres0[:, 1],
               zs=x_spheres0[:, 2],
               s=jtp.SPHERES_RAD * 7000, alpha=1)

    ax.plot(tcp_pos[:, 0],
            tcp_pos[:, 1],
            tcp_pos[:, 2], color='k', marker='o', ls='', alpha=0.5)


def order_poses(q, q0, variable_joints, time_limit=300, verbose=1):

    q_o = np.concatenate((q0, q), axis=0)

    route = tsp.solve_tsp(points=q_o[..., 0, variable_joints] * np.sqrt(joint_weighting[variable_joints]),
                          time_limit=time_limit, verbose=verbose)

    q_o = q_o[route, :]
    q_o = np.concatenate((q_o, q0), axis=0)

    if verbose-1 > 0:
        ja.sphere_path_animation(q=q_o, robot=par.robot, show_frames=[13, 22])

    return q_o, route


def main_torso_right(n_samples=100, verbose=1, safe=False):

    sample_sizes = [200, 500]

    variable_joints = pp.get_free_joints(torso=True, right=True, left=False, head=False)

    # Create the samples
    q = sample.sample_q(robot=par.robot, n_samples=n_samples)
    q0_left_down = pp.justin_primitives(justin='getready_left_side_down')
    q[:, :, ~variable_joints] = q0_left_down[:, :, ~variable_joints]

    frames, x_spheres = forward.get_x_spheres(q=q, robot=par.robot, return_frames2=True)

    # Check the feasibility of the poses
    feasible_marker_rotation_right = check_marker_rotation_right(frames=frames)

    feasible_marker_occlusion_right = check_marker_occlusion_right(x_spheres=x_spheres)
    feasible_marker_occlusion_head = check_marker_occlusion_head(x_spheres=x_spheres)

    feasible_self_collision = Optimizer.self_collision.sc_check(x_spheres=x_spheres, par=par.sc)
    feasible_joint_limits = fc.justin_coupled_torso_constraints_limits_check(q, limits=par.robot.limits)
    feasible_center_of_mass = fc.center_of_mass_check(frames=frames, par=par.com)

    feasible = print_correlation(bool_lists=[feasible_marker_rotation_right,
                                             feasible_marker_occlusion_right,
                                             feasible_marker_occlusion_head,
                                             feasible_self_collision,
                                             feasible_center_of_mass,
                                             feasible_joint_limits],
                                 names=['Rotation',
                                        'Occ Right', 'Occ Head',
                                        'Collision', 'COM', 'Limits'])

    q = q[feasible]
    show_joint_value_distribution(q=q, variable_joints=variable_joints)


def main_torso_right_left(n_samples=100, verbose=1, safe=False):

    sample_sizes = [200, 500]

    variable_joints = pp.get_free_joints(torso=True, right=True, left=True, head=False)

    # Create the samples
    q = sample.sample_q(robot=par.robot, n_samples=n_samples)
    q[:, :, ~variable_joints] = q_getready[:, :, ~variable_joints]
    frames, x_spheres = forward.get_x_spheres(q=q, robot=par.robot, return_frames2=True)

    # Check the feasibility of the poses
    feasible_marker_rotation_right = check_marker_rotation_right(frames=frames)
    feasible_marker_rotation_left = check_marker_rotation_left(frames=frames)

    feasible_marker_occlusion_right = check_marker_occlusion_right(x_spheres=x_spheres)
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


# TODO compare current pose of justin with the tracking system base and adapt the positions accordingly
#   keep or redo?
def save_loop(n):
    q = np.zeros((0, 1, 19))
    while len(q) < n:
        print_progress(len(q), n)
        q = np.concatenate((q, main_torso_right_left_filter(100000)), axis=0)

    q = q[:n]
    np.save(f"{ICHR20_CALIBRATION_DATA}random_poses_two_arms{get_timestamp()}_{n}.npy", q)


# q0 = pp.justin_primitives(justin='getready')
# q = q0.copy()
# q[..., 3:-2] = np.deg2rad([0, -80, 0, 90, -60, 0, 30]*2)
# qq = np.vstack((q0, q, q0))[:, 0, :]
# np.save(f"{ICHR20_CALIBRATION_DATA}ordered_poses_1.npy", qq)

# save_loop(10000)
# main_torso_right_left_filter(n_samples=100000)

# q = np.load(ICHR20_CALIBRATION_DATA + 'TorsoRightLeft/random_poses_10000.npy')
# print(q.shape)
# main_torso_right_left_filter(q=q)


# choice = np.random.choice(np.arange(len(q)), n, replace=False)
# q_c = q[choice]
# qb_c = qb[choice]
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

#
# q_o20 = order_poses(q=q20, q0=q_getready, variable_joints=pp.get_free_joints(torso=True, right=True, left=True),
#                     time_limit=20, verbose=0)
#
# q_o50 = order_poses(q=q50, q0=q_getready, variable_joints=pp.get_free_joints(torso=True, right=True, left=True),
#                     time_limit=50, verbose=0)
# #
# ff = ICHR20_CALIBRATION_DATA + 'TorsoRightLeft/TCP_RIGHT/ordered_poses_{}.npy'
# # # np.save(ff.format(10), q_o10)
# # # np.save(ff.format(20), q_o20)
# # # np.save(ff.format(50), q_o50)
#
# q_o10 = np.load(ff.format(10))
# q_o20 = np.load(ff.format(20))
# q_o50 = np.load(ff.format(50))
#

# q, qb = np.load('front_tcp_calibration.npy')
# from Justin.Calibration.use import load_get_corrected_q_lin


# qb = get_corrected_q2(q=q[:, 0])[:, np.newaxis]
# #
# q_o10b = get_corrected_q2(q=q_o10[:, 0])[:, np.newaxis]
# par.check.obstacle_collision = False
# feasible = fc.feasibility_check(q=q.reshape(-1, 1, 19), par=par)
# q = q[feasible == 1]
# q2 = q2[feasible == 1]
# np.save('front_tcp_calibration_w_CAL.npy', (q, qb, []))



# Find interesting TCP poses
# q = np.load('/volume/USERSTORE/tenh_jo/0_Data/Calibration/Results/Paper/Measurements/front_tcp_calibration_fine_1740.npy')
# q2 = get_corrected_q2(q=q, verbose=1)
# TODO idea learn this part supervised, do make it quicker
#   could be molded to optimized c++ code and could allow good parallelization,
#   this would a simple forward function, their exists a derivative, otherwise it is an iterative root search
#   Possible to formulate as linearized GLS solve  q_corrected = 19x19 * q_commanded

# q2 = np.load('/volume/USERSTORE/tenh_jo/0_Data/Calibration/Results/Paper/Measurements/front_tcp_calibration_fine_1740_cal.npy')
# np.save('/volume/USERSTORE/tenh_jo/0_Data/Calibration/Results/Paper/Measurements/front_tcp_calibration_fine_1236_cal_filtered.npy', q2)
# np.save('/volume/USERSTORE/tenh_jo/0_Data/Calibration/Results/Paper/Measurements/front_tcp_calibration_fine_1236_filtered.npy', q)

#
# n = 50
# def find_far_far_away_random():
#     def f(q, i):
#         q_i = q[i]
#         f2 = kinematic2(q_i)
#         tx2 = f2[:, 0, :3, -1]
#         dtx2 = tx2[np.newaxis, :, :] - tx2[:, np.newaxis, :]
#         dtx2_norm = np.linalg.norm(dtx2, axis=-1)
#
#         return dtx2_norm
#
#
#     objective = 0
#     for ii in range(10000):
#         i = np.random.choice(np.arange(len(q)), n, replace=False)
#         dtx2_norm = f(q, i)
#         objective_new = dtx2_norm[np.tri(n, n, -1, dtype=bool)].mean() # + dtx2_norm.max()
#         if objective_new > objective:
#             objective = objective_new
#             print(ii, objective_new)
#             j = i
#
#     print(f(q, j), f(q, j).max())
#
# i5 = np.array([1137, 492, 881, 701, 427])
# # i10 = np.array([195,  41, 527, 519, 776, 521, 956, 391, 923, 980])
# i10 = np.array([145,  407,  647,  308,  718, 1138,  749,  427,  253,  310])
# i20 = np.array([626,   37,  554,  427,  768,  521,  466,   28,  315,  290,
#                 809,  474, 1075,  162,  467,  988,  275,  145,  501,  928])
# i50 = np.array([432,   31,  448, 1186,  335, 1032, 1098,  855,  142,   30,
#                171,  908,  644,  113,   35, 1179,  222,  684, 1126,  427,
#                628, 1209,  260,  393,  327,   74,  984, 1070,  499,  920,
#                445,  809,  315,  408,  210, 1019,  285,  516,  859,  846,
#                250,  833,  602, 1158,  732,  858,  720,  829,  121,  411])
#
# i = np.arange(len(q))
# res = order(q=q[i][:, np.newaxis], q2=q2[i][:, np.newaxis])


q = np.load("/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right3/ordered_poses_1236.npy")
# q = np.load("/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right/ordered_poses_50.npy")
#
# corrected_q = load_corrected_q()
# q[1:-1, 0] = corrected_q(q[1:-1, 0])
# feasible = fc.feasibility_check(q=q.reshape(-1, 1, 19), par=par)
#
# q = q[feasible == 1]
#
# ff = ICHR20_CALIBRATION_DATA + 'TorsoRightLeft/TCP_right03_cal/ordered_poses_{}.npy'
# np.save(ff.format(len(q)-2), q)

corrected_q = load_get_corrected_q_lin()
q = np.load("/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right3/ordered_poses_1236.npy")

n = 1000
q = q[:n]
q[:] = q[25]  # 662 in unfiltered
q_arm = sample.sample_q(robot=par.robot, n_samples=n)
variable_joints_left = pp.get_free_joints(torso=False, right=False, left=True, head=False)
q[..., variable_joints_left] = q_arm[..., variable_joints_left]
feasible = fc.feasibility_check(q=q.reshape(-1, 1, 19), par=par)
print((feasible == 1).sum())

q = q[feasible == 1]

m = 30
q_m = q[np.random.choice(np.arange(len(q)), m, replace=False)]

q2 = np.array([corrected_q(qq, verbose=1) for qq in q_m[:, 0]])
feasible = fc.feasibility_check(q=q2.reshape((-1, 1, 19)), par=par)

q_m = q_m[feasible == 1][:20]
q2 = q2[feasible == 1][:20]


q2 = q2[:, np.newaxis, :]
q_o, route = order_poses(q=q_m, q0=q_getready, variable_joints=pp.get_free_joints(torso=True, right=True, left=True),
                         time_limit=20, verbose=0)
q2_o = q_o.copy()
q2_o[1:-1] = q2[route[1:] - 1]

np.round(q_o, 3)
np.save('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right_left3/ordered_poses_20.npy', q_o)
np.save('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right_left3_cal/ordered_poses_20.npy', q2_o)


#
# dq = q[np.newaxis, :, :] - q[:, np.newaxis, :]
# dq_norm = np.linalg.norm(dq, axis=-1)
# dq_norm[np.eye(len(q), dtype=bool)] = np.inf
# np.sort(dq
#
# _norm.flat)

#
# np.rad2deg((q_o10 - q_o10b).max())
# # q_o20b = get_corrected_q2(q=q_o20[:, 0])[:, np.newaxis]
# q_o50b = get_corrected_q2(q=q_o50[:, 0])[:, np.newaxis]
#
# q_o10b = q_o10b[fc.feasibility_check(q=q_o10b, par=par)]
# q_o20b = q_o20b[fc.feasibility_check(q=q_o20b, par=par)]
# q_o50b = q_o50b[fc.feasibility_check(q=q_o50b, par=par)]
#
# ff = ICHR20_CALIBRATION_DATA + 'TorsoRightLeft/TCP_RIGHT_CAL/ordered_poses_{}.npy'
# np.save(ff.format(10), q_o10b)
# np.save(ff.format(20), q_o20b)
# np.save(ff.format(50), q_o50b)


# Evaluation of the Calibration Position Filtering
########################################################################################################################
# np.save(DLR_USERSTORE_PAPER_2020_CALIB + 'Measurements/front_tcp_calibration_50_fine_ordered_subset.npy', q_o)


#            Rot Right  Rot Left    Occ Right  Occ Left   Collision       COM    Limits
#  Rot Right   0.03285  , 0.00103,   0.02018,   0.01703,   0.01978 ,  0.01337 ,  0.03285
#   Rot Left   0.00103  , 0.03277,   0.01694,   0.02015,   0.01974 ,  0.01343 ,  0.03277
#  Occ Right   0.02018  , 0.01694,   0.53960,   0.29777,   0.37961 ,  0.22607 ,  0.53960
#   Occ Left   0.01703  , 0.02015,   0.29777,   0.53638,   0.37737 ,  0.22531 ,  0.53638
#  Collision   0.01978  , 0.01974,   0.37961,   0.37737,   0.60003 ,  0.25172 ,  0.60003
#        COM   0.01337  , 0.01343,   0.22607,   0.22531,   0.25172 ,  0.40781 ,  0.40781
#     Limits   0.03285  , 0.03277,   0.53960,   0.53638,   0.60003 ,  0.40781 ,  1.00000
# Total: 138/1000000 = 0.000138
#
#
# Orientation Right 3.3%
# Orientation Left 3.3%
# Occlusion Right = 54%
# Occlusion Left = 54%
# Self Collision = 64%
# CoM = 40 %

