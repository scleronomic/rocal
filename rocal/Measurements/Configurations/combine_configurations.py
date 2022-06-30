import numpy as np

from wzk import get_exclusion_mask, read_msgpack, write_msgpack
from wzk.trajectory import inner2full

from rokin.Robots.Justin19 import justin19_par as jtp, justin19_primitives, Justin19
from rokin.Vis.robot_3d import animate_path

from mopla.Optimizer import InitialGuess, gradient_descent, feasibility_check, choose_optimum
from mopla.Parameter import parameter

from rocal.definitions import *
ICHR20_CALIBRATION_DATA = ICHR20_CALIBRATION


# kauket_file = '/net/kauket/home_local/baeuml/tmp/two_arm_A_10.msgpk'

# General
verbose = 1

par = parameter.Parameter(robot=Justin19())
par.n_wp = 20
world_limits = np.array([[-2, 2],
                         [-2, 2],
                         [-0.5, 3.5]])
par.update_oc(img=None, limits=world_limits)


par.plan.include_end = False
par.plan.obstacle_collision = False
par.plan.self_collision = True
par.check.obstacle_collision = False
par.check.self_collision = True

# Model for obstacle collision
exclude_frames = []  # Base-0, Right_Hand-13, Left_Hand-22
par.oc.active_spheres = get_exclusion_mask(a=jtp.SPHERES_F_IDX, exclude_values=exclude_frames)

par.oc.n_substeps = 2
par.oc.n_substeps = 2
par.oc.n_substeps_check = 2
par.sc.n_substeps = 1
par.sc.n_substeps = 1
par.sc.n_substeps_check = 2

gd = parameter.GradientDescent()
gd.n_steps = 100
gd.clipping = 0.5
gd.n_processes = 12
n_multi_start_rp = [[0, 1, 2, 3], [1, 2 * gd.n_processes - 1, 2 * gd.n_processes, 1 * gd.n_processes]]


par.weighting.length = 1
par.weighting.collision = 500

import numpy as np

from wzk import geometry, tsp, new_fig, spatial

from rokin.Vis.robot_3d import animate_path
from mopla.Planner.slow_down_time import slow_down_q_path


def order_configurations(q, q0, time_limit=300,
                         weighting=None, variable_joints=None, verbose=1):
    q_ordered = np.concatenate((q0, q), axis=0)

    if weighting is None:
        points = q_ordered
    else:
        points = q_ordered * weighting

    if variable_joints is not None:
        points = points[..., variable_joints]

    route = tsp.solve_tsp(points=points, time_limit=time_limit, verbose=verbose)

    q_ordered = q_ordered[route, :]
    q_ordered = np.concatenate((q_ordered, q0), axis=0)

    return q_ordered, route


def smooth_paths(q_path_list, verbose=1):
    timestep_size = 0.001
    delta_ramps = 1
    mean_vel_q = 0.5
    max_vel_q = 1
    is_periodic = None

    q_path_list_smooth = []
    for i, q_path in enumerate(q_path_list):
        path_smooth = slow_down_q_path(q=q_path, mean_vel_q=mean_vel_q, max_vel_q=max_vel_q,
                                       delta_ramps=delta_ramps, timestep_size=timestep_size,
                                       is_periodic=is_periodic)
        q_path_list_smooth.append(path_smooth.tolist())
        if verbose > 0:
            print(f"{i} | shape: {path_smooth.shape}")

    return q_path_list_smooth


def calculate_trajectories_between(par, gd, q_list):

    fail_count_max = 50
    n = len(q_list)

    get_x0 = InitialGuess.path.q0s_random_wrapper(robot=par.robot, n_multi_start=n_multi_start_rp,
                                                  n_wp=par.n_wp, order_random=True, mode='inner')
    q_path_list = np.zeros((n-1, par.n_wp, par.robot.n_dof))

    i = 0
    fail_count = 0
    weighting = par.weighting.copy()
    while i < n-1:
        print(f"Path: {i}/{n-1}")
        q_start = q_list[i, np.newaxis]
        q_end = q_list[i+1, np.newaxis]

        q_opt = InitialGuess.path.q0s_random(start=q_start, end=q_end, robot=par.robot,
                                             n_wp=par.n_wp, n_multi_start=[[0], [1]], order_random=True)
        feasible = feasibility_check(q=q_opt, par=par, verbose=0)
        feasible = feasible >= 0

        if not feasible:
            x0 = get_x0(start=q_start, end=q_end)
            par.q_start, par.q_end = q_start, q_end
            par.weighting = weighting.copy()
            q_opt, objective = gradient_descent.gd_chomp(q0=x0, par=par, gd=gd)

            q_opt = inner2full(inner=q_opt, start=q_start, end=q_end)
            feasible = feasibility_check(q=q_opt, par=par, verbose=0)
            feasible = feasible >= 0
            q_opt, *_ = choose_optimum.get_feasible_optimum(q=q_opt, par=par, verbose=2)
            print(feasible, q_opt.shape[0])

            if q_opt.shape[0] == 0:
                i -= 1
                fail_count += 1
                if fail_count >= fail_count_max:
                    raise ValueError(f"No feasible path could be found at iteration {i} "
                                     f"between start {q_start} and end {q_end}")
            else:
                fail_count = 0
                q_path_list[i] = q_opt

            print("Fail Count", fail_count)

        else:
            q_path_list[i] = q_opt

        i += 1

    return q_path_list


def calculate_path(directory, n_samples, mode='ordered'):
    print(f"directory: {directory}, n_samples: {n_samples}")

    q_list = np.load(f"{directory}/{mode}_poses_{n_samples}.npy")

    q_path_list = calculate_trajectories_between(q_list=q_list)

    np.save(f"{directory}/{mode}_poses_path_{n_samples}.npy", q_path_list)

    return q_path_list


def no_unnecessary_motion(directory, n_samples, variable_joints, q0, mode='ordered'):
    try:
        q_test_paths = np.load(f"{directory}/{mode}_poses_path_{n_samples}.npy")
        print("Load path.npy")

    except FileNotFoundError:
        print("Calculate path.npy")
        q_test_paths = calculate_path(directory=directory, n_samples=n_samples, mode=mode)

    # Check closeness to the initial position, the other limbs should not move
    for i in range(n_samples + 1):
        q_test_paths_temp = q_test_paths[i].copy()
        q_test_paths_temp[..., ~variable_joints] = q0[..., ~variable_joints].ravel()
        feasible = feasibility_check(q=q_test_paths_temp[np.newaxis, ...], par=par, verbose=0)
        feasible = feasible >= 0
        if feasible:
            q_test_paths[i] = q_test_paths_temp
            print("OK")
        else:
            print("Not Smooth", i)

    # Create smooth path [List of Lists]
    np.save(f"{directory}/{mode}_poses_path_{n_samples}.npy", q_test_paths)


def test_read():
    file = ICHR20_CALIBRATION_DATA + 'TorsoRightArm/' + 'random_poses_path_smooth_100.bin'
    file = PROJECT_ROOT + "random_poses_smooth_1.bin"
    file = PROJECT_ROOT + "front_tcp_calibration_50_fine_ordered_subset_smooth.bin"

    file = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right3/random_poses_smooth_20.bin'

    q_list = read_msgpack(file)
    q_list2 = []
    for qq in q_list:
        qq = np.array(qq)
        print(qq.shape)
        q_list2.append(qq)
        q_list2.append(qq[-1:, :].repeat(2000, axis=0))

    q = np.concatenate(q_list2)

    print(q.shape)
    robot_path_interactive(q=np.array(q), robot=par.robot, kwargs_frames=dict(f_idx_robot=[13, 22]))

    q_poses = np.load('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_right3/ordered_poses_20.npy')

    q_torso0_b = np.array([ql[-1][0] for ql in q_list])[:-1]
    q_torso0 = q_poses[1:-1, 0, 0]
    np.allclose(q_torso0, q_torso0_b)

    from wzk.mpl import new_fig

    fig, ax = new_fig()

    ax.plot(np.array(q_list[2]))


def main():
    # directory_list = ["D", "E", "A", "B", "C"]
    # # directory_list = ["B"]
    # n_samples_list = [10, 100, 1000, 10000]
    # # n_samples_list = [10]

    q0 = justin19_primitives.justin19_primitives(justin='getready')
    # q_test0[jtp.JOINTS_ARM_LEFT] = np.array([0, -80, 90, -40, 0, 0, 0]) * np.pi / 180
    # q0[jtp.IDX_J_LEFT] = np.array([0, -80, 0, 0, 0, 0, 0]) * np.pi / 180

    variable_joints = justin19_primitives.get_free_joints(torso=True, right=True, left=True)

    # for f in [str(i) for i in range(10, 20)]:
    for f in ['TEST']:
        for n in [1]:
            calculate_path(directory=f"TorsoRightLeft/{f}/", mode='ordered', n_samples=n)
            no_unnecessary_motion(directory=f"TorsoRightLeft/{f}", mode='ordered', n_samples=n,
                                  variable_joints=variable_joints, q0=q0)
