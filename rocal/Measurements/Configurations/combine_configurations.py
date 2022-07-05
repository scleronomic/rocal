import numpy as np

from wzk import tsp, read_msgpack, write_msgpack, new_fig

from rokin.Robots.Justin19 import justin19_primitives

from mopla.Optimizer import InitialGuess, gradient_descent, feasibility_check, choose_optimum
from mopla.Planner.slow_down_time import slow_down_q_path
from mopla.main import chomp_mp
from mopla.Parameter import get_par_justin19
from mopla.World.real_obstacles import add_tube_img


from rocal.definitions import *
ICHR20_CALIBRATION_DATA = ICHR20_CALIBRATION


par, gd, staircase = get_par_justin19()

par.plan.obstacle_collision = False
par.plan.self_collision = True

par.check.obstacle_collision = False
par.check.self_collision = True

add_tube_img(img=par.oc.img, x=np.array([1.3, 0.0, 0.0]), length=1.2, radius=0.2, limits=par.world.limits)
par.update_oc(img=par.oc.img)


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


def smooth_paths(q_path_list,
                 delta_ramps=1,  # s
                 mean_vel=0.5,   # rad/s
                 max_vel=1,      # rad/s
                 verbose=1):
    timestep_size = 0.001

    is_periodic = None

    q_path_list_smooth = []
    for i, q_path in enumerate(q_path_list):
        path_smooth = slow_down_q_path(q=q_path, mean_vel=mean_vel, max_vel=max_vel,
                                       delta_ramps=delta_ramps, timestep_size=timestep_size,
                                       is_periodic=is_periodic)
        if len(path_smooth) < int((delta_ramps/timestep_size) * 3):
            path_smooth = slow_down_q_path(q=q_path, mean_vel=mean_vel*1.5, max_vel=max_vel,
                                           delta_ramps=delta_ramps/2, timestep_size=timestep_size,
                                           is_periodic=is_periodic)

        q_path_list_smooth.append(path_smooth.tolist())
        if verbose > 0:
            print(f"{i} | shape: {path_smooth.shape}")

    return q_path_list_smooth


def calculate_trajectories_between(q_list):

    fail_count_max = 50
    n = len(q_list)

    # get_x0 = InitialGuess.path.q0s_random_wrapper(robot=par.robot, n_multi_start=n_multi_start_rp,
    #                                               n_wp=par.n_wp, order_random=True, mode='inner')
    q_path_list = np.zeros((n-1, par.n_wp, par.robot.n_dof))

    i = 0
    fail_count = 0
    # weighting = par.weighting.copy()
    while i < n-1:
        print(f"Path: {i}/{n-1}")
        q_start = q_list[i, np.newaxis]
        q_end = q_list[i+1, np.newaxis]

        q_opt = InitialGuess.path.q0s_random(start=q_start, end=q_end, robot=par.robot,
                                             n_wp=par.n_wp, n_multi_start=[[0], [1]], order_random=True, mode='full')
        feasible = feasibility_check(q=q_opt, par=par, verbose=0) > 0

        if feasible[0]:
            q_path_list[i] = q_opt[0]

        else:
            print('Multi-Start')
            # x0 = get_x0(start=q_start, end=q_end)
            par.q_start, par.q_end = q_start, q_end
            # par.weighting = weighting.copy()
            # q_opt, objective = gradient_descent.gd_chomp(q0=x0, par=par, gd=gd)
            q_opt, objective = chomp_mp(par, gd=gd, staircase=staircase)
            feasible = feasibility_check(q=q_opt, par=par, verbose=0) > 0
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
    directory_list = ["D", "E", "A", "B", "C"]
    # directory_list = ["B"]
    n_samples_list = [10, 100, 1000, 10000]
    # n_samples_list = [10]

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

