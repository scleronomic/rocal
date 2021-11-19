import numpy as np

from wzk import get_exclusion_mask, read_msgpack, write_msgpack
from wzk.trajectory import inner2full
from rokin.Robots.Justin19 import justin19_par as jtp, justin19_primitives, Justin19
from rokin.Vis.robot_3d import robot_path_interactive

from mopla.Justin.Planner.slow_down_time import slow_down_q_path
from mopla.Optimizer import InitialGuess, gradient_descent, feasibility_check, choose_optimum
from mopla import parameter

from rocal.definitions import *

ICHR20_CALIBRATION_DATA = ICHR20_CALIBRATION


# kauket_file = '/net/kauket/home_local/baeuml/tmp/two_arm_A_10.msgpk'

# General
verbose = 1

par = parameter.Parameter(robot=Justin19())

parameter.initialize_sc(sc=par.sc, robot=par.robot)
world_limits = np.array([[-2, 2],
                         [-2, 2],
                         [-0.5, 3.5]])

par.world = parameter.World(n_dim=par.robot.n_dim, limits=world_limits)


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

par.weighting.joint_motion = np.array([10, 5, 5,
                                       3, 3, 2, 2, 1, 1, 1,
                                       3, 3, 2, 2, 1, 1, 1,
                                       2, 2])
par.weighting.length = np.linspace(start=20, stop=1, num=gd.n_steps + 1)
par.weighting.length[-1] = 1
par.weighting.collision = np.linspace(start=500, stop=500, num=gd.n_steps + 1)
par.weighting.collision[-1] = 1000

parameter.initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img='empty')


def calculate_trajectories_between(q_list):

    fail_count_max = 50
    n = len(q_list)

    get_x0 = InitialGuess.path.q0_random_wrapper(robot=par.robot, n_multi_start=n_multi_start_rp,
                                                 n_waypoints=par.n_waypoints, order_random=True, mode='inner')
    q_path_list = np.zeros((n-1, par.n_waypoints, par.robot.n_dof))

    i = 0
    fail_count = 0
    weighting = par.weighting.copy()
    while i < n-1:
        print(f"Path: {i}/{n}")
        q_start = q_list[i, np.newaxis]
        q_end = q_list[i+1, np.newaxis]

        q_opt = InitialGuess.path.q0_random(start=q_start, end=q_end, robot=par.robot,
                                            n_waypoints=par.n_waypoints, n_random_points=0, order_random=True)
        feasible = feasibility_check(q=q_opt, par=par, verbose=0)
        feasible = feasible >= 0

        if not feasible:
            x0 = get_x0(start=q_start, end=q_end)

            par.weighting = weighting.copy()
            q_opt, objective = gradient_descent.gd_chomp(q0=x0, q_start=q_start, q_end=q_end, par=par, gd=gd)

            q_opt = inner2full(inner=q_opt, start=q_start, end=q_end)
            feasible = feasibility_check(q=q_opt, par=par, verbose=0)
            feasible = feasible >= 0
            q_opt, _ = choose_optimum.get_feasible_optimum(q=q_opt, par=par, verbose=2)
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
            print("Not Smooth", i)  # D, 1000, 743
            # animate_poses(q=q_test_paths[o-1:o+1].reshape((-1, par.shape.n_dof)))

    # Create smooth path [List of Lists]
    np.save(f"{directory}/{mode}_poses_path_{n_samples}.npy", q_test_paths)


def smooth_paths(q_path_list, verbose=1):
    timestep_size = 0.001
    delta_ramps = 1
    mean_vel_q = 0.5
    max_vel_q = 1
    q_test_paths_smooth = []
    for q_path in q_path_list:
        path_smooth = slow_down_q_path(q=q_path, mean_vel_q=mean_vel_q, max_vel_q=max_vel_q,
                                       delta_ramps=delta_ramps, timestep_size=timestep_size,
                                       infinity_joints=par.robot.infinity_joints)
        q_test_paths_smooth.append(path_smooth.tolist())  # smooth list for writing to msgpack
        if verbose > 0:
            print('Smooth path shape:', path_smooth.shape)

    return q_test_paths_smooth


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

    q0 = justin19_primitives.justin_primitives(justin='getready')
    # q_test0[jtp.JOINTS_ARM_LEFT] = np.array([0, -80, 90, -40, 0, 0, 0]) * np.pi / 180
    # q0[jtp.IDX_J_LEFT] = np.array([0, -80, 0, 0, 0, 0, 0]) * np.pi / 180

    variable_joints = justin19_primitives.get_free_joints(torso=True, right=True, left=True)

    # for f in [str(i) for i in range(10, 20)]:
    for f in ['TEST']:
        for n in [1]:
            calculate_path(directory=f"TorsoRightLeft/{f}/", mode='ordered', n_samples=n)
            no_unnecessary_motion(directory=f"TorsoRightLeft/{f}", mode='ordered', n_samples=n,
                                  variable_joints=variable_joints, q0=q0)


# test_read()
# main()
# n = 50
# q_list = np.load(f'/volume/USERSTORE/tenh_jo/0_Data/Calibration/ordered_poses_{n}.npy')
# q_path_list = calculate_trajectories_between(q_list=q_list)
# np.save(f'/volume/USERSTORE/tenh_jo/0_Data/Calibration/ordered_paths_{n}.npy', q_path_list)

#
# for f in [str(i) for i in range(10, 20)]:
# for f in ['TEST']:
#     n = 1
#     directory = f'TorsoRightLeft/{f}'
#     q_path_list = np.load(ICHR20_CALIBRATION_DATA + f"{directory}/ordered_poses_path_{n}.npy")
#     q_path_list = smooth_paths(q_path_list, verbose=1)
#     write_msgpack(file=ICHR20_CALIBRATION_DATA + f"{directory}/random_poses_smooth_{n}.bin", nested_list=q_path_list)


# q_path_list = np.load('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/C/random_poses_path_50.npy')
# q_path_list = smooth_paths(q_path_list)
# write_msgpack(file='/volume/USERSTORE/tenh_jo/0_Data/Calibration/random_poses_smooth50_C.bin',nested_list=q_path_list)

# q_list = np.load('front_tcp_calibration2.npy')
# q_list = np.load(ICHR20_CALIBRATION_DATA + 'Measurements/front_tcp_calibration_50_fine_ordered_subset.npy')


n = 50
directory = "/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_left3/"
# q_list = np.load(directory + f'ordered_poses_{n}.npy',)

# feasibility_check(q=q_list, par=par)
# q_path_list = calculate_trajectories_between(q_list=q_list)
# np.save(directory + f'ordered_paths_{n}.npy', q_path_list)
#

file = f"/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightLeft/TCP_left3/ordered_paths_{n}.npy"
q_path_list = np.load(file)
q_path_list = smooth_paths(q_path_list)
write_msgpack(file=directory + f'random_poses_smooth_{n}.bin',
              nested_list=q_path_list)
