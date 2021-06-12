import numpy as np

from wzk import new_fig, k_farthest_neighbors, tsp
from wzk.spatial import frame_difference, invert


import mopla.Justin.primitives_torso as pp
from mopla.Visualization.pyvista2.pyvista2 import sphere_path_animation
from rocal.Robots import Justin19Cal

robot = Justin19Cal()

# get_frame_arm_tcp = get_frame_arm_tcp_wrapper(file_results='selected_dh_torques_xyz')
# get_frame_arm_tcp = get_frame_arm_tcp_wrapper(file_results='selected_dh_torques_z')


def get_q_rand(n=10000):
    q = robot.sample_q(n)
    validate_q(q)


def get_q_calibration():
    abcde = 'ABCDE'
    q = np.zeros((0, 1, 19))
    for a in abcde:
        q = np.concatenate((q,
                            np.load('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/' +
                                    a + '/random_poses_1000.npy')[:, np.newaxis, :]), axis=0)

    # Finding: The worst cases are pretty similar for the different calibration models -> good
    idx_var0 = np.array([1085, 3394, 4217, 3158, 1083])
    idx_fixed0 = np.array([2023, 3348,  873, 3394,   67, 1085, 4654, 1083, 3244])
    q_list = q[idx_fixed0]
    q_list[..., 0] = 0


def get_q_fixed_tcp():
    q = np.load('/home/tenh_jo/PycharmProjects/aj-mopla/Justin/Calibration/FullBody_19/Validation'
                '/far_front_tcp_calibration.npy')

    #               # far               # rand
    idx = np.array([810, 413, 148, 578, 1433, 1269, 1065, 964])
    q_list = q[idx]
    return q


def get_q_circus():
    q_list = np.zeros((16, 19))
    q_list[0] = pp.justin_primitives(torso='getready', right_arm='getready', left_arm='getready')
    q_list[1] = pp.justin_primitives(torso='getready', right_arm='front_horizontal', left_arm='getready')
    q_list[2] = pp.justin_primitives(torso='getready', right_arm='side_horizontal', left_arm='getready')
    q_list[3] = pp.justin_primitives(torso='getready', right_arm='back_horizontal', left_arm='getready')
    q_list[4] = pp.justin_primitives(torso='zero', right_arm='front_horizontal', left_arm='getready')
    q_list[5] = pp.justin_primitives(torso='zero', right_arm='side_horizontal', left_arm='getready')
    q_list[6] = pp.justin_primitives(torso='zero', right_arm='back_horizontal', left_arm='getready')
    q_list[7] = pp.justin_primitives(torso='zero', right_arm='front_horizontal', left_arm='front_horizontal')
    q_list[8] = pp.justin_primitives(torso='zero', right_arm='side_horizontal', left_arm='front_horizontal')
    q_list[9] = pp.justin_primitives(torso='zero', right_arm='back_horizontal', left_arm='front_horizontal')
    q_list[10] = pp.justin_primitives(torso='zero', right_arm='front_horizontal', left_arm='side_horizontal')
    q_list[11] = pp.justin_primitives(torso='zero', right_arm='side_horizontal', left_arm='side_horizontal')
    q_list[12] = pp.justin_primitives(torso='zero', right_arm='back_horizontal', left_arm='back_horizontal')
    q_list[13] = pp.justin_primitives(torso='zero', right_arm='front_horizontal', left_arm='back_horizontal')
    q_list[14] = pp.justin_primitives(torso='zero', right_arm='side_horizontal', left_arm='back_horizontal')
    q_list[15] = pp.justin_primitives(torso='zero', right_arm='back_horizontal', left_arm='back_horizontal')
    q = q_list[:, np.newaxis, :]
    q_path_list = get_dance(q_list[:, np.newaxis, :])
    file = '/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/Validation/rotated_zeros'
    write(q_list=q_list, q_path_list=q_path_list, file=file)


def validate_q(q):

    q[..., 0] = 0
    frames = robot.get_frames(q=q)

    f_target_cal = get_frame_arm_tcp(q=q)
    f_target = frames[:, 0, 13, :, :]

    trans_diff, rot_diff = frame_difference(f_target_cal, f_target)

    fig, ax = new_fig()
    ax.hist(trans_diff, bins=20)
    x_target = f_target[..., :-1, -1]
    x_target_cal = f_target_cal[..., :-1, -1]
    x_target_diff = x_target - x_target_cal
    x_target_diff_norm = np.linalg.norm(x_target_diff, axis=-1)

    m_dist_x = x_target_diff[np.newaxis, :, :] - x_target_diff[:, np.newaxis, :]
    m_dist_x_norm = np.linalg.norm(m_dist_x, axis=-1)

    worst_rel_diff = m_dist_x_norm.max(initial=0)
    idx_worst_rel_diff = np.unravel_index(np.argmax(m_dist_x_norm), shape=m_dist_x_norm.shape)
    print(f"Worst relative tcp difference between to configurations {worst_rel_diff}m")

    k = 10
    idx_kfn_x_rel = k_farthest_neighbors(x=x_target_diff, k=k, weighting=None)
    # idx_kfn_x_rel = np.random.choice(np.arange(len(q)), shape=k, replace=False)
    print(idx_kfn_x_rel)
    q_kfn_x_rel = q[idx_kfn_x_rel]

    print(x_target_diff_norm[idx_kfn_x_rel])
    print(m_dist_x_norm[idx_kfn_x_rel[:, np.newaxis], idx_kfn_x_rel[np.newaxis, :]])

    fig, ax = new_fig()
    ax.imshow(m_dist_x_norm)
    # q_kfn_x_rel[..., 0] = 0

    # 0 1 3 7 9
    # idx_kfn_x_rel
    sphere_path_animation(q=q_kfn_x_rel, robot=robot,
                          show_frames=np.array([13, 22]))

    fig, ax = new_fig()
    ax.hist(m_dist_x_norm.ravel(), bins=100)

    # n_dof = 10
    # fig, ax = new_fig(n_rows=n_dof)
    # for o in range(n_dof):
    #     ax[o].plot(q[:, :, o].ravel(), x_target_diff_norm, ls='', marker='o')


def get_dance(q_list, verbose=1):
    q_getready = pp.justin_primitives(justin='getready')
    q_zero = pp.justin_primitives(justin='zero')
    q_list = np.concatenate([q_zero, q_list.reshape(-1, 1, np.shape(q_list)[-1])])
    route = tsp.solve_tsp(points=q_list[:, 0, :], time_limit=5, verbose=0)
    q_list = np.concatenate([q_getready, q_list[route], q_zero, q_getready])

    if verbose > 0:
        sphere_path_animation(q=q_list, robot=robot,
                              show_frames=np.array([13, 22]), additional_frames=None)

    q_path_list = combine.calculate_trajectories_between(q_list=q_list, par=combine.par, gd=combine.gd)

    q_path_list_smooth = combine.smooth_paths(q_path_list=q_path_list, mean_vel_q=0.3, delta=2)

    if verbose > 0:
        plot_smooth_dance(q_path_list_smooth=q_path_list_smooth)

    return q_path_list_smooth


def plot_smooth_dance(q_path_list_smooth=None):
    sphere_path_animation(q=np.concatenate(q_path_list_smooth), robot=combine.par.robot,
                          show_frames=np.array([13, 22]), additional_frames=None)


def write(q_list, q_path_list, file):
    np.save(file=file + '_points.npy', arr=q_list)
    combine.write_msgpack(nested_list=q_path_list, file=file + '_paths.bin')


def get_calibrated_path(points_file):

    file_wo = points_file[:-len('points.npy')]

    q_list = np.load(points_file)[1:-1].reshape(-1, 1, 19)

    frame_list = robot.get_frames(q=q_list)[:, :, jtp.IDX_F_RIGHT_TCP, :, :]
    q_active_bool = np.zeros(19, dtype=bool)
    q_active_bool[:10] = True
    q_list_cal = get_corrected_q(q_list[:, 0, :], frame_list,
                                 q_active_bool, get_frame=get_frame_arm_tcp)

    q_path_list_smooth = get_dance(q_list=q_list, verbose=2)

    # write(q_list=q_list_cal, q_path_list=q_path_list_smooth, file=file_wo + 'cal')


get_calibrated_path('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/Validation/fixed0_points.npy')
# get_calibrated_path('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/Validation/var0_points.npy')
# get_calibrated_path('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/Validation/fixed_tcp_points.npy')
# get_calibrated_path('/volume/USERSTORE/tenh_jo/0_Data/Calibration/TorsoRightArm/Validation/rotated_zeros_points.npy')


def visualize_relative_error():
    # if file_res is None:
    #     file_res = 'mass_01'
    # file_res = f"{directory}results/{file_res}.npy"
    #
    # file_data = file_100
    # (q_meas, target_meas) = load_measurements_right_head(file=file_data)
    x, x_dict = load_results(file=file_res)

    f_world_targetArm, f_world_targetHead = np.split(target_meas, 2, axis=1)
    x_wrapper = create_x_wrapper(**x_dict)
    get_target = create_wrapper_kinematic(x_wrapper=x_wrapper, robot=robot)
    f_head, f_right = np.split(get_target(q=q_meas, x=x), 2, axis=1)

    x_dict0 = set_bool_dict_false(x0=x, x_dict=x_dict)
    x_wrapper0 = create_x_wrapper(**x_dict0)
    x0 = x[:3 * 6]
    get_target0 = create_wrapper_kinematic(x_wrapper=x_wrapper0, robot=robot)
    f_head0, f_right0 = np.split(get_target0(q=q_meas, x=x0), 2, axis=1)

    # d = q - justin_primitives(justin='getready')
    # d_norm = np.linalg.norm(d[0], axis=-1)
    # print(min(d_norm))

    f_relative_meas = (invert(f_world_targetHead[-1]) @ f_world_targetArm[:-1])[:, np.newaxis]
    f_relative_cal = invert(f_head[-1]) @ f_right[:-1]
    f_relative_cal0 = invert(f_head0[-1]) @ f_right0[:-1]

    f_meas = np.concatenate([f_relative_meas, f_relative_meas], axis=1)
    f_cal = np.concatenate([f_relative_cal, f_relative_cal0], axis=1)
    plot_frame_difference(f0=f_meas, f1=f_cal,
                          frame_names=('Calibration', 'No Calibration'), verbose=2)

    save_plots(file_res+'relative')

