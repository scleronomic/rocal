import numpy as np
from wzk import pv, verbose_reject_x, tictoc, spatial, safe_mkdir, find_array_occurrences
from wzk import new_fig, save_fig

from rokin.Robots.Justin19 import Justin19, justin19_primitives
from rokin.Vis import robot_3d
from mopla.Optimizer import feasibility_check
from mopla.Parameter import get_par_justin19

from rocal.Measurements.Configurations.combine_configurations import order_configurations
from rocal.definitions import ICHR22_AUTOCALIBRATION


class Generation:
    __slots__ = ('name',
                 'par',
                 'camera_marker',
                 'check',
                 'adjust_camera_orientation',
                 'qclose')

    def __init__(self, name, par, camera_marker, check, adjust_camera_orientation, qclose):
        self.name = name
        self.par = par
        self.camera_marker = camera_marker
        self.check = check
        self.adjust_camera_orientation = adjust_camera_orientation
        self.qclose = qclose


class Check:
    __slots__ = ('camera_orientation',
                 'marker_orientation',
                 'marker_occlusion',
                 'robot')

    def __init__(self, camera_orientation, marker_orientation, marker_occlusion ,robot):
        self.camera_orientation = camera_orientation
        self.marker_orientation = marker_orientation
        self.marker_occlusion = marker_occlusion
        self.robot = robot


def sample_q(gen, n):
    q = gen.par.robot.sample_q(n)
    b = gen.qclose != (gen.par.robot.limits[:, 0] - 1)
    q[..., b] = gen.qclose[b]
    return q


def __reject(gen, q):
    if gen.check.marker_orientation:
        q = gen.camera_marker.check_marker_orientations(q=q, mode='reduce')

    if gen.check.marker_occlusion:
        q = gen.camera_marker.check_marker_occlusions(q=q)

    if gen.check.camera_orientation:
        if gen.adjust_camera_orientation:
            q = gen.camera_marker.adjust_and_check_camera_orientation(q=q)

        else:
            q = gen.camera_marker.check_camera_orientations(q=q, mode='reduce')

    if gen.check.robot and len(q) > 0:
        feasible = feasibility_check(q=q[:, np.newaxis, :], par=gen.par) > 0
        q = verbose_reject_x(title='Robot Feasibility', x=q, b=feasible)

    return q


def reject(gen, q):
    # Not the cleanest approach, it is faster to but adjust_camera_orientation (which changes the previous results)
    # at a later stage
    q = __reject(gen=gen, q=q)
    q = __reject(gen=gen, q=q)
    return q


def rejection_sampling(gen, n, m=10000):

    q = np.zeros((0, gen.par.robot.n_dof))

    count = 0
    while len(q) < n:
        print(f"Count {count}: {len(q)} / {n}")
        qq = sample_q(gen=gen, n=m)
        qq = reject(gen=gen, q=qq)

        q = np.concatenate((q, qq), axis=0)
        count += 1

    return q[:n]


def define_generation_parameters(mode):
    from rocal.Tools import (CamerasAndMarkers,
                             VICON, MARKER_RIGHT, MARKER_LEFT, MARKER_POLE,
                             KINECT, MARKER_STAR_RIGHT, MARKER_STAR_LEFT, MARKER_STAR_HEAD)

    par, gd, _ = get_par_justin19()
    par.check.center_of_mass = True
    par.sc.dist_threshold = 0.05

    gen_vicon = Generation(name='vicon-rlh', par=par,
                           camera_marker=CamerasAndMarkers(cameras=VICON,
                                                           markers=(MARKER_STAR_RIGHT, MARKER_STAR_LEFT, MARKER_STAR_HEAD),
                                                           robot=par.robot),
                           check=Check(camera_orientation=False, marker_orientation=True, marker_occlusion=True, robot=True),
                           adjust_camera_orientation=False, qclose=par.robot.limits[:, 0] - 1)

    gen_kinect_pole = Generation(name='kinect-pole', par=par,
                                 camera_marker=CamerasAndMarkers(cameras=KINECT, markers=MARKER_POLE, robot=par.robot),
                                 check=Check(camera_orientation=True, marker_orientation=True, marker_occlusion=True, robot=True),
                                 adjust_camera_orientation=True, qclose=par.robot.limits[:, 0] - 1)

    q_close_right = justin19_primitives.justin19_primitives(torso='getready_high_170', left_arm='side_down_b')
    b = justin19_primitives.justin19_get_free_joints(right=True, head=True)
    q_close_right[b] = (par.robot.limits[:, 0] - 1)[b]
    gen_kinect_right = Generation(name='kinect-right', par=par,
                                  camera_marker=CamerasAndMarkers(cameras=KINECT, markers=MARKER_RIGHT, robot=par.robot),
                                  check=Check(camera_orientation=True, marker_orientation=True, marker_occlusion=True, robot=True),
                                  adjust_camera_orientation=True,
                                  qclose=q_close_right)

    q_close_left = justin19_primitives.justin19_primitives(torso='getready_high_170', right_arm='side_down_b')
    b = justin19_primitives.justin19_get_free_joints(left=True, head=True)
    q_close_left[b] = (par.robot.limits[:, 0] - 1)[b]
    gen_kinect_left = Generation(name='kinect-left', par=par,
                                 camera_marker=CamerasAndMarkers(cameras=KINECT, markers=MARKER_LEFT, robot=par.robot),
                                 check=Check(camera_orientation=True, marker_orientation=True, marker_occlusion=True, robot=True),
                                 adjust_camera_orientation=True,
                                 qclose=q_close_left)

    mode_dict = dict(vicon_rlh=gen_vicon,
                     kinect_pole=gen_kinect_pole,
                     kinect_right=gen_kinect_right,
                     kinect_left=gen_kinect_left)

    return mode_dict[mode]


def add_4corner_head_configurations(robot, camera, marker, q, verbose=1):
    # marker position in image
    # initial:
    #  __________
    # |         |
    # |    X    |
    # |_________|
    #
    # additional:
    #  __________
    # |o       o|
    # |    X    |
    # |o_______o|

    def check_u(u):

        b_in_img_x = np.logical_and(boarder_px < u[..., 0], u[..., 0] < camera.resolution[0] - boarder_px)
        b_in_img_y = np.logical_and(boarder_px < u[..., 1], u[..., 1] < camera.resolution[1] - boarder_px)
        b_in_img = np.logical_and(b_in_img_x, b_in_img_y)

        b_interesting = np.linalg.norm(u - u0[:, np.newaxis, :], axis=-1) > interesting_px
        b = np.logical_and(b_in_img, b_interesting)
        return b

    def get_q4(dq_pan, dq_tilt):
        q4_ = np.repeat(q[:, np.newaxis, :], repeats=4, axis=1)

        q4_[:, 0, -2:] += np.deg2rad([-dq_pan, -dq_tilt])
        q4_[:, 1, -2:] += np.deg2rad([-dq_pan, +dq_tilt])
        q4_[:, 2, -2:] += np.deg2rad([+dq_pan, +dq_tilt])
        q4_[:, 3, -2:] += np.deg2rad([+dq_pan, -dq_tilt])
        return q4_

    def get_q4_and_check_u4(dq_pan, dq_tilt):
        q4_ = get_q4(dq_pan=dq_pan, dq_tilt=dq_tilt)
        q4_ = robot.prune_joints2limits(q=q4_)

        u4_ = camera.project_marker2image(robot=robot, marker=marker, q=q4_, distort=False)
        b4_ = check_u(u=u4_)
        return q4_, b4_

    def plot_u4(_u0, _u4, _b4):
        x_px, y_px = camera.resolution

        directory = f"{ICHR22_AUTOCALIBRATION}/{repr(marker)}/"
        safe_mkdir(directory=directory)

        fig, ax = new_fig(aspect=1, title=repr(marker))
        ax.set_xlim(0, x_px)
        ax.set_ylim(0, y_px)
        ax.plot(*_u0.T, ls='', marker='x', color='k', zorder=10)
        for i in range(0, 4):
            ax.plot(*_u4[_b4[:, i], i, :].T, ls='', marker='o', alpha=0.3)
        if verbose > 10:
            save_fig(f"{directory}/u_distribution", formats='pdf')

        fig, ax = new_fig(title=repr(marker))
        ax.hist(np.rad2deg(q[:, -2]), bins=100, color='blue', alpha=0.5, label='q_pan', density=True)
        ax.hist(np.rad2deg(q[:, -1]), bins=100, color='red', alpha=0.5, label='q_tilt', density=True)
        ax.set_xlabel('Degree')
        ax.legend()
        if verbose > 10:
            save_fig(f"{directory}/q_distribution", formats='pdf')

    dq_pan_max = 18
    dq_tilt_max = 24

    boarder_px = 50
    interesting_px = 100

    u0 = camera.project_marker2image(robot=robot, marker=marker, q=q, distort=False)

    q4 = np.empty((q.shape[0], 4, q.shape[1]))
    b4 = np.zeros((q.shape[0], 4), dtype=bool)

    mean = -1
    for i in np.linspace(1, 0.1, 20):
        q4_i, b4_i = get_q4_and_check_u4(dq_pan=dq_pan_max*i, dq_tilt=dq_tilt_max*i)

        q4[~b4, :] = q4_i[~b4, :]
        b4[~b4] = b4_i[~b4]
        if mean == b4.mean():
            break
        else:
            mean = b4.mean()

    u4 = camera.project_marker2image(robot=robot, marker=marker, q=q4, distort=False)

    if verbose > 0:
        print('Percentage of the 4 corners', b4.mean())

    if verbose > 1:
        plot_u4(u0, u4, b4)

    return q4, b4, u4


def main():
    pass


def rejection_sampling_pole(n):
    gen = define_generation_parameters(mode='kinect_pole')

    q = rejection_sampling(gen=gen, n=n*2, m=1)

    def reject_other_marker(q_, marker=''):
        gen2 = define_generation_parameters(mode=marker)
        gen2.adjust_camera_orientation = False
        gen2.camera_marker.cameras[0].threshold_frustum = max(gen2.camera_marker.cameras[0].frustum) * 1.2
        gen2.camera_marker.markers[0].threshold_orientation = np.pi / 2
        gen2.camera_marker.markers[0].inflate_spheres_rad = -0.05

        q2 = reject(gen2, q_)
        i = find_array_occurrences(a=q_, o=q2)
        q_ = np.delete(q_, i[:, 0], axis=0)
        return q_

    q = reject_other_marker(q_=q, marker='kinect_right')
    q = reject_other_marker(q_=q, marker='kinect_left')
    q = q[:n]

    np.save(f"{ICHR22_AUTOCALIBRATION}/q{n}_random_{gen.name}.npy", q)
    return q


if __name__ == '__main__':
    n = 10000
    gen = define_generation_parameters(mode='vicon_rlh')

    # with tictoc() as _:
    #     q = rejection_sampling(gen=gen, n=n, m=1000)

    # rejection_sampling_pole(n=10000)

    q = np.load(f"{ICHR22_AUTOCALIBRATION}/q{n}_random_{gen.name}.npy")
    # np.save(f"{ICHR22_AUTOCALIBRATION}/q{n}_random_{gen.name}.npy", q)

    q = reject(gen=gen, q=q)


    # add_4corner_head_configurations(robot=gen.par.robot,
    #                                 camera=gen.camera_marker.cameras[0],
    #                                 marker=gen.camera_marker.markers[0],
    #                                 q=q, verbose=20)

    f_fix = gen.camera_marker.markers[2].get_frames(q=q[:1], robot=gen.par.robot)
    # robot_3d.animate_path(robot=Justin19(), q=q,
    #                       kwargs_frames=dict(f_idx_robot=[13, 22, 26], f_fix=f_fix, scale=0.1))

    # q = sample_q(n=n*1000, camera=VICON, marker=MARKER_STAR_LEFT)

    # q0 = justin19_primitives.justin19_primitives(justin='getready_right')[np.newaxis, :]
    # q, route = order_configurations(q=q[:n], q0=q0, time_limit=3)
    #
    # f_fix1 = MARKER_STAR_LEFT.get_frames(robot=Justin19(), q=q)
    # f_fix2 = MARKER_STAR_RIGHT.get_frames(robot=Justin19(), q=q)
    # f_fix = np.concatenate((f_fix1, f_fix2), axis=0)
    # robot_3d.animate_path(robot=Justin19(), q=q,
    #                       kwargs_frames=dict(f_idx_robot=[13, 22, 26], f_fix=f_fix, scale=0.1))


    # directory = f"{ICHR22_AUTOCALIBRATION}/Measurements/Test"
    #
    # # q_path_list = calculate_trajectories_between(q_list=q)
    # q_path_list = np.load(f"{directory}/q_path_{n}.npy")
    #
    # safe_mkdir(directory)
    # np.save(f"{directory}/q_path_{n}.npy", q_path_list)
    # # robot_3d.animate_path(robot=Justin19(), q=np.concatenate(q_path_list, axis=0), kwargs_frames=dict(f_idx_robot=[26], f_fix=F_WORLD_MARKER))
    #
    # q_path_list_smooth = configurations.smooth_paths(q_path_list=q_path_list)
    # np.save(f"{directory}/q_path_smooth_{n}.npy", q_path_list_smooth)
    #
    # write_msgpack(file=f"{directory}/random_poses_floor_10.bin",
    #               nested_list=q_path_list_smooth)



