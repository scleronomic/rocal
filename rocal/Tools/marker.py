import numpy as np

from wzk import geometry


class Marker:
    __slots__ = ('name',
                 'f_robot_marker',
                 'f_idx_robot',

                 'threshold_orientation',
                 'global_orientation',
                 'threshold_n_cameras',

                 'remove_spheres_f_idx',
                 'inflate_spheres_rad',  # m
                 'axis'
                 )

    def __init__(self, name, f_robot_marker, f_idx_robot,
                 threshold_orientation=None, global_orientation=None, threshold_n_cameras=1,
                 remove_spheres_f_idx=None, inflate_spheres_rad=0.02):
        self.name = name

        self.f_robot_marker = f_robot_marker
        self.f_idx_robot = f_idx_robot

        self.threshold_orientation = threshold_orientation
        self.global_orientation = global_orientation
        self.threshold_n_cameras = threshold_n_cameras

        self.remove_spheres_f_idx = remove_spheres_f_idx
        self.inflate_spheres_rad = inflate_spheres_rad

        self.axis = 2

    def __repr__(self):
        return f"Marker:{self.name}"

    def get_frames(self, f=None,
                   robot=None, q=None):

        if f is None:
            f = robot.get_frames(q)

        if self.f_idx_robot is None:
            f_world_marker = np.zeros(f.shape[:-3] + self.f_robot_marker.shape)
            f_world_marker[...] = self.f_robot_marker

        else:
            f_world_marker = f[..., self.f_idx_robot, :, :] @ self.f_robot_marker

        return f_world_marker

    def check_orientation(self, robot, camera, q):
        """
        The rotation of marker is close to the desired pose (pointing towards the ceiling)
         -> justin_torso_right_10: around 3% fulfil this criterion
        """
        if self.global_orientation is not None:
            angle = geometry.angle_between_vectors(a=self.get_frames(robot=robot, q=q)[..., :-1, self.axis],
                                                   b=self.global_orientation)

        else:
            angle = geometry.angle_between_axis_and_point(f=self.get_frames(robot=robot, q=q),
                                                          p=camera.get_frames(robot=robot, q=q)[..., :-1, -1],
                                                          axis=self.axis)

        feasible = angle < self.threshold_orientation
        return feasible

    def __get_adjusted_spheres(self, robot):

        r_spheres = robot.spheres_rad.copy() + self.inflate_spheres_rad

        if self.remove_spheres_f_idx is not None:
            self.remove_spheres_f_idx = np.atleast_1d(self.remove_spheres_f_idx)
            for i in self.remove_spheres_f_idx:
                r_spheres[robot.spheres_f_idx == i] = -1

        return r_spheres

    def check_occlusion(self, robot, camera, q,
                        threshold=1,  # How many of the cameras need to see the marker
                        ):

        x_spheres = robot.get_spheres(q=q)
        r_spheres = self.__get_adjusted_spheres(robot=robot)

        x_cameras = camera.get_frames(robot=robot, q=q)[..., :-1, -1]
        x_cameras = x_cameras[:, np.newaxis, :]
        x_marker = self.get_frames(robot=robot, q=q)[..., :-1, -1]

        assert x_spheres.ndim == 3
        assert x_cameras.ndim == 3
        assert x_marker.ndim == 2

        n, n_spheres, n_dim = x_spheres.shape
        _, n_cameras, _ = x_cameras.shape

        rays = np.zeros((n, n_cameras, 2, 3))
        rays[:, :, 0, :] = x_cameras
        rays[:, :, 1, :] = x_marker[:, np.newaxis, :]

        b = r_spheres >= 0
        intersection = geometry.ray_sphere_intersection(rays=rays, spheres=x_spheres[:, b, :], r=r_spheres[b])

        feasible = intersection.sum(axis=-1) == 0  # is the view of a camera blocked?
        feasible = feasible.sum(axis=-1) >= threshold  # do enough cameras have a free view?
        return feasible


# My Tools
# Wrist and Floor Markers for KINECT
MARKER_POLE = Marker(name='POLE',
                     f_robot_marker=np.array([[+np.sin(np.pi/6), 0, -np.cos(np.pi/6), +1.3],
                                              [0, 1, 0, 0],
                                              [+np.cos(np.pi/6), 0, +np.sin(np.pi/6), +1.0 - 0.09 - 0.5885],
                                              [0, 0, 0, 1]]),
                     f_idx_robot=0,
                     threshold_orientation=np.deg2rad(25), threshold_n_cameras=1,
                     remove_spheres_f_idx=[25], inflate_spheres_rad=0.03)

MARKER_RIGHT = Marker(name='RIGHT',
                      f_robot_marker=np.array([[0, 0, 1, 0.03],
                                               [1, 0, 0, 0.00],
                                               [0, 1, 0, 0.01],
                                               [0, 0, 0, 1]]),
                      f_idx_robot=13,
                      threshold_orientation=np.deg2rad(25), threshold_n_cameras=1,
                      remove_spheres_f_idx=[13, 25], inflate_spheres_rad=0.03)

MARKER_LEFT = Marker(name='LEFT',
                     f_robot_marker=np.array([[0, 0, 1, 0.03],
                                              [1, 0, 0, 0.00],
                                              [0, 1, 0, 0.01],
                                              [0, 0, 0, 1]]),
                     f_idx_robot=22,
                     threshold_orientation=np.deg2rad(25), threshold_n_cameras=1,
                     remove_spheres_f_idx=[22, 25], inflate_spheres_rad=0.03)


# Back-of-Hand and Head Stars for VICON
MARKER_STAR_RIGHT = Marker(name='STAR_RIGHT',
                           f_robot_marker=np.array([[0, 0, -1, -0.1],
                                                    [0, -1, 0, -0.03],
                                                    [-1, 0, 0, +0.1],
                                                    [0, 0, 0, 1]]),
                           f_idx_robot=13,
                           threshold_orientation=np.deg2rad(25), global_orientation=np.array([0, 0, 1.]), threshold_n_cameras=4,
                           remove_spheres_f_idx=[12, 13], inflate_spheres_rad=0.1)

MARKER_STAR_LEFT = Marker(name='STAR_LEFT',
                          f_robot_marker=np.array([[0, 0, -1, -0.1],
                                                   [-1, 0, 0, +0.03],
                                                   [0, +1, 0, +0.1],
                                                   [0, 0, 0, 1]]),
                          f_idx_robot=22,
                          threshold_orientation=np.deg2rad(25), global_orientation=np.array([0, 0, 1.]), threshold_n_cameras=4,
                          remove_spheres_f_idx=[21, 22], inflate_spheres_rad=0.1)

MARKER_STAR_HEAD = Marker(name='STAR_HEAD',
                          f_robot_marker=np.array([[1, 0, 0, 0.05],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0.30],
                                                   [0, 0, 0, 1]]),
                          f_idx_robot=26,
                          threshold_orientation=np.deg2rad(35), global_orientation=np.array([0, 0, 1.]), threshold_n_cameras=4,
                          remove_spheres_f_idx=[25, 26], inflate_spheres_rad=0.1)
