import numpy as np
from wzk import spatial, geometry, pv, atleast_tuple, verbose_reject_x


class Camera:
    __slots__ = ('name',
                 'focal_length',  # px
                 'center_point',  # px
                 'distortion',

                 'resolution',
                 'frustum',

                 'threshold_frustum',

                 'f_idx_robot',
                 'f_robot_camera',
                 'axis')

    def __init__(self, name, focal_length, center_point, distortion, resolution,
                 frustum, threshold_frustum,
                 f_idx_robot=None, f_robot_camera=None):
        self.name = name
        self.focal_length = focal_length
        self.center_point = center_point
        self.distortion = distortion

        self.resolution = resolution
        self.frustum = frustum

        self.threshold_frustum = threshold_frustum

        self.f_idx_robot = f_idx_robot
        self.f_robot_camera = f_robot_camera
        self.axis = 2

    def __repr__(self):
        return f"Camera:{self.name}"

    @staticmethod
    def p2plane(p):
        u = np.array((p[..., 0], p[..., 1])) / p[..., 2]
        u = np.moveaxis(u, 0, -1)  # TODO
        return u

    def radial_distortion(self, u):
        return u / (1 + self.distortion * u**2)

    def project2image(self, p, distort=True):
        # Pinhole Model
        u = self.p2plane(p=p)
        if distort:
            u = self.radial_distortion(u)
        u = self.center_point + self.focal_length * u
        return u

    def get_frames(self, f=None,
                   robot=None, q=None):

        if f is None:
            f = robot.get_frames(q)

        if self.f_idx_robot is None:
            f_world_camera = np.zeros(f.shape[:-3] + self.f_robot_camera.shape)
            f_world_camera[...] = self.f_robot_camera
        else:
            f_world_camera = f[..., self.f_idx_robot, :, :] @ self.f_robot_camera

        return f_world_camera

    def project_marker2image(self, marker,
                             f=None,
                             robot=None, q=None,
                             distort=True):
        f_world_camera = self.get_frames(f=f, robot=robot, q=q)
        f_world_marker = marker.get_frames(f=f, robot=robot, q=q)

        f_camera_marker = spatial.invert(f_world_camera) @ f_world_marker
        p_camera_marker = f_camera_marker[..., :-1, -1]
        return self.project2image(p=p_camera_marker, distort=distort)

    def check_orientation(self, robot, marker, q):
        """
        idx: which axis is aligned with the lens
        012 <-> xyz

        # TODO can be done exactly with the x and y
        x_camera_marker
        angle_x = np.atan2(x_camera_marker[0], x_camera_marker[2])
        angle_y = np.atan2(x_camera_marker[1], x_camera_marker[2])
        """
        angle = geometry.angle_between_axis_and_point(f=self.get_frames(robot=robot, q=q),
                                                      p=marker.get_frames(robot=robot, q=q)[..., :-1, -1],
                                                      axis=self.axis)

        feasible = angle < self.threshold_frustum
        return np.array(feasible, dtype=bool)

    def orientate_towards_marker(self, robot, marker, q, clip=False, verbose=0):
        raise NotImplementedError

    def adjust_and_check_orientation(self, robot, marker, q):
        try:
            q = self.orientate_towards_marker(robot=robot, marker=marker, q=q)
        except NotImplementedError:
            pass

        feasible = self.check_orientation(robot=robot, marker=marker, q=q)

        q = verbose_reject_x(title='Camera - Marker Orientation', x=q, b=feasible)
        return q


class CameraOnRobot(Camera):
    def __init__(self, name, focal_length, center_point, distortion,
                 resolution, frustum,
                 threshold_frustum,
                 f_idx_robot=None, f_robot_camera=None
                 ):
        super().__init__(name=name, focal_length=focal_length, center_point=center_point, distortion=distortion,
                         resolution=resolution, frustum=frustum,
                         threshold_frustum=threshold_frustum,
                         f_idx_robot=f_idx_robot, f_robot_camera=f_robot_camera)

    def __get_q_neck(self, robot, marker, q):
        """
        for this to work the general orientation of the camera must be known.
        In this case
        x: up
        y: right
        z: camera
        # TODO write general for arbitrary robot, this than has to be done via optimization
        """

        f_world_camera = self.get_frames(robot=robot, q=q)
        f_world_marker = marker.get_frames(robot=robot, q=q)

        f_camera_marker = spatial.invert(f_world_camera) @ f_world_marker
        p_camera_marker = f_camera_marker[..., :-1, -1]
        q_pan = np.arctan2(p_camera_marker[..., 2], p_camera_marker[..., 1]) - np.pi / 2
        q_tilt = -(np.pi / 2 - np.arctan2(np.sqrt(p_camera_marker[..., 1] ** 2 + p_camera_marker[..., 2] ** 2),
                                          p_camera_marker[..., 0]))
        q_neck = np.vstack([q_pan, q_tilt]).T
        return q_neck

    @staticmethod
    def __adapt_q_neck(q, q_neck):
        q2 = q.copy()
        q2[..., -2:] += q_neck
        q2 = spatial.angle2minuspi_pluspi(q2)
        return q2

    def orientate_towards_marker(self, robot, marker, q, clip=False, verbose=0):
        n = 10

        q = q.copy()
        for i in range(n):
            q_neck = self.__get_q_neck(robot=robot, marker=marker, q=q)
            q = self.__adapt_q_neck(q=q, q_neck=q_neck)
            if verbose > 1:
                print(i, np.rad2deg(self.check_orientation(robot=robot, marker=marker, q=q)))

        if clip:
            q = robot.prune_joints2limits(q=q, safety_eps=1e-6)

        return q


class CamerasAndMarkers:
    __slots__ = ('cameras',
                 'markers',
                 'robot',
                 'relation')

    def __init__(self, cameras, markers, robot, relation='all4all'):
        self.cameras, self.markers = atleast_tuple(cameras, markers)
        self.robot = robot
        self.relation = relation

    def get_pairs(self):
        if self.relation == 'all4all':
            pairs = [(c, m) for c in self.cameras for m in self.markers]

        elif self.relation == 'one2one':
            assert len(self.cameras) == len(self.markers)
            pairs = [(c, m) for c, m in zip(self.cameras, self.markers)]

        else:
            raise ValueError

        return pairs

    def loop_trough_pairs(self, title, q, fun, mode='reduce'):

        if mode == 'reduce':
            for i, (camera, marker) in enumerate(self.get_pairs()):
                feasible = fun(camera=camera, marker=marker, q_=q)
                q = verbose_reject_x(title=title.format(camera=repr(camera), marker=repr(marker)), x=q, b=feasible)
            return q

        elif mode == 'collect':
            res = []
            for i, (camera, marker) in enumerate(self.get_pairs()):
                res.append(fun(camera=camera, marker=marker, q_=q))
            return res

        elif mode == 'reduce_q':
            for i, (camera, marker) in enumerate(self.get_pairs()):
                q = fun(camera=camera, marker=marker, q_=q)
            return q
        else:
            raise ValueError

    def adjust_and_check_camera_orientation(self, q):
        return self.loop_trough_pairs(title='{camera} -> {marker} \tOrientation', q=q, mode='reduce_q',
                                      fun=lambda camera, marker, q_:
                                      camera.adjust_and_check_orientation(robot=self.robot, marker=marker, q=q_))

    def check_camera_orientations(self, q, mode='reduce'):
        return self.loop_trough_pairs(title='{camera} -> {marker} \tOrientation', q=q, mode=mode,
                                      fun=lambda camera, marker, q_:
                                      camera.check_orientation(robot=self.robot, marker=marker, q=q_))

    def check_marker_orientations(self, q, mode='reduce'):
        return self.loop_trough_pairs(title='{camera} <- {marker}\tOrientation', q=q, mode=mode,
                                      fun=lambda camera, marker, q_:
                                      marker.check_orientation(robot=self.robot, camera=camera, q=q_))

    def check_marker_occlusions(self, q):
        title = '{camera} <-> {marker} \tOcclusion'
        feasible = self.loop_trough_pairs(title=title, q=q, mode='collect',
                                          fun=lambda camera, marker, q_:
                                          marker.check_occlusion(robot=self.robot, camera=camera, q=q_))
        for i, (c, m) in enumerate(self.get_pairs()):
            verbose_reject_x(title=title.format(camera=c, marker=m), b=feasible[i], x=np.empty_like(feasible[i]))

        feasible = np.array(feasible)
        feasible = feasible.reshape((len(self.cameras), len(self.markers), -1))  # TODO only works for all4all
        feasible = feasible.sum(axis=0) >= np.array([m.threshold_n_cameras for m in self.markers])[:, np.newaxis]
        feasible = feasible.sum(axis=0) == len(self.markers)
        q = verbose_reject_x(title="\t" + title.format(camera='Cameras', marker='Markers'), b=feasible, x=q)

        return q

# In the end the marker has to be correctly orientated towards the camera:
#   camera <       > marker
# The disadvantage here is that
#   A) the thresholds for camera frustum and marker orientation must be combined into one criterion.
#   B) camera_orientation can be solved via IK of the neck, the full problem not.

# One can ensure this by making two independent checks from either side:
# camera_orientation  |  checking if marker is inside camera frustum:
#   camera <       x marker
#
# marker_orientation  |  checking if marker points towards camera:
#   camera x       > marker


# My Cameras
def define_vicon(verbose=0):
    p_vicon = np.array([[6077.15865599469, 6865.63869380712, 3459.09046758155],
                        [6459.22910380056, -2223.04094947304, 3536.0336791628],
                        [6290.30342915698, 2523.45970996789, 3479.58439388084],
                        [-4471.83982524364, 6605.68129326139, 3535.35709930714],
                        [-5848.87908174607, 1865.37341366668, 3612.22699719695],
                        [-4129.2376834596, -2428.24745395319, 3642.60795853699]]) / 1000

    # as measured at 2020-08-17
    f_world_base = np.array([[1, 0, 0, 0.92],
                             [0, 1, 0, 1.93],
                             [0, 0, 1, 0.07],
                             [0, 0, 0, 1.]])

    f_world_base[:3, -1] += 0.1
    p_vicon = p_vicon - f_world_base[:3, -1]

    f_vicon = np.zeros((6, 4, 4))
    f_vicon[:, :-1, :-1] = np.array([[+1., 0., 0.],
                                     [0., -1., 0.],
                                     [0., 0., -1.]])
    f_vicon[:, :-1, -1] = p_vicon
    if verbose > 0:
        if verbose == 2:
            from wzk.mpl import new_fig, grid_lines_data
            fig, ax = new_fig(aspect=1)
            ax.plot(*p_vicon[:, :2].T, ls='', c='k', marker='o', label='cameras')
            ax.plot(0, 0, ls='', c='b', marker='o', label='robot')
            grid_lines_data(ax=ax, x=p_vicon)
            ax.legend()
            # save_fig(file='vicon_camera_poss_2020-08-17', fig=fig, formats='pdf')

        if verbose == 3:
            p = pv.Plotter()
            p.add_axes_at_origin(labels_off=True)
            pv.plot_frames(p=p, f=f_vicon, scale=0.3)
            p.show()

    return tuple(Camera(name=f"VICON-{i}", focal_length=None, center_point=None, distortion=None,
                        frustum=None, resolution=None,
                        threshold_frustum=np.pi,
                        f_idx_robot=None, f_robot_camera=f) for i, f in enumerate(f_vicon))


def define_kinect(verbose=0):
    if verbose > 0:
        pass

    focal_length = 523.1053,
    center_point = np.array([323.9319, 244.0806])
    distortion = 0.023217

    return CameraOnRobot(name='KINECT',
                         focal_length=focal_length,
                         center_point=center_point,
                         distortion=distortion,
                         frustum=np.deg2rad([43, 57]),
                         resolution=(640, 480),
                         threshold_frustum=np.deg2rad(25),
                         f_idx_robot=26,
                         # f_robot_camera=spatial.trans_rotvec2frame(trans=np.array([0.135, 0.002, 0.143]),
                         #                                           rotvec=np.array([2.234, 0.024, 2.198])),
                         # f_robot_camera=np.array([[0.0136515221298320,  0.0035454505680285,  0.9999005279145556,  0.1302487301832420],
                         #                          [0.0189507422071757,  -0.9998150173133284,  0.0032864151510077,  0.0006042403509293],
                         #                          [0.9997272154509622,  0.0189039925681658,  -0.0137161857543143,  0.1462887526215707],
                         #                          [0., 0., 0., 1.]])
                         f_robot_camera=np.array([[ 2.17783050e-02,  2.04164655e-02,  9.99554337e-01, 1.35092871e-01],
                                                  [ 1.79227071e-02, -9.99638769e-01,  2.00276898e-02, 7.00945008e-04],
                                                  [ 9.99602162e-01,  1.74785504e-02, -2.21363563e-02, 1.74262252e-01],
                                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))


VICON = define_vicon(verbose=0)
KINECT = define_kinect(verbose=0)


def test_vicon():
    from rokin.Robots import Justin19

    robot = Justin19()

    q = robot.sample_q(100)
    f = np.array([v.get_frames(robot=robot, q=q) for v in VICON])
    print(f.shape)
