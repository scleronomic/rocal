import numpy as np

from rokin.Robots import Justin19
from rokin.Robots.Justin19 import justin19_par
from rocal.Robots import RobotCal
from rocal.Tools import KINECT, MARKER_POLE, MARKER_LEFT, MARKER_RIGHT


# Bool Parameter
dh_bool_torso = np.array([[2, 9, 3, 3],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
                          [3, 2, 2, 2],
                          [8, 2, 3, 1],
                          [8, 2, 2, 2]]) < 8

dh_bool_torso = np.zeros_like(dh_bool_torso, dtype=bool)


# dh_bool_arm = np.array([[8, 1, 8, 3],  # 3 -> 11
#                         [13, 2, 13, 3],
#                         [12, 8, 13, 3],   # 12 justin ellen
#                         [13, 2, 13, 2],
#                         [12, 3, 13, 2],
#                         [13, 3, 13, 3],
#                         [11, 8, 11, 8]]) < 8

dh_bool_right = np.array([[9, 1, 9, 1],  # 3 -> 11
                          [9, 1, 9, 1],
                          [9, 1, 9, 1],   # 12 justin ellen
                          [9, 7, 9, 7],
                          [9, 7, 9, 7],
                          [9, 7, 9, 7],
                          [9, 9, 9, 9]]) < 5

dh_bool_left = np.array([[9, 1, 9, 1],  # 3 -> 11
                         [9, 1, 9, 1],
                         [9, 1, 9, 1],   # 12 justin ellen
                         [9, 7, 9, 7],
                         [9, 7, 9, 7],
                         [9, 7, 9, 7],
                         [9, 9, 9, 9]]) < 5
# dh_bool_left = np.array([[13, 12, 13, 13],  # 3 -> 11
#                          [13, 12, 13, 13],
#                          [13, 12, 13, 13],   # 12 justin ellen
#                          [13, 12, 13, 13],
#                          [13, 12, 13, 13],
#                          [13, 12, 13, 13],
#                          [13, 8, 13, 13]]) < 8


dh_bool_head = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=bool)  # TODO Not tuned yet
# dh_bool_head = np.zeros_like(dh_bool_head, dtype=bool)

el_bool_torso = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [1, 1, 1],
                          [0, 0, 1]], dtype=bool)

# el_bool_arm = np.array([[1, 0, 1],
#                         [1, 0, 1],
#                         [1, 0, 1],
#                         [0, 0, 0],
#                         [0, 0, 0],
#                         [0, 0, 0],
#                         [0, 0, 0]], dtype=bool)
el_bool_right = np.array([[1, 0, 1],
                          [1, 0, 1],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]], dtype=bool)
#
el_bool_left = np.array([[1, 0, 1],
                         [1, 0, 1],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]], dtype=bool)
# el_bool_arm = np.zeros_like(el_bool_arm, dtype=bool)

el_bool_head = np.array([[1, 0, 1],
                         [1, 0, 1]], dtype=bool)
el_bool_head = np.zeros_like(el_bool_head, dtype=bool)


cm_bool = np.array([[1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1]], dtype=bool)


class Justin19CalKinect(Justin19, RobotCal):
    def __init__(self, dkmca, target_mode='p', el_loop=0, use_imu=False, add_nominal_offsets=True, include_beta=False):
        Justin19.__init__(self)
        RobotCal.__init__(self, dkmca=dkmca, target_mode=target_mode,
                          el_loop=el_loop,
                          use_imu=use_imu,
                          add_nominal_offsets=add_nominal_offsets,
                          include_beta=include_beta)

        # DH
        self.n_dh = len(self.dh)
        self.dh_bool_c = np.vstack((dh_bool_torso, dh_bool_right, dh_bool_left, dh_bool_head))
        self.dh = justin19_par.DH4

        # EL
        self.n_el = len(self.dh)
        self.el = np.zeros((self.n_el, 3))
        self.el = justin19_par.CP3 * 100
        self.el_bool_c = np.vstack((el_bool_torso, el_bool_right, el_bool_left, el_bool_head))

        # MA
        self.n_ma = len(self.masses)
        self.ma = np.hstack((justin19_par.MASS_POS[:, :3], justin19_par.MASSES[:, np.newaxis] / 100))

        # CM
        self.cm = np.stack(( # np.eye(4),
                            MARKER_POLE.f_robot_marker.copy(),
                            MARKER_RIGHT.f_robot_marker.copy(),
                            MARKER_LEFT.f_robot_marker.copy(),
                            KINECT.f_robot_camera.copy()), axis=0)
        self.cm_f_idx = [ # 0,
                         MARKER_POLE.f_idx_robot,
                         MARKER_RIGHT.f_idx_robot,
                         MARKER_LEFT.f_idx_robot,
                         KINECT.f_idx_robot]
        self.cm_bool_c = cm_bool
        self.n_cm = len(self.cm)

        # AD
        self.n_ad = 4

        # Camera-Marker System
        self.marker_pole = MARKER_POLE
        self.marker_right = MARKER_RIGHT
        self.marker_left = MARKER_LEFT
        self.kinect = KINECT

        self.kinect_focal_length = self.kinect.focal_length
        self.kinect_center_point = self.kinect.center_point.copy()
        self.kinect_distortion = self.kinect.distortion

        # self.dh = np.array([[1.05500000e-01, 0.00000000e+00, 0.00000000e+00,
        #                      0.00000000e+00],
        #                     [8.67195330e-04, -1.58524142e+00, -3.13009809e-03,
        #                      -1.56995087e+00],
        #                     [0.00000000e+00, 1.29347621e-02, 3.02722144e-01,
        #                      -8.40427550e-03],
        #                     [0.00000000e+00, -1.08180244e-02, 3.08323342e-01,
        #                      1.02677069e-02],
        #                     [0.00000000e+00, 5.86458216e-03, 0.00000000e+00,
        #                      0.00000000e+00],
        #                     [-9.02478960e-04, -7.05040383e-03, -6.56354280e-04,
        #                      1.57084281e+00],
        #                     [3.99991236e-01, -1.58589415e+00, 2.22001030e-04,
        #                      -1.56943791e+00],
        #                     [1.37334055e-03, -1.55232787e-03, -6.29269137e-04,
        #                      1.57047919e+00],
        #                     [3.89526215e-01, -3.14693491e+00, 6.05240741e-04,
        #                      -1.57456158e+00],
        #                     [0.00000000e+00, 1.57080000e+00, 0.00000000e+00,
        #                      1.57080000e+00],
        #                     [0.00000000e+00, -1.57080000e+00, 0.00000000e+00,
        #                      1.57080000e+00],
        #                     [0.00000000e+00, -1.26418596e-03, 0.00000000e+00,
        #                      3.14159000e+00],
        #                     [-2.16380901e-04, -4.70713280e-03, -7.84930950e-06,
        #                      1.56982725e+00],
        #                     [-3.99090594e-01, -1.57112237e+00, -3.84066834e-04,
        #                      -1.57179751e+00],
        #                     [-1.02403713e-03, 1.97798411e-03, -1.16415193e-04,
        #                      1.56973713e+00],
        #                     [-3.90065771e-01, 7.98451331e-03, -5.65576178e-04,
        #                      -1.57153232e+00],
        #                     [0.00000000e+00, 1.57080000e+00, 0.00000000e+00,
        #                      -1.57080000e+00],
        #                     [0.00000000e+00, -1.57080000e+00, 0.00000000e+00,
        #                      -1.57080000e+00],
        #                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        #                      0.00000000e+00],
        #                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        #                      -1.57080000e+00]])
        # self.el = np.array([[0., 0., 0.],
        #                     [0.0045964, 0., 0.01086758],
        #                     [0.00370492, 0., -0.00206211],
        #                     [0., 0., 0.01619614],
        #                     [-0.00477244, 0., 0.03411075],
        #                     [0.01849401, 0., 0.04028375],
        #                     [-0.02075699, 0., 0.03435228],
        #                     [0.01992572, 0., 0.04312094],
        #                     [0.02462048, 0., -0.03540452],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.],
        #                     [-0.00278014, 0., 0.03166792],
        #                     [0.01630589, 0., 0.03929686],
        #                     [-0.00611156, 0., 0.02365998],
        #                     [-0.00900408, 0., 0.02287086],
        #                     [0.03482747, 0., 0.15924367],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.],
        #                     [0., 0., 0.]])

        # self.dh = self.dh = np.array([[ 1.05500000e-01,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
        #                               [-4.02295400e-04, -1.56130712e+00, -2.50162671e-03, -1.57038353e+00],
        #                               [ 0.00000000e+00, -2.75955106e-02,  3.10656736e-01, -7.91599239e-03],
        #                               [ 0.00000000e+00,  8.13520705e-03,  3.15801028e-01, 1.09672373e-02],
        #                               [ 0.00000000e+00,  4.03926601e-03,  0.00000000e+00, 0.00000000e+00],
        #                               [-1.34096879e-03, -7.96155985e-03, -7.65680290e-04,
        #                                1.57158712e+00],
        #                               [ 4.05255316e-01, -1.58615444e+00, -3.04514387e-04,
        #                                 -1.56840851e+00],
        #                               [-3.54751328e-03,  2.89468281e-03, -1.72121532e-03,
        #                                1.57291325e+00],
        #                               [ 3.95625944e-01, -3.14747131e+00,  3.14017180e-03,
        #                                 -1.56834031e+00],
        #                               [ 0.00000000e+00,  1.57079633e+00,  0.00000000e+00,
        #                                 1.57079633e+00],
        #                               [ 0.00000000e+00, -1.57079633e+00,  0.00000000e+00,
        #                                 1.57079633e+00],
        #                               [ 0.00000000e+00, -2.10208201e-03,  0.00000000e+00,
        #                                 3.14159265e+00],
        #                               [ 5.07088833e-04, -4.82244797e-03, -1.02843273e-03,
        #                                 1.56906770e+00],
        #                               [-4.04638179e-01, -1.57003033e+00,  1.49674339e-03,
        #                                -1.57139764e+00],
        #                               [-7.88045588e-04,  5.59115165e-03, -1.32662260e-03,
        #                                1.56292617e+00],
        #                               [-3.94334301e-01,  1.83315627e-02, -2.14333815e-03,
        #                                -1.56529161e+00],
        #                               [ 0.00000000e+00,  1.57079633e+00,  0.00000000e+00,
        #                                 -1.57079633e+00],
        #                               [ 0.00000000e+00, -1.57079633e+00,  0.00000000e+00,
        #                                 -1.57079633e+00],
        #                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #                                 0.00000000e+00],
        #                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #                                 -1.57079633e+00]])
