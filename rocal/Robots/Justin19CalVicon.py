import numpy as np

from rokin.Robots import Justin19
from rokin.Robots.Justin19 import justin19_par as jtp
from rocal.Robots import RobotCal


# Bool Parameter
dh_bool_torso = np.array([[2, 9, 3, 3],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
                          [3, 2, 2, 2],
                          [8, 2, 3, 1],
                          [8, 2, 2, 2]]) < 8

# dh_bool_torso = np.array([[11, 9, 11, 11],   # NEW 11 | 8 from redundancy analysis, 9 from redundancy with f_world_robot
#                           [3, 2, 2, 2],
#                           [8, 2, 3, 1],
#                           [8, 2, 2, 2]]) < 8

# dh_bool_torso = np.array([[11, 9, 11, 11],   # NEW 11 | 8 from redundancy analysis, 9 from redundancy with f_world_robot
#                           [3, 2, 9, 2],
#                           [8, 2, 9, 1],
#                           [8, 2, 9, 2]]) < 8

# dh_bool_torso = np.array([[0, 0, 0, 0],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
#                           [0, 1, 0, 0],
#                           [0, 1, 0, 0],
#                           [0, 1, 0, 0]], dtype=bool)
# dh_bool_torso = np.zeros_like(dh_bool_torso, dtype=bool)

# dh_bool_torso = np.array([[2, 9, 3, 3, 0],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
#                           [3, 2, 2, 2, 0],
#                           [8, 2, 9, 1, 1],
#                           [8, 2, 9, 2, 1]]) < 8

dh_bool_arm = np.array([[11, 1, 11, 11],  # 3 -> 11
                        [3, 2, 3, 3],
                        [3, 1, 3, 3],
                        [4, 2, 3, 2],
                        [3, 3, 3, 2],
                        [9, 9, 9, 9],
                        [9, 9, 9, 9]]) <= 8


# dh_bool_arm = np.array([[11, 1, 11, 11],  # 3 -> 11
#                         [9, 2, 3, 3],
#                         [9, 1, 3, 3],
#                         [9, 2, 3, 2],
#                         [9, 3, 3, 2],
#                         [9, 9, 9, 9],
#                         [9, 9, 9, 9]]) <= 8
# dh_bool_arm = np.array([[0, 1, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 0, 0],
#                         [0, 0, 0, 0]], dtype=bool)
# dh_bool_arm = np.zeros_like(dh_bool_arm, dtype=bool)

dh_bool_head = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=bool)  # TODO Not tuned yet
dh_bool_head = np.zeros_like(dh_bool_head, dtype=bool)


# el_bool_torso = np.array([[5, 3, 9],
#                           [5, 5, 3],
#                           [4, 5, 5],
#                           [9, 5, 4]]) <= 4

el_bool_torso = np.array([[0, 0, 0],
                          [1, 0, 1],
                          [1, 0, 1],
                          [0, 0, 1]], dtype=bool)
# el_bool_torso = np.array([[0, 0, 0],
#                           [0, 0, 1],
#                           [0, 0, 1],
#                           [0, 0, 1]], dtype=bool)
#
# el_bool_arm = np.array([[1, 1, 1],
#                         [0, 1, 1],
#                         [1, 0, 1],
#                         [1, 1, 1],
#                         [1, 1, 1],
#                         [0, 0, 1],
#                         [0, 0, 1]], dtype=bool)
# el_bool_arm = np.array([[0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1]], dtype=bool)
el_bool_arm = np.array([[1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0]], dtype=bool)
# el_bool_arm = np.array([[0, 0, 1],
#                         [0, 0, 1],
#                         [0, 0, 1],
#                         [0, 0, 1],
#                         [0, 0, 1],
#                         [0, 0, 0],
#                         [0, 0, 0]], dtype=bool)
# el_bool_arm = np.zeros_like(el_bool_arm, dtype=bool)

# el_bool_head = np.array([[1, 1, 1],
#                          [1, 1, 1]], dtype=bool)
el_bool_head = np.array([[1, 0, 1],
                         [1, 0, 1]], dtype=bool)
el_bool_head = np.zeros_like(el_bool_head, dtype=bool)


cm_bool = np.array([[1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0]], dtype=bool)

# Target Frames
f_world_base0 = np.array([[0,  -1,  0, -2.55],  # Meas 3
                          [+1,  0,  0, +2.0],
                          [0,   0, +1, +0.1],
                          [0,   0,  0, 1]])
# f_world_base0 = np.array([[ 0.61727578, -0.78471752,  0.05647137, -0.28828218],  # Meas 4 / July 22 for  ICHR22
#                           [-0.32204662, -0.1865344 ,  0.92816534, -0.48303376],
#                           [-0.71781375, -0.5911204 , -0.36785879,  7.84573046],
#                           [ 0.        ,  0.        ,  0.        ,  1.        ]])

f_right_target0 = np.array([[0, 0, -1, -0.1],  # Meas 2, left & right
                            [0, -1, 0, -0.03],
                            [-1, 0, 0, +0.1],
                            [0, 0, 0, 1]])
f_left_target0 = np.array([[0, 0, -1, -0.1],
                           [-1, 0, 0, 0.03],
                           [0, 1, 0, 0.05],
                           [0, 0, 0, 1]])
f_head_target0 = np.array([[1, 0, 0, 0.05],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0.25],
                           [0, 0, 0, 1]])


class Justin19CalVicon(Justin19, RobotCal):
    def __init__(self, dkmca, target_mode='p', el_loop=0, use_imu=False, add_nominal_offsets=True,
                 include_beta=False):
        Justin19.__init__(self)
        RobotCal.__init__(self, dkmca=dkmca, target_mode=target_mode,
                          el_loop=el_loop,
                          use_imu=use_imu,
                          add_nominal_offsets=add_nominal_offsets,
                          include_beta=include_beta)

        # DH
        self.n_dh = len(self.dh)
        self.dh_bool_c = np.vstack((dh_bool_torso, dh_bool_arm, dh_bool_arm, dh_bool_head))
        # self.dh = jtp.DH4

        # EL
        self.n_el = len(self.dh)
        self.el = np.zeros((self.n_el, 3))
        self.el_bool_c = np.vstack((el_bool_torso, el_bool_arm, el_bool_arm, el_bool_head))
        # self.el = jtp.CP3 * 100

        # MA
        self.n_ma = len(self.masses)
        self.ma = np.hstack((jtp.MASS_POS[:, :3], jtp.MASSES[:, np.newaxis] / 100))

        # CM
        self.cm = np.stack((f_world_base0, f_right_target0, f_left_target0), axis=0)
        self.cm_f_idx = [jtp.IDX_F_RIGHT_TCP, jtp.IDX_F_LEFT_TCP]
        self.cm_bool_c = cm_bool
        self.n_cm = len(self.cm)

        # AD
        self.n_ad = 0

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
