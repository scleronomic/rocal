import numpy as np

from rokin.Robots import Justin19
from rokin.Robots.Justin19 import justin19_par as jtp
from rocal.Robots import RobotCal


# Bool Parameter
dh_bool_torso = np.array([[2, 9, 3, 3],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
                          [3, 2, 2, 2],
                          [8, 2, 3, 1],
                          [8, 2, 2, 2]]) < 8

dh_bool_torso = np.array([[11, 9, 11, 11],   # NEW 11 | 8 from redundancy analysis, 9 from redundancy with f_world_robot
                          [3, 2, 2, 2],
                          [8, 2, 3, 1],
                          [8, 2, 2, 2]]) < 8

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
f_world_base0 = np.array([[ 0.61727578, -0.78471752,  0.05647137, -0.28828218],
                          [-0.32204662, -0.1865344 ,  0.92816534, -0.48303376],
                          [-0.71781375, -0.5911204 , -0.36785879,  7.84573046],
                          [ 0.        ,  0.        ,  0.        ,  1.        ]])

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
        self.dh = jtp.DH4

        # EL
        self.n_el = len(self.dh)
        self.el = np.zeros((self.n_el, 3))
        self.el_bool_c = np.vstack((el_bool_torso, el_bool_arm, el_bool_arm, el_bool_head))
        self.el = jtp.CP3 * 100

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

        # self.dh = np.array([[ 1.05500e-01,  0.00000e+00,  0.00000e+00,  0.00000e+00],
        #                     [ 5.00000e-05, -1.58197e+00, -2.91000e-03, -1.56989e+00],
        #                     [ 0.00000e+00,  8.72000e-03,  3.03410e-01, -8.86000e-03],
        #                     [ 0.00000e+00, -1.13300e-02,  3.07890e-01,  7.63000e-03],
        #                     [-1.88000e-03,  7.41000e-03,  2.37000e-03, -3.32000e-03],
        #                     [-5.70000e-04, -5.11000e-03,  8.90000e-04,  1.57124e+00],
        #                     [ 3.99460e-01, -1.58469e+00,  9.70000e-04, -1.56972e+00],
        #                     [ 1.30000e-04, -6.37000e-03, -4.10000e-04,  1.57096e+00],
        #                     [ 3.89070e-01, -3.14132e+00, -2.04000e-03, -1.57297e+00],
        #                     [ 6.30000e-04,  1.54959e+00,  9.80000e-04,  1.56503e+00],
        #                     [ 0.00000e+00, -1.57072e+00,  0.00000e+00,  1.56752e+00],
        #                     [ 2.06000e-03, -6.52000e-03,  2.60000e-04,  3.14509e+00],
        #                     [-9.30000e-04,  2.46000e-03,  4.30000e-04,  1.57225e+00],
        #                     [-3.99520e-01, -1.57016e+00, -2.17000e-03, -1.57338e+00],
        #                     [ 1.45000e-03, -2.00000e-04, -7.50000e-04,  1.57321e+00],
        #                     [-3.89310e-01,  1.30300e-02,  1.10000e-03, -1.56782e+00],
        #                     [-1.08000e-03,  1.57410e+00,  2.30000e-04, -1.56643e+00],
        #                     [ 0.00000e+00, -1.57033e+00,  0.00000e+00, -1.57184e+00],
        #                     [ 0.00000e+00,  2.80100e-02,  0.00000e+00,  8.40000e-04],
        #                     [ 0.00000e+00,  6.10000e-04,  0.00000e+00, -1.56749e+00]])
        # self.el = np.array([[ 0.     ,  0.     ,  0.     ],
        #                     [ 0.00678,  0.     ,  0.01124],
        #                     [ 0.00236,  0.     , -0.00058],
        #                     [ 0.     ,  0.     ,  0.01551],
        #                     [-0.00471,  0.     ,  0.03093],
        #                     [ 0.01411,  0.     ,  0.04108],
        #                     [-0.0079 ,  0.     ,  0.04182],
        #                     [ 0.02183,  0.     ,  0.04178],
        #                     [-0.01459,  0.     ,  0.06537],
        #                     [ 0.     ,  0.     ,  0.     ],
        #                     [ 0.     ,  0.     ,  0.     ],
        #                     [-0.00144,  0.     ,  0.03098],
        #                     [ 0.01242,  0.     ,  0.04062],
        #                     [-0.00311,  0.     ,  0.03638],
        #                     [ 0.00773,  0.     ,  0.03177],
        #                     [ 0.00488,  0.     ,  0.02751],
        #                     [ 0.     ,  0.     ,  0.     ],
        #                     [ 0.     ,  0.     ,  0.     ],
        #                     [ 0.     ,  0.     ,  0.     ],
        #                     [ 0.     ,  0.     ,  0.     ]])
        # self.ma = np.array([[ 1.3300e-03,  1.5390e-02,  7.4300e-03,  4.6510e-02],
        #                     [ 2.1859e-01, -1.3020e-02,  6.2200e-03,  5.4700e-02],
        #                     [ 1.5320e-01, -2.2720e-02,  1.9300e-03,  2.9250e-02],
        #                     [ 5.3670e-02,  7.4570e-02,  6.8000e-04,  5.7590e-02],
        #                     [-4.1000e-04,  2.0290e-02, -4.5940e-02,  2.6370e-02],
        #                     [-3.0000e-05,  1.1407e-01,  1.6450e-02,  2.3260e-02],
        #                     [-3.0000e-05,  1.6510e-02, -1.1329e-01,  2.2390e-02],
        #                     [-3.0000e-05,  1.1374e-01,  1.6370e-02,  2.2040e-02],
        #                     [ 1.1000e-04,  2.1900e-02, -1.0562e-01,  1.1010e-02],
        #                     [ 5.0000e-04, -7.8500e-03, -8.7100e-03,  1.6410e-02],
        #                     [ 2.9080e-02,  7.8110e-02,  1.9580e-02,  6.7500e-03],
        #                     [ 0.0000e+00,  0.0000e+00,  1.0000e-01,  2.0000e-02],
        #                     [ 4.1000e-04, -2.0290e-02,  4.5940e-02,  2.6370e-02],
        #                     [ 3.0000e-05, -1.1407e-01,  1.6450e-02,  2.3260e-02],
        #                     [-3.0000e-05, -1.6510e-02,  1.1329e-01,  2.2390e-02],
        #                     [-3.0000e-05, -1.1374e-01, -1.6370e-02,  2.2040e-02],
        #                     [-1.1000e-04,  2.1900e-02,  1.0562e-01,  1.1010e-02],
        #                     [ 5.0000e-04, -7.8500e-03,  8.7100e-03,  1.6410e-02],
        #                     [ 2.9080e-02,  7.8110e-02, -1.9580e-02,  6.7500e-03],
        #                     [ 0.0000e+00,  0.0000e+00,  1.0000e-01,  2.0000e-02],
        #                     [ 1.0000e-02, -1.4000e-01,  0.0000e+00,  3.0000e-02]])

