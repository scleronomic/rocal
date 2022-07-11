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

        # EL
        self.n_el = len(self.dh)
        self.el = np.zeros((self.n_el, 3))
        self.el_bool_c = np.vstack((el_bool_torso, el_bool_arm, el_bool_arm, el_bool_head))

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

        self.dh = np.array([[ 1.0550e-01,  0.0000e+00,  0.0000e+00, -1.9200e-02],
          [ 7.3000e-03, -1.5875e+00, -8.0000e-04, -1.5730e+00],
          [ 0.0000e+00,  6.4000e-03,  3.0870e-01, -9.5000e-03],
          [ 0.0000e+00, -1.2200e-02,  3.1160e-01,  1.1500e-02],
          [ 1.0000e-03,  5.9000e-03, -1.0200e-02,  1.9000e-03],
          [-3.0000e-04, -6.4000e-03, -1.0000e-04,  1.5707e+00],
          [ 3.9330e-01, -1.5859e+00,  3.0000e-04, -1.5700e+00],
          [ 6.0000e-04, -5.5000e-03, -4.0000e-04,  1.5694e+00],
          [ 3.8380e-01, -3.1424e+00, -1.6000e-03, -1.5729e+00],
          [ 3.0000e-04,  1.5595e+00, -0.0000e+00,  1.5686e+00],
          [-2.4000e-03, -1.5712e+00, -2.0000e-04,  1.5696e+00],
          [-1.1000e-03, -3.5000e-03,  1.1500e-02,  3.1457e+00],
          [-8.0000e-04,  9.0000e-04,  4.0000e-04,  1.5717e+00],
          [-3.9270e-01, -1.5708e+00, -8.0000e-04, -1.5731e+00],
          [ 3.0000e-04, -6.0000e-04, -4.0000e-04,  1.5714e+00],
          [-3.8340e-01,  9.9000e-03,  8.0000e-04, -1.5697e+00],
          [-1.1000e-03,  1.5814e+00,  3.0000e-04, -1.5662e+00],
          [ 6.4000e-03, -1.5709e+00,  6.0000e-04, -1.5680e+00],
          [ 8.7000e-03,  2.6800e-02, -1.0000e-03,  3.0000e-04],
          [ 0.0000e+00, -7.1000e-03, -3.1000e-03, -1.5682e+00]])
        self.el = np.array([[ 0.     , 0. ,     0.    ],
             [ 0.0123 , 0. ,     0.0078],
             [-0.0022 , 0. ,    -0.004 ],
             [ 0.     , 0. ,     0.0301],
             [ 0.0036 , 0. ,     0.0275],
             [ 0.0126 , 0. ,     0.044 ],
             [-0.0048 , 0. ,     0.0432],
             [ 0.0198 , 0. ,     0.0485],
             [-0.0171 , 0. ,     0.0645],
             [ 0.     , 0. ,     0.    ],
             [ 0.     , 0. ,     0.    ],
             [ 0.0017 , 0. ,     0.0295],
             [ 0.0075 , 0. ,     0.053 ],
             [ 0.0063 , 0. ,     0.0495],
             [ 0.005  , 0. ,     0.038 ],
             [ 0.0028 , 0. ,     0.0271],
             [ 0.     , 0. ,     0.    ],
             [ 0.     , 0. ,     0.    ],
             [ 0.0051 , 0. ,    -0.0007],
             [ 0.0008 , 0. ,     0.0059]])
        self.ma = np.array([[ 1.300e-03,  1.540e-02,  7.400e-03,  4.650e-02],
                            [ 2.183e-01, -1.240e-02,  6.700e-03,  5.620e-02],
                            [ 1.533e-01, -2.250e-02,  1.500e-03,  3.080e-02],
                            [ 5.360e-02,  7.520e-02,  2.000e-04,  5.740e-02],
                            [ 1.900e-03,  2.400e-02, -4.610e-02,  2.800e-02],
                            [ 1.500e-03,  1.142e-01,  1.440e-02,  1.880e-02],
                            [-1.040e-02,  1.010e-02, -1.152e-01,  2.060e-02],
                            [-5.300e-03,  1.142e-01,  2.350e-02,  2.240e-02],
                            [-0.000e+00,  2.330e-02, -1.044e-01,  1.600e-03],
                            [ 7.000e-04, -1.250e-02, -4.800e-03,  1.500e-02],
                            [ 2.780e-02,  7.700e-02,  2.200e-02,  7.100e-03],
                            [-1.900e-03,  5.000e-03,  9.980e-02,  1.680e-02],
                            [ 2.200e-03, -1.770e-02,  4.570e-02,  1.820e-02],
                            [-5.000e-03, -1.180e-01,  1.410e-02,  2.070e-02],
                            [ 1.800e-03, -1.800e-02,  1.101e-01,  2.530e-02],
                            [ 3.900e-03, -1.119e-01, -1.660e-02,  2.350e-02],
                            [-1.900e-03,  2.220e-02,  1.074e-01,  1.010e-02],
                            [ 2.100e-03, -8.100e-03,  9.800e-03,  1.000e-02],
                            [ 2.850e-02,  7.750e-02, -2.030e-02, -1.900e-03],
                            [ 1.100e-03,  1.900e-03,  1.010e-01,  1.740e-02],
                            [ 1.160e-02, -1.395e-01, -2.600e-03,  3.200e-02]])

