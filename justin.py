import numpy as np
import Justin.parameter_torso as jtp
from Kinematic.Robots import Robot, Justin19


# Bool Parameter
dh_bool_torso = np.array([[2, 9, 3, 3],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
                          [3, 2, 2, 2],
                          [8, 2, 3, 1],
                          [8, 2, 2, 2]]) < 8

# dh_bool_torso = np.array([[2, 9, 3, 3, 0],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
#                           [3, 2, 2, 2, 0],
#                           [8, 2, 9, 1, 1],
#                           [8, 2, 9, 2, 1]]) < 8

dh_bool_arm = np.array([[3, 1, 3, 3],
                        [3, 2, 3, 3],
                        [3, 1, 3, 3],
                        [4, 2, 3, 2],
                        [3, 3, 3, 2],
                        [9, 9, 9, 9],                # [2, 1, 3, 2],
                        [9, 9, 9, 9]]) <= 8  # was 2 # [2, 2, 3, 2]]) <= 2


# dh_bool_arm = np.array([[3, 1, 3, 3, 0],
#                         [3, 2, 3, 3, 0],
#                         [3, 1, 3, 3, 0],
#                         [4, 2, 3, 2, 0],
#                         [3, 3, 3, 2, 0],
#                         [9, 9, 9, 9, 0],                # [2, 1, 3, 2],
#                         [9, 9, 9, 9, 0]]) <= 8  # was 2 # [2, 2, 3, 2]]) <= 2


dh_bool_head = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=bool)  # TODO Not tuned yet

# dh_bool_head = np.array([[0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0]], dtype=bool)  # TODO Not tuned yet

cp_bool_torso = np.array([[5, 3, 9],
                          [5, 5, 3],
                          [4, 5, 5],
                          [9, 5, 4]]) <= 4

cp_bool_torso = np.array([[1, 0, 0],
                          [1, 0, 1],
                          [1, 0, 1],
                          [0, 0, 1]], dtype=bool)

# cp_bool_torso = np.array([[1, 0, 0],
#                           [1, 0, 1],
#                           [1, 1, 1],
#                           [1, 1, 1]], dtype=bool)

cp_bool_arm = np.array([[1, 1, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [0, 0, 1],
                        [0, 0, 1]], dtype=bool)
cp_bool_arm = np.array([[1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0]], dtype=bool)

cp_bool_head = np.array([[1, 0, 1],
                         [1, 0, 1]], dtype=bool)
# cp_bool_arm = np.array([[0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1],
#                         [0, 1, 1]], dtype=bool)

cp_bool_head = np.array([[1, 1, 1],
                         [1, 1, 1]], dtype=bool)  # TODO Not tuned yet

fr_bool_robot = np.array([[1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0]], dtype=bool)
# Target Frames
# f_world_base0 = np.array([[1, 0, 0, 1.4],  # Meas 1, only right, dummy
#                           [0, 1, 0, 0.9],
#                           [0, 0, 1, 0.08],
#                           [0, 0, 0, 1]])
# f_right_target0 = np.array([[0, 0, -1, -0.1],  # Meas 1, only right
#                             [1, 0, 0, -0.03],
#                             [0, -1, 0, 0.05],
#                             [0, 0, 0, 1]])
# f_world_base0 = np.array([[1, 0, 0, 0.90],  # Meas 2, left & right, better centered with respect ot the cameras
#                           [0, 1, 0, 1.90],
#                           [0, 0, 1, 0.07],
#                           [0, 0, 0, 1]])

f_world_base0 = np.array([[ 0, -1,  0, -2.55],  # Meas 3
                          [+1,  0,  0, +2.0],
                          [ 0,  0, +1, +0.1],
                          [ 0,  0,  0, 1    ]])
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

p_r_m = [-0.1, -0.03, +0.1]
p_l_m = [-0.1, 0.03, 0.05]


class Justin19Calib(Justin19):

    def __init__(self, dcmf, ma0=True, fr0=True, config_filter='ff', target_mode='p',
                 cp_loop=0, use_imu=False, add_nominal_offsets=True,
                 include_beta=False):

        self.dcmf = dcmf
        self.config_filter = config_filter
        self.target_mode = target_mode  # (p)os (r)ot pr
        self.use_imu = use_imu
        self.add_nominal_offsets = add_nominal_offsets

        self.include_beta = include_beta

        super().__init__()
        # DH
        self.n_dh = len(self.dh)
        self.dh_bool_c = np.vstack((dh_bool_torso, dh_bool_arm, dh_bool_arm, dh_bool_head))

        # CP
        self.n_cp = len(self.dh)
        self.cp = np.zeros((self.n_cp, 3))
        self.cp_loop = cp_loop
        self.cp_bool_c = np.vstack((cp_bool_torso, cp_bool_arm, cp_bool_arm, cp_bool_head))

        # FR
        self.fr0 = fr0
        self.n_fr = 3
        if fr0:
            self.fr = np.stack((f_world_base0, f_right_target0, f_left_target0), axis=0)
        else:
            self.fr = np.stack((np.eye(4), np.eye(4), np.eye(4)), axis=0)

        self.idx_fr = [jtp.IDX_F_RIGHT_TCP, jtp.IDX_F_LEFT_TCP]
        self.fr_c = fr_bool_robot

        # MA
        self.ma0 = ma0
        self.n_ma = len(self.masses)
        if ma0:
            self.ma = np.hstack((jtp.MASS_POS[:, :3], jtp.MASSES[:, np.newaxis] / 100))
        else:
            self.ma = np.zeros((self.n_ma, 4))



# rad / Nm
# N = kg * m / s2
# 1kg * 1m -> 10Nm,

#  10 kN / rad

#  compliance      *  m * kg * g  [in mm]
# (1 / (20 * 1000) *  1 * 10 * 10) * 1000  -> 5mm

# FINDING
#   stiffness [Nm / rad] ~ cp [rad / Nm]->  1e4 ~ 1e-4
#   cp ~ 5e-5 <-> 20kN/rad (don't forget factor x100)
