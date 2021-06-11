import numpy as np
from mopla.Kinematic.Robots import Justin19
from mopla.Justin import parameter_torso as jtp
from rocal.Robots import RobotCal


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

# cp_bool_torso = np.array([[5, 3, 9],
#                           [5, 5, 3],
#                           [4, 5, 5],
#                           [9, 5, 4]]) <= 4

cp_bool_torso = np.array([[1, 0, 0],
                          [1, 0, 1],
                          [1, 0, 1],
                          [0, 0, 1]], dtype=bool)

# cp_bool_torso = np.array([[1, 0, 0],
#                           [1, 0, 1],
#                           [1, 1, 1],
#                           [1, 1, 1]], dtype=bool)

# cp_bool_arm = np.array([[1, 1, 1],
#                         [0, 1, 1],
#                         [1, 0, 1],
#                         [1, 1, 1],
#                         [1, 1, 1],
#                         [0, 0, 1],
#                         [0, 0, 1]], dtype=bool)
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

# cp_bool_head = np.array([[1, 1, 1],
#                          [1, 1, 1]], dtype=bool)

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

p_r_m = [-0.1, -0.03, +0.1]
p_l_m = [-0.1, 0.03, 0.05]


class Justin19Cal(Justin19, RobotCal):
    def __init__(self, **kwargs):
        Justin19.__init__(self)
        RobotCal.__init__(self, **kwargs)

        # DH
        self.n_dh = len(self.dh)
        self.dh_bool_c = np.vstack((dh_bool_torso, dh_bool_arm, dh_bool_arm, dh_bool_head))

        # CP
        self.n_cp = len(self.dh)
        self.cp = np.zeros((self.n_cp, 3))
        self.cp_bool_c = np.vstack((cp_bool_torso, cp_bool_arm, cp_bool_arm, cp_bool_head))

        # FR
        self.fr = np.stack((f_world_base0, f_right_target0, f_left_target0), axis=0)
        self.idx_fr = [jtp.IDX_F_RIGHT_TCP, jtp.IDX_F_LEFT_TCP]
        self.fr_c = fr_bool_robot
        self.n_fr = len(self.fr)


        # MA
        self.n_ma = len(self.masses)
        self.ma = np.hstack((jtp.MASS_POS[:, :3], jtp.MASSES[:, np.newaxis] / 100))
