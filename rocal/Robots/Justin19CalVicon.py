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
    def __init__(self, **kwargs):
        Justin19.__init__(self)
        RobotCal.__init__(self, **kwargs)

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
