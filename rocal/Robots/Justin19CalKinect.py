import numpy as np

from rokin.Robots import Justin19
from rokin.Robots.Justin19 import justin19_par as jtp
from rocal.Robots import RobotCal
from rocal.Tools import KINECT, MARKER_POLE, MARKER_LEFT, MARKER_RIGHT


# Bool Parameter
# dh_bool_torso = np.array([[2, 9, 3, 3],   # 8 from redundancy analysis, 9 from redundancy with f_world_robot
#                           [3, 2, 2, 2],
#                           [8, 2, 3, 1],
#                           [8, 2, 2, 2]]) < 8

dh_bool_torso = np.array([[2, 9, 3, 3],   # NEW 11 | 8 from redundancy analysis, 9 from redundancy with f_world_robot
                          [3, 2, 2, 2],
                          [8, 2, 3, 1],
                          [8, 2, 2, 2]]) < 8


dh_bool_arm = np.array([[3, 1, 3, 3],  # 3 -> 11
                        [3, 2, 3, 3],
                        [3, 1, 3, 3],
                        [4, 2, 3, 2],
                        [3, 3, 3, 2],
                        [9, 9, 9, 9],
                        [9, 9, 9, 9]]) <= 8

dh_bool_head = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1]], dtype=bool)  # TODO Not tuned yet
# dh_bool_head = np.zeros_like(dh_bool_head, dtype=bool)

el_bool_torso = np.array([[0, 0, 0],
                          [1, 0, 1],
                          [1, 0, 1],
                          [0, 0, 1]], dtype=bool)

el_bool_arm = np.array([[1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0]], dtype=bool)
# el_bool_arm = np.zeros_like(el_bool_arm, dtype=bool)

el_bool_head = np.array([[1, 0, 1],
                         [1, 0, 1]], dtype=bool)
el_bool_head = np.zeros_like(el_bool_head, dtype=bool)


cm_bool = np.array([[1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0]], dtype=bool)


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
        self.dh_bool_c = np.vstack((dh_bool_torso, dh_bool_arm, dh_bool_arm, dh_bool_head))

        # EL
        self.n_el = len(self.dh)
        self.el = np.zeros((self.n_el, 3))
        self.el_bool_c = np.vstack((el_bool_torso, el_bool_arm, el_bool_arm, el_bool_head))

        # MA
        self.n_ma = len(self.masses)
        self.ma = np.hstack((jtp.MASS_POS[:, :3], jtp.MASSES[:, np.newaxis] / 100))

        # CM
        self.cm = np.stack((MARKER_POLE.f_robot_marker.copy(),
                            MARKER_RIGHT.f_robot_marker.copy(),
                            MARKER_LEFT.f_robot_marker.copy(),
                            KINECT.f_robot_camera.copy(),
                            KINECT.f_robot_camera.copy(),
                            ), axis=0)
        self.cm_f_idx = [MARKER_POLE.f_idx_robot,
                         MARKER_RIGHT.f_idx_robot,
                         MARKER_LEFT.f_idx_robot,
                         KINECT.f_idx_robot,
                         KINECT.f_idx_robot,
                         ]
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
