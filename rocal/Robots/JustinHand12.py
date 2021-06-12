import numpy as np
from mopla.Kinematic.Robots import JustinHand12
from rocal.Robots import RobotCal

# t0 = trans_rotvec2frame(trans=np.array([0.01, 0, 0.05]), rotvec=np.zeros(3))
# t1 = trans_rotvec2frame(trans=np.array([0.02, 0.05, -0.05]), rotvec=np.zeros(3))
# t2 = trans_rotvec2frame(trans=np.array([0.03, -0.05, 0.05]), rotvec=np.zeros(3))
# t3 = trans_rotvec2frame(trans=np.array([0.04, 0.05, 0.05]), rotvec=np.zeros(3))


class JustinHand12Cal(JustinHand12, RobotCal):
    def __init__(self, **kwargs):
        JustinHand12.__init__(self)
        RobotCal.__init__(self, **kwargs)

        self.cm_f_idx = [5, 11, 17, 23]
        self.cm = np.eye(4)[np.newaxis, :, :].repeat(4, axis=0)
        self.n_cm = 4
