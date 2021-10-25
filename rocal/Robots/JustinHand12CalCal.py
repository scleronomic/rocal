import numpy as np
from rokin.Robots import JustinHand12Cal

from rocal.Robots import RobotCal


class JustinHand12CalCal(JustinHand12Cal, RobotCal):
    def __init__(self, **kwargs):
        JustinHand12Cal.__init__(self)
        RobotCal.__init__(self, **kwargs)

        self.cm_f_idx = np.array([5, 11, 17, 23])
        self.cm = np.eye(4)[np.newaxis, :, :].repeat(4, axis=0)
        self.n_cm = 4
