import numpy as np

from rokin.Robots import JustinFinger03

from rocal.Robots import RobotCal


class JustinFinger03Cal(JustinFinger03, RobotCal):
    def __init__(self, **kwargs):
        JustinFinger03.__init__(self)
        RobotCal.__init__(self, **kwargs)
        self.cm_f_idx = np.array([self.n_frames - 1])
