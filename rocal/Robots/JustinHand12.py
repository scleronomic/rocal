import numpy as np
from mopla.Kinematic.Robots import JustinHand12
from rocal.Robots import RobotCal


class JustinHand12Cal(JustinHand12, RobotCal):
    def __init__(self, **kwargs):
        JustinHand12.__init__(self)
        RobotCal.__init__(self, **kwargs)
