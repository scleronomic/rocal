from mopla.Kinematic.Robots import Robot


class RobotCal(Robot):
    def __init__(self, dcmf,
                 target_mode='p',
                 cp_loop=0,
                 use_imu=False,
                 add_nominal_offsets=True,
                 include_beta=False):
        self.dcmf = dcmf  # TODO better name, maybe dkmc to make it consistent with the paper
        # self.config_filter = config_filter
        self.target_mode = target_mode  # (p)os (r)ot pr
        self.use_imu = use_imu
        self.add_nominal_offsets = add_nominal_offsets
        self.include_beta = include_beta


        # self.fr0 = fr0
        # self.ma0 = ma0

        # D, DH (Denavit-Hartenberg) parameters, geometric model

        # C, compliances, non-geometric model
        self.cp_loop = cp_loop

        # F, parameters to close the measurement loop

        # M, masses, non-geometric model

        pass