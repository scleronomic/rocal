from mopla.Kinematic.Robots import Robot


class RobotCal(Robot):
    def __init__(self, dkmc,
                 target_mode='p',
                 el_loop=0,
                 use_imu=False,
                 add_nominal_offsets=True,
                 include_beta=False):
        
        self.dkmc = dkmc
        # self.config_filter = config_filter
        self.target_mode = target_mode  # (p)os (r)ot pr
        self.use_imu = use_imu
        self.add_nominal_offsets = add_nominal_offsets
        self.include_beta = include_beta

        # self.fr0 = fr0
        # self.ma0 = ma0

        # D, DH (Denavit-Hartenberg) parameters, geometric model
        self.n_dh = len(self.dh)

        # K, elasticities, non-geometric model
        self.el = 0
        self.el_loop = el_loop
        self.n_el = len(self.dh)

        # M, masses, non-geometric model
        self.n_ma = 0
        self.ma = 0

        # F, parameters to close the measurement loop
        self.n_cm = 0
        self.cm = 0
