from rokin.Robots import Robot


class RobotCal(Robot):
    def __init__(self, dkmca,
                 target_mode='p',
                 el_loop=0,
                 use_imu=False,
                 add_nominal_offsets=True,
                 include_beta=False):

        super().__init__()

        self.dkmca = dkmca
        # self.config_filter = config_filter
        self.target_mode = target_mode  # (p)os (r)ot pr
        self.use_imu = use_imu
        self.add_nominal_offsets = add_nominal_offsets
        self.include_beta = include_beta

        # D, DH (Denavit-Hartenberg) parameters, geometric model
        self.n_dh = len(self.dh)

        # K, elasticities, non-geometric model
        self.el_loop = el_loop
        self.n_el = len(self.dh)
        self.el = 0

        # M, masses, non-geometric model
        self.n_ma = 0
        self.ma = 0

        # C, parameters to close the measurement loop
        self.n_cm = 0
        self.cm = 0

        # A, additional parameters the user might want to use
        self.n_a = 0
        self.ad = 0
