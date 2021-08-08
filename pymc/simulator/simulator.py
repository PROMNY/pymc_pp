
class Simulator():
    def __init__(self, FK, T_range, obs, obs_conv=None):
        self.FK = FK
        self.obs = obs
        if obs_conv:
            self.obs_converge = obs_conv
        else:
            self.obs_converge = obs
        self.T_range = T_range

    def run_termalization(self, n_max, mc_step, stop_if_not_converged=False):
        pass
