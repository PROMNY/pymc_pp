from .obs_list import ObsList
from tqdm import trange


class Simulator():
    def __init__(self, FK, solver, obs_list, obs_list_conv=None):
        self.FK = FK
        self.obs = obs_list
        self.solver = solver
        if obs_list_conv:
            self.obs_converge = obs_list_conv
        else:
            self.obs_converge = obs_list

        self.mc_step = 100
        self.converge_len = 100
        self.measure_step = 10

    def run_termalization(self, n_max):
        self.obs_converge.reset()

        for _ in trange(n_max // self.measure_step, desc="Termalization", leave=False):
            self._solve_and_calculate(
                range(self.measure_step), self.obs_converge)
            if self.obs_converge.has_converged():
                break
        else:
            print("\nSystem didn't converge. Continuing.")

    def run_measurements(self, n_max):
        self.obs.reset()
        iter = trange(n_max, desc="Measurements", leave=False)
        self._solve_and_calculate(iter, self.obs)
        return self.obs.get_result()

    def _solve_and_calculate(self, iter, obs_list):
        for _ in iter:
            self.solver(self.FK, self.mc_step)
            obs_list.calculate()
