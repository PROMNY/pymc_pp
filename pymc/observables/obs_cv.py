import numpy as np
from .basic_obs import BasicObs


class CVObs(BasicObs):

    def __init__(self, model):
        self.model = model
        self.value_list = [[], []]

    def calculate(self, add_result=True):
        E = self.model.get_F()

        if add_result:
            self.value_list[0].append(E)
            self.value_list[1].append(E**2)
        return (E, E**2)

    def get_result(self):
        cv = np.average(self.value_list[1]) - np.average(self.value_list[0])**2
        cv /= (self.model.n**2 * self.model.T**2)
        return cv

    def reset(self):
        self.value_list = [[], []]
