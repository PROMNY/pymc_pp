import numpy as np
from .basic_obs import BasicObs


class EnergyObs(BasicObs):

    def calculate(self, add_result=True):
        E = self.model.get_F()
        if add_result:
            self.value_list.append(E)
        return E
