import numpy as np


class BasicObs():

    def __init__(self, model):
        self.value_list = []
        self.model = model

    def has_converged(self):
        pass

    def get_result(self):
        return np.average(self.value_list)
