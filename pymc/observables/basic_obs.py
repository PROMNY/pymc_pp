import numpy as np


class BasicObs():

    def __init__(self, model):
        self.value_list = []
        self.model = model

    def reset(self):
        self.value_list = []

    def has_converged(self, min_len=100):
        n = len(self.value_list)
        if n < min_len:
            return False

        k = min_len // 2
        l1_avg = np.average(self.value_list[-k:])
        l2_avg = np.average(self.value_list[-2*k:-k])
        print(l1_avg, l2_avg)
        if abs(l1_avg - l2_avg) > 0.01 * abs(l1_avg):
            return False

        l1_std = np.std(self.value_list[-k:])
        l2_std = np.std(self.value_list[-2*k:-k])

        return l1_std < l2_std

    def get_result(self):
        return np.average(self.value_list)
