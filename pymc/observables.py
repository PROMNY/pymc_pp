import numpy as np
from itertools import product


class BasicObs():

    def __init__(self, model):
        self.value_list = []
        self.model = model

    def reset(self):
        self.value_list = []

    def has_converged(self, min_len=100):
        if len(self.value_list) < min_len:
            return False

        k = min_len // 2
        l1_avg = np.average(self.value_list[-k:])
        l2_avg = np.average(self.value_list[-2*k:-k])

        if abs(l1_avg - l2_avg) > 0.01 * abs(l1_avg):
            return False

        l1_std = np.std(self.value_list[-k:])
        l2_std = np.std(self.value_list[-2*k:-k])

        return l1_std <= l2_std

    def get_result(self):
        return np.average(self.value_list)


class EnergyObs(BasicObs):

    def calculate(self, add_result=True):
        E = self.model.get_F()
        if add_result:
            self.value_list.append(E)
        return E


class CorrelationObs(BasicObs):

    def __init__(self, model):
        BasicObs.__init__(self, model)
        self.sublattice_A_set = set(self.model.lattice.sub_matrix[0])
        self.sublattice_B_set = set(self.model.lattice.sub_matrix[1])

    def calculate(self, add_result=True):
        filled_set = set(self.model.filled_sites)
        sites_A = list(self.sublattice_A_set.intersection(filled_set))
        sites_B = list(self.sublattice_B_set.intersection(filled_set))

        pairs = product(sites_A, sites_B)

        c1_filter = filter(
            (lambda x: self.model.lattice.adj_matrix[x[0], x[1]]), pairs)
        c = len(list(c1_filter)) / (self.model.n**2 * 3)
        c = self.model.nad_norm**2 - 2*c

        if c < 0:
            c /= (self.model.nad_norm - self.model.nad_norm**2)
        else:
            if self.model.nad_norm <= 0.5:
                c /= self.model.nad_norm**2
            else:
                c /= (1 - self.model.nad_norm)**2

        if add_result:
            self.value_list.append(c)
        return c


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


class DeltaObs(BasicObs):

    def __init__(self, model):
        BasicObs.__init__(self, model)
        self.sublattice_A_set = set(self.model.lattice.sub_matrix[0])

    def calculate(self, add_result=True):

        filled_set = set(self.model.filled_sites)
        A = len(list(filled_set.intersection(self.sublattice_A_set)))
        d = abs(2*A - self.model.nad) / self.model.n**2

        if self.model.nad_norm <= 0.5:
            d /= self.model.nad_norm
        else:
            d /= (1 - self.model.nad_norm)

        if add_result:
            self.value_list.append(d)
        return d
