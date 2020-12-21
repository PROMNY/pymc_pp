import numpy as np
from itertools import product


class Obs(object):

    def __init__(self):
        self.delta_list = []
        self.E_list = []
        self.E2_list = []
        self.c1_list = []

    def calculate_delta(self, add_to_list=True):
        d = len(set(self.sub_matrix[0]).intersection(self.filled_sites))
        d = abs(2*d - self.nad)/self.nad
        if add_to_list:
            self.delta_list.append(d)
        return d

    def calculate_c1(self, add_to_list=True):
        sites_A = list(self.filled_sites.intersection(set(self.sub_matrix[0])))
        sites_B = list(self.filled_sites.intersection(set(self.sub_matrix[1])))
        pairs = product(sites_A, sites_B)
        c = len(list(filter((lambda x: self.adj_matrix[x[0], x[1]]), pairs)))
        c = (self.nad_norm**2 - 2*c/(self.n**2*3)) / self.nad_norm**2

        if add_to_list:
            self.c1_list.append(c)
        return c

    def calculate_E(self, T, cp, add_to_list=True):
        eigv = np.linalg.eigvalsh(self.H)
        E = -T*sum(np.log(np.exp((cp-eigv)/T) + 1))

        if add_to_list:
            self.E_list.append(E)
            self.E2_list.append(E**2)
        return E

    def avrg_delta(self):
        return np.average(self.delta_list)

    def avrg_c1(self):
        return np.average(self.c1_list)

    def avrg_cv(self, T):
        cw = np.average(self.E2_list)-np.average(self.E_list)**2
        cw /= self.n**2 * T**2
        return cw

    def reset_obs(self):
        self.delta_list = []
        self.E_list = []
        self.E2_list = []
        self.c1_list = []
