import numpy as np
import matplotlib.pyplot as plt
from .adatoms import AdatomsMixin


class Hamiltonian(AdatomsMixin):

    def __init__(self, lattice, t=-1, U=0, cp=0, T=1):

        self.lattice = lattice
        self.n = self.lattice.n
        self.t = t
        self.U = U
        self.H = np.copy(self.lattice.adj_matrix)
        self.H *= self.t
        self.temp_H = np.copy(self.H)
        self.cp = cp
        self.T = T
        self.eigv = None
        self.temp_eigv = None

    def calculate_eigv(self, temp=False):
        if temp:
            self.temp_eigv = np.linalg.eigvalsh(self.temp_H)
        else:
            self.eigv = np.linalg.eigvalsh(self.H)

    def get_F(self, temp=False, T=None, cp=None):
        if T is None:
            T = self.T
        if cp is None:
            cp = self.cp
        if temp:
            eigv = self.temp_eigv
        else:
            eigv = self.eigv

        F = -1 * T * sum(np.log(np.exp((cp - eigv)/T) + 1))
        return F

    def get_ne(self, temp=False, T=None, cp=None):
        if T is None:
            T = self.T
        if cp is None:
            cp = self.cp
        if temp:
            eigv = self.temp_eigv
        else:
            eigv = self.eigv
        Ne = sum(1.0 / (np.exp((eigv - cp)/T) + 1))
        return Ne
