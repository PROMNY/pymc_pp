import numpy as np
import matplotlib.pyplot as plt
from adatoms import AdatomsMixin


class Hamiltonian(AdatomsMixin):

    def __init__(self, lattice, t=-1, U=0, cp=0, T=1):

        self.lattice = lattice
        self.n = self.lattice
        self.t = t
        self.U = U
        self.H = np.copy(self.lattice.adj_matrix)
        self.H *= self.t
        self.cp = cp
        self.T = T
        self.calculate_eigv()

    def calculate_eigv(self):
        self.eigv = np.linalg.eigvalsh(self.H)

    def get_F(self, T=None, cp=None):
        if T is None:
            T = self.T
        if cp is None:
            cp = self.cp
        F = -1 * T * sum(np.log(np.exp((cp - self.eigv)/T) + 1))
        return F

    def get_ne(self, T=None, cp=None):
        pass
