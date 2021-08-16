import numpy as np
import matplotlib.pyplot as plt
from .adatoms import AdatomsMixin


class Hamiltonian(AdatomsMixin):

    def __init__(self, lattice, t: float = -1, U: float = 0, cp: float = 0, T: float = 1) -> None:

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

    def calculate_eigv(self, temp: bool = False):
        if temp:
            self.temp_eigv = np.linalg.eigvalsh(self.temp_H)
        else:
            self.eigv = np.linalg.eigvalsh(self.H)

    def get_F(self, temp: bool = False, T: float = None, cp: float = None) -> float:
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

    def get_ne(self, temp: bool = False, T: float = None, cp: float = None) -> float:
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
