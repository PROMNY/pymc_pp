import numpy as np


class Hamiltonian:

    def __init__(self, lattice, U=0, t=-1.0):
        self.t = t
        self.U = U
        self.H = np.copy(lattice.adj_matrix)
        self.H *= self.t
        self.n = lattice.n
        self.lattice = lattice

    def put_adatoms(self, nad, order="random"):
        assert (self.U > 0), "U is equal to zero"
        assert (nad < self.n**2), "too many adatoms"
        assert (order in ["random", "sublattice", "separation"])

        for i in range(self.n**2):
            self.H[i][i] = 0.0

        if order == "random":
            for _ in range(nad):
                while True:
                    i = np.floor(np.random.rand()*self.n**2)
                    if self.H[i][i] < self.U:
                        self.H[i][i] = self.U
                        break

        elif order == "sublattice":
            if nad >= self.n**2 // 2:
                for i in self.lattice.sub_matrix[0]:
                    self.H[i][i] = self.U
                nad -= self.n**2 // 2
            for _ in range(nad):
                while True:
                    i = np.floor(np.random.rand()*(self.n**2 // 2))
                    j = self.lattice.sub_matrix[1][i]
                    if self.H[j][j] < self.U:
                        self.H[j][j] = self.U
                        break
