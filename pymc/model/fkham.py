import numpy as np
import matplotlib.pyplot as plt
from .lattice import Lattice


class Hamiltonian(Lattice):

    def __init__(self, n, lattice_type, is_periodic=True, t=-1, U=0):

        super(Hamiltonian, self).__init__(n, lattice_type, is_periodic)
        self.t = t
        self.U = U
        self.H = np.copy(self.adj_matrix)
        self.H *= self.t

    def put_adatoms(self, nad, order="random"):
        assert (self.U != 0), "U is equal to zero, adatoms can not be put"
        assert (nad < self.n**2), "too many adatoms"
        assert (order in ["random", "sublattice", "separation"])

        self.nad = nad
        self.nad_norm = nad / self.n**2
        self.nad_to_put = nad

        for i in range(self.n**2):
            self.H[i, i] = 0.0

        if order == "random":
            self.put_adatoms_order_random()
        elif order == "sublattice":
            self.put_adatoms_order_sublattice()
        elif order == "separation":
            self.put_adatoms_order_separation()

        self.filled_sites = set([i for i in range(self.n**2) if self.H[i, i] != 0.0])
        self.empty_sites = set(range(self.n**2)).difference(set(self.filled_sites))

    def put_adatoms_order_random(self):
        index = np.random.choice(self.n**2, self.nad_to_put, replace=False)
        for i in index:
            self.H[i, i] = self.U

    def put_adatoms_order_separation(self):
        i = 0
        while i < self.nad_to_put:
            x_index = i % self.n
            y_index = i // self.n
            k = x_index * self.n + y_index
            self.H[k, k] = self.U
            i += 1

    def put_adatoms_order_sublattice(self):

        if self.nad_to_put >= self.n**2 // 2:
            for i in self.sub_matrix[0]:
                self.H[i, i] = self.U
            self.nad_to_put -= self.n**2 // 2

        index = np.random.choice(self.sub_matrix[1],
                                 self.nad_to_put, replace=False)
        for i in index:
            self.H[i, i] = self.U

    def plot_system(self, show=True):
        self.plot(False)

        X = []
        Y = []
        for i in range(self.n**2):
            if self.H[i, i] > 0:
                X.append(self.pos_matrix[i, 0])
                Y.append(self.pos_matrix[i, 1])

        plt.scatter(X, Y, color="red", zorder=3, s=100)

        if show:
            plt.show()

    def get_F(self, T, cp):
        eigv = np.linalg.eigvalsh(self.H)
        E = -T * sum(np.log(np.exp((cp - eigv)/T) + 1))
        return E
