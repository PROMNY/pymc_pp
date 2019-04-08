import numpy as np
import matplotlib.pyplot as plt


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
            self.H[i, i] = 0.0

        if order == "random":
            self.put_adatoms_order_random(nad)
        elif order == "sublattice":
            self.put_adatoms_order_sublattice(nad)
        elif order == "separation":
            self.put_adatoms_order_separation(nad)

    def put_adatoms_order_random(self, nad):
        index = np.random.choice(self.n**2, nad, replace=False)
        for i in index:
            self.H[i, i] = self.U

    def put_adatoms_order_separation(self, nad):
        i = 0
        while i < nad:
            x_index = i % self.n
            y_index = i // self.n
            k = x_index * self.n + y_index
            self.H[k, k] = self.U
            i += 1

    def put_adatoms_order_sublattice(self, nad):

        if nad >= self.n**2 // 2:
            for i in self.lattice.sub_matrix[0]:
                self.H[i, i] = self.U
            nad -= self.n**2 // 2

        index = np.random.choice(self.lattice.sub_matrix[1],
                                 nad, replace=False)
        for i in index:
            self.H[i, i] = self.U

    def plot(self, show=True):
        self.lattice.plot(False)

        X = []
        Y = []
        for i in range(self.n**2):
            if self.H[i, i] > 0:
                X.append(self.lattice.pos_matrix[i, 0])
                Y.append(self.lattice.pos_matrix[i, 1])

        plt.scatter(X, Y, color="red", zorder=3, s=100)

        if show:
            plt.show()
