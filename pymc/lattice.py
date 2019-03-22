import numpy as np
import matplotlib.pyplot as plt


class lattice:

    def __init__(self, n, lattice_type, is_periodic=True,):
        """Initialize new lattice with given parameters:
        n - linear lattice size
        lattice_type - sqrare or graphene
        is_periodic=True pbc"""
        self.n = n
        self.lattice_type = lattice_type
        self.is_periodic = is_periodic
        assert (lattice_type in ["square", "graphene"]), "Unknown lattice type"

        self.fill_adj_matrix()

        self.fill_position_matrix()

    def fill_adj_matrix(self):
        n = self.n
        self.adj_matrix = np.zeros((n**2, n**2))

        for i in range(n**2 - n):
            self.adj_matrix[i][i+n] = 1
            self.adj_matrix[i+n][i] = 1

        if self.is_periodic:
            for i in range(n):
                self.adj_matrix[i][-n+i] = 1
                self.adj_matrix[-n+i][i] = 1

        if self.lattice_type == "square":
            for i in range(n**2):
                if i % n > 0:
                    self.adj_matrix[i][i-1] = 1
                    self.adj_matrix[i-1][i] = 1

            if self.is_periodic:
                for i in range(n):
                    self.adj_matrix[i*n][i*n + n - 1] = 1
                    self.adj_matrix[i*n + n - 1][i*n] = 1

        else:
            assert (n % 2 == 0), "for graphen lattice n must be even"
            assert (n >= 4), "for graphen lattice n grater or equal 4"

            R = [2*i for i in range(n//2)]
            R += [2*i+1+n for i in range(n//2)][:-1]

            for i in range(n**2):
                j = i % (2*n)
                if j in R:
                    self.adj_matrix[i][i+1] = 1
                    self.adj_matrix[i+1][i] = 1

            if self.is_periodic:
                for i in range(1, n-1):
                    self.adj_matrix[n*i][n*i + n - 1] = 1
                    self.adj_matrix[n*i + n - 1][n*i] = 1

    def fill_position_matrix(self):
        n = self.n
        self.position_matrix = []

        if self.lattice_type == "square":
            for i in range(n**2):
                x = i % n
                y = n - (i // n) - 1
                self.position_matrix.append([x, y])
        else:
            for i in range(n**2):
                j = i % n
                if i % (2 * n) < n:
                    if i % 2 == 0:
                        x = 0.5 + 1.5 * j
                    else:
                        x = 1.5 * j
                else:
                    if i % 2 == 0:
                        x = 1.5 * j
                    else:
                        x = 0.5 + 1.5 * j
                y = np.sqrt(3)/2 * (n - (i // n) - 1)
                self.position_matrix.append([x, y])

        self.position_matrix = np.asarray(self.position_matrix)

    def plot(self):
        plt.clf()
        n = self.n
        X = [pos[0] for pos in self.position_matrix]
        Y = [pos[1] for pos in self.position_matrix]
        plt.scatter(X, Y)

        for i in range(n**2):
            for j in range(i, n**2):
                if self.adj_matrix[i][j] == 1:
                    v = self.position_matrix[i] - self.position_matrix[j]
                    len = np.sqrt(v[0]**2 + v[1]**2)

                    x = [self.position_matrix[i][0],
                         self.position_matrix[j][0]]
                    y = [self.position_matrix[i][1],
                         self.position_matrix[j][1]]

                    if len < 1.1:
                        plt.plot(x, y, color="black")
                    else:
                        plt.plot(x, y, color="red")

        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-0.1, np.amax(X) + 0.1)
        plt.ylim(-0.1, np.amax(Y) + 0.1)
        plt.show()
