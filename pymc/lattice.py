import numpy as np
import matplotlib.pyplot as plt


class lattice:

    def __init__(self, n, lattice_type, is_periodic=True,):
        """Initialize new lattice with given parameters:
        n - linear lattice size
        lattice_type - sqrare or graphene
        is_periodic=True pbc"""

        assert (lattice_type in ["square", "graphene"]), "Unknown lattice type"
        assert (n % 2 == 0), "n must be even"

        self.n = n
        self.lattice_type = lattice_type
        self.is_periodic = is_periodic

        if self.lattice_type == "graphene":
            assert (n >= 4), "for graphen lattice n grater or equal 4"
            self.fill_adj_matrix_graphene()
            self.fill_pos_matrix_graphene()

        elif self.lattice_type == "square":
            self.fill_adj_matrix_square()
            self.fill_pos_matrix_square()

    def fill_adj_matrix_square(self):
        n = self.n
        self.adj_matrix = np.zeros((n**2, n**2))

        for i in range(n**2 - n):
            self.adj_matrix[i][i+n] = 1
            self.adj_matrix[i+n][i] = 1

        if self.is_periodic:
            for i in range(n):
                self.adj_matrix[i][-n+i] = 1
                self.adj_matrix[-n+i][i] = 1

        for i in range(n**2):
            if i % n > 0:
                self.adj_matrix[i][i-1] = 1
                self.adj_matrix[i-1][i] = 1

        if self.is_periodic:
            for i in range(n):
                self.adj_matrix[i*n][i*n + n - 1] = 1
                self.adj_matrix[i*n + n - 1][i*n] = 1

    def fill_adj_matrix_graphene(self):
        n = self.n
        self.adj_matrix = np.zeros((n**2, n**2))

        for i in range(n**2 - n):
            self.adj_matrix[i][i+n] = 1
            self.adj_matrix[i+n][i] = 1

        if self.is_periodic:
            for i in range(n):
                self.adj_matrix[i][-n+i] = 1
                self.adj_matrix[-n+i][i] = 1

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

    def fill_pos_matrix_square(self):
        n = self.n
        self.pos_matrix = []
        self.sub_matrix = [[], []]

        for i in range(n**2):
            x = i % n
            y = n - (i // n) - 1
            self.pos_matrix.append([x, y])

            if i % (2 * n) < n:
                if i % 2 == 0:
                    self.sub_matrix[0].append(i)
                else:
                    self.sub_matrix[1].append(i)
            else:
                if i % 2 == 0:
                    self.sub_matrix[1].append(i)
                else:
                    self.sub_matrix[0].append(i)

        self.pos_matrix = np.asarray(self.pos_matrix)
        self.sub_matrix = np.asarray(self.sub_matrix)

    def fill_pos_matrix_graphene(self):
        n = self.n
        self.pos_matrix = []
        self.sub_matrix = [[], []]
        for i in range(n**2):
            j = i % n
            if i % (2 * n) < n:
                if i % 2 == 0:
                    x = 0.5 + 1.5 * j
                    self.sub_matrix[0].append(i)
                else:
                    x = 1.5 * j
                    self.sub_matrix[1].append(i)
            else:
                if i % 2 == 0:
                    x = 1.5 * j
                    self.sub_matrix[1].append(i)
                else:
                    x = 0.5 + 1.5 * j
                    self.sub_matrix[0].append(i)
            y = np.sqrt(3)/2 * (n - (i // n) - 1)
            self.pos_matrix.append([x, y])

        self.pos_matrix = np.asarray(self.pos_matrix)
        self.sub_matrix = np.asarray(self.sub_matrix)

    def plot(self):
        if self.lattice_type == "graphene":
            plt.figure(figsize=(10, 6))
        else:
            plt.figure(figsize=(10, 10))
        n = self.n

        X = [[], []]
        Y = [[], []]
        for sub in range(2):
            for i in self.sub_matrix[sub]:
                X[sub].append(self.pos_matrix[i][0])
                Y[sub].append(self.pos_matrix[i][1])

        # X = [pos[0] for pos in self.pos_matrix]
        # Y = [pos[1] for pos in self.pos_matrix]
        plt.scatter(X[0], Y[0], color="blue")
        plt.scatter(X[1], Y[1], color="red")

        for i in range(n**2):
            for j in range(i, n**2):
                if self.adj_matrix[i][j] == 1:
                    v = self.pos_matrix[i] - self.pos_matrix[j]
                    len = np.sqrt(v[0]**2 + v[1]**2)

                    x = [self.pos_matrix[i][0],
                         self.pos_matrix[j][0]]
                    y = [self.pos_matrix[i][1],
                         self.pos_matrix[j][1]]

                    if len < 1.1:
                        plt.plot(x, y, color="black")
                    else:
                        plt.plot(x, y, color="red", ls="--")

        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-0.1, np.amax(X) + 0.1)
        plt.ylim(-0.1, np.amax(Y) + 0.1)
        plt.show()
