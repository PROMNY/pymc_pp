import numpy as np
import itertools
from .common_lattice import fill_sub_matrix


class GrapheneLattice():
    """Initialize a graphene lattice with given parameters:
        n - linear lattice size
        periodic=True pbc
        Attributes: 
        self.pos_matrix[i] return x and y coordinates of i-th site
        self.adj_matrix[i][j] is 1 if sites are connected, 0 if not
        self.sub_matrix[i], i in [0,1] returns list of atoms in given sublattice  
        """

    def __init__(self, n, periodic=True):
        assert (n % 2 == 0), "n must be even"
        assert (n >= 4), "for graphene lattice n grater or equal 4"

        self.n = n
        self.lattice_type = "graphene"
        self.periodic = periodic

        self.__fill_adj_matrix()
        self.sub_matrix = fill_sub_matrix(self.n)
        self.__fill_pos_matrix()

    def __fill_adj_matrix(self):
        n = self.n
        self.adj_matrix = np.zeros((n**2, n**2))

        for i in range(n**2 - n):
            self.adj_matrix[i][i+n] = 1
            self.adj_matrix[i+n][i] = 1

        R = [2*i for i in range(n//2)]
        R += [2*i+1+n for i in range(n//2)][:-1]

        for i in range(n**2):
            j = i % (2*n)
            if j in R:
                self.adj_matrix[i][i+1] = 1
                self.adj_matrix[i+1][i] = 1

        if self.periodic:
            for i in range(n):
                if i % 2 == 1:
                    self.adj_matrix[n*i][n*i + n - 1] = 1
                    self.adj_matrix[n*i + n - 1][n*i] = 1

            for i in range(n):
                self.adj_matrix[i][n*(n-1) + i] = 1
                self.adj_matrix[n*(n-1) + i][i] = 1

    def __fill_pos_matrix(self):
        n = self.n
        self.pos_matrix = []
        for i in range(n**2):
            j = i % n
            x = 1.5 * j
            if i in self.sub_matrix[0]:
                x += 0.5
            y = np.sqrt(3)/2 * (n - (i // n) - 1)
            self.pos_matrix.append([x, y])

        self.pos_matrix = np.asarray(self.pos_matrix)
