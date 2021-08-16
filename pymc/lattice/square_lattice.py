import numpy as np
import itertools
from .common_lattice import fill_sub_matrix


class SquareLattice():
    """Initialize a square lattice with given parameters:
        n - linear lattice size
        is_periodic=True pbc
        Attributes: 
        self.pos_matrix[i] return x and y coordinates of i-th site
        self.adj_matrix[i][j] is 1 if sites are connected, 0 if not
        self.sub_matrix[i] i in [0,1] returns list of atoms in given sublattice  
        """

    def __init__(self, n: int, periodic: bool = True) -> None:
        assert (n % 2 == 0 and n > 0), "n must be even and positive"

        self.n = n
        self.lattice_type = "square"
        self.periodic = periodic

        self.__fill_adj_matrix()
        self.sub_matrix = fill_sub_matrix(self.n)
        self.__fill_pos_matrix()

    def __fill_adj_matrix(self) -> None:
        n = self.n
        self.adj_matrix = np.zeros((n**2, n**2))

        for i in range(n**2 - n):
            self.adj_matrix[i][i+n] = 1
            self.adj_matrix[i+n][i] = 1

        for i in range(n**2):
            if i % n > 0:
                self.adj_matrix[i][i-1] = 1
                self.adj_matrix[i-1][i] = 1

        if self.periodic:
            for i in range(n):
                self.adj_matrix[i*n][i*n + n - 1] = 1
                self.adj_matrix[i*n + n - 1][i*n] = 1
                self.adj_matrix[i][-n+i] = 1
                self.adj_matrix[-n+i][i] = 1

        if self.periodic:
            for i in range(1, n-1):
                self.adj_matrix[n*i][n*i + n - 1] = 1
                self.adj_matrix[n*i + n - 1][n*i] = 1

            for i in range(n):
                self.adj_matrix[i][-n+i] = 1
                self.adj_matrix[-n+i][i] = 1

    def __fill_pos_matrix(self) -> None:
        n = self.n
        pos_matrix = []

        for i in range(n**2):
            x = i % n
            y = n - (i // n) - 1
            pos_matrix.append([x, y])

        self.pos_matrix = np.asarray(pos_matrix)
