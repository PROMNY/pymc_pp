import numpy as np


def fill_adj_matrix_square(self):
    n = self.n
    self.adj_matrix = np.zeros((n**2, n**2))

    for i in range(n**2 - n):
        self.adj_matrix[i][i+n] = 1
        self.adj_matrix[i+n][i] = 1

    for i in range(n**2):
        if i % n > 0:
            self.adj_matrix[i][i-1] = 1
            self.adj_matrix[i-1][i] = 1

    if self.is_periodic:
        for i in range(n):
            self.adj_matrix[i*n][i*n + n - 1] = 1
            self.adj_matrix[i*n + n - 1][i*n] = 1
            self.adj_matrix[i][-n+i] = 1
            self.adj_matrix[-n+i][i] = 1


def fill_adj_matrix_graphene(self):
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

    if self.is_periodic:
        for i in range(1, n-1):
            self.adj_matrix[n*i][n*i + n - 1] = 1
            self.adj_matrix[n*i + n - 1][n*i] = 1

        for i in range(n):
            self.adj_matrix[i][-n+i] = 1
            self.adj_matrix[-n+i][i] = 1


def fill_pos_matrix_square(self):
    n = self.n
    self.pos_matrix = []

    for i in range(n**2):
        x = i % n
        y = n - (i // n) - 1
        self.pos_matrix.append([x, y])

    self.pos_matrix = np.asarray(self.pos_matrix)
    self.sub_matrix = np.asarray(self.sub_matrix)


def fill_pos_matrix_graphene(self):
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


def fill_sub_matrix(self):
    n = self.n
    self.sub_matrix = [[], []]

    for i in range(n**2):
        if (i % n + i//n) % 2 == 0:
            self.sub_matrix[0].append(i)
        else:
            self.sub_matrix[1].append(i)

    self.sub_matrix = np.asarray(self.sub_matrix)
