import numpy as np


def fill_sub_matrix(n):
    sub_matrix = [[], []]

    for i in range(n**2):
        if (i % n + i//n) % 2 == 0:
            sub_matrix[0].append(i)
        else:
            sub_matrix[1].append(i)

    return np.asarray(sub_matrix)
