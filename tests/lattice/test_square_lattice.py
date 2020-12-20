import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../pymc')))
import lattice

class SquareLatticeTests(unittest.TestCase):
    """Basic test cases."""

    def test_adj_matrix_square(self):
        a = lattice.SquareLattice(4)
        res = np.asarray([
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]])

        np.testing.assert_array_almost_equal(a.adj_matrix, res, verbose=False)

    def test_pos_matrix_square(self):
        a = lattice.SquareLattice(4)
        res = np.asarray([[0, 3],
                          [1, 3],
                          [2, 3],
                          [3, 3],
                          [0, 2],
                          [1, 2],
                          [2, 2],
                          [3, 2],
                          [0, 1],
                          [1, 1],
                          [2, 1],
                          [3, 1],
                          [0, 0],
                          [1, 0],
                          [2, 0],
                          [3, 0]])

        np.testing.assert_array_almost_equal(a.pos_matrix, res, verbose=False)

    def test_sub_matrix_square(self):
        a = lattice.SquareLattice(4)
        res = np.asarray([[0,  2,  5,  7,  8, 10, 13, 15],
                          [1,  3,  4,  6,  9, 11, 12, 14]])

        np.testing.assert_array_almost_equal(a.sub_matrix, res, verbose=False)
