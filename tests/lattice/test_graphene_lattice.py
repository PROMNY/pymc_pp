import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../pymc')))
import lattice

class GrapheneLatticeTests(unittest.TestCase):
    """Basic test cases."""

    def test_adj_matrix_graphene(self):
        a = lattice.GrapheneLattice(4)
        res = np.asarray([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

        np.testing.assert_array_equal(a.adj_matrix, res, verbose=False)

    def test_pos_matrix_graphene(self):
        a = lattice.GrapheneLattice(4)
        res = np.asarray([[0.5,        2.59807621],
                          [1.5,        2.59807621],
                          [3.5,       2.59807621],
                          [4.5,        2.59807621],
                          [0.,       1.73205081],
                          [2.,       1.73205081],
                          [3.,       1.73205081],
                          [5.,       1.73205081],
                          [0.5,       0.8660254],
                          [1.5,       0.8660254],
                          [3.5,       0.8660254],
                          [4.5,       0.8660254],
                          [0.,       0.],
                          [2.,       0.],
                          [3.,       0.],
                          [5.,       0.]])

        np.testing.assert_array_almost_equal(a.pos_matrix, res, verbose=False)

    def test_sub_matrix_graphene(self):
        a = lattice.GrapheneLattice(4)
        res = np.asarray([[0,  2,  5,  7,  8, 10, 13, 15],
                          [1,  3,  4,  6,  9, 11, 12, 14]])

        np.testing.assert_array_almost_equal(a.sub_matrix, res, verbose=False)
