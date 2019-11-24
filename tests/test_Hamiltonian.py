from .context import pymc

import unittest
import numpy as np


class HamiltonianTests(unittest.TestCase):

    def test_h_matrix_sum_random(self):
        ham = pymc.Hamiltonian(4, "graphene", t=-1, U=1)
        ham.put_adatoms(6, "random")

        res = np.trace(ham.H)

        np.testing.assert_almost_equal(res, 6.0)

    def test_h_matrix_sum_sublattice(self):
        ham = pymc.Hamiltonian(4, "graphene", t=-1, U=1)
        ham.put_adatoms(6, "sublattice")

        res = np.trace(ham.H)

        np.testing.assert_almost_equal(res, 6.0)

    def test_h_matrix_sum_separation(self):
        ham = pymc.Hamiltonian(4, "graphene", t=-1, U=1)
        ham.put_adatoms(6, "separation")

        res = np.trace(ham.H)

        np.testing.assert_almost_equal(res, 6.0)

    def test_h_matrix_separation(self):
        ham = pymc.Hamiltonian(4, "graphene", t=-1, U=1)
        ham.put_adatoms(8, "separation")

        res = np.asarray([
            [1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [-1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0, -1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0]])

        np.testing.assert_array_almost_equal(ham.H, res, verbose=False)

    def test_h_matrix_sublattice(self):
        ham = pymc.Hamiltonian(4, "graphene", t=-1, U=1)
        ham.put_adatoms(8, "sublattice")

        res = np.asarray([
            [1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, ],
            [-1,  0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, ],
            [0, 0,  1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, ],
            [0, 0, -1,  0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
            [-1, 0, 0, 0,  0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, ],
            [0, -1, 0, 0, 0,  1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, ],
            [0, 0, -1, 0, 0, -1,  0, 0, 0, 0, -1, 0, 0, 0, 0, 0, ],
            [0, 0, 0, -1, -1, 0, 0,  1, 0, 0, 0, -1, 0, 0, 0, 0, ],
            [0, 0, 0, 0, -1, 0, 0, 0,  1, -1, 0, -1, -1, 0, 0, 0, ],
            [0, 0, 0, 0, 0, -1, 0, 0, -1,  0, 0, 0, 0, -1, 0, 0, ],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  1, -1, 0, 0, -1, 0, ],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1,  0, 0, 0, 0, -1, ],
            [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  0, 0, 0, 0, ],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  1, -1, 0, ],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1,  0, 0, ],
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,  1, ]])

        np.testing.assert_array_almost_equal(ham.H, res, verbose=False)
