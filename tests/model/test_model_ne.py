import numpy as np
import pymc


class TestModelNe():

    def test_hamiltonian_Ne_1(self):
        l1 = pymc.GrapheneLattice(10)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK.calculate_eigv()
        Ne = FK.get_ne(T=0.01, cp=0)
        np.testing.assert_almost_equal(Ne, 50)

    def test_hamiltonian_Ne_2(self):
        l1 = pymc.GrapheneLattice(10)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK.put_adatoms(50, "sublattice")
        FK.calculate_eigv()
        Ne = FK.get_ne(T=0.01, cp=1)
        np.testing.assert_almost_equal(Ne, 50)

    def test_hamiltonian_Ne_3(self):
        l1 = pymc.GrapheneLattice(10)
        FK1 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK1.calculate_eigv(temp=True)
        Ne1 = FK1.get_ne(temp=True, T=0.01, cp=-1)

        FK2 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK2.T = 0.01
        FK2.cp = 1
        FK2.calculate_eigv()
        Ne2 = FK2.get_ne()

        assert Ne1 < Ne2
