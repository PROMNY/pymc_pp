import pymc
import numpy as np


class TestModelEnergy():

    def test_hamiltonian_hopping(self):
        l1 = pymc.GrapheneLattice(4)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        res = -1 * np.asarray([
            [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.]])

        np.testing.assert_array_equal(FK.H, res, verbose=False)
        np.testing.assert_array_equal(FK.temp_H, res, verbose=False)

    def test_hamiltonian_eigv(self):
        l1 = pymc.GrapheneLattice(10)
        FK = pymc.Hamiltonian(lattice=l1, t=-1)
        FK.calculate_eigv()
        assert len(FK.eigv) == 100
        np.testing.assert_almost_equal(np.amax(FK.eigv), 3.0)
        np.testing.assert_almost_equal(np.amin(FK.eigv), -3.0)
        np.testing.assert_almost_equal(np.sum(FK.eigv), 0)

    def test_hamiltonian_energy_1(self):
        l1 = pymc.GrapheneLattice(10)
        FK1 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK1.put_adatoms(50, "sublattice")
        FK1.calculate_eigv()
        E1 = FK1.get_F(T=1, cp=1)

        FK2 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK2.put_adatoms(50, "random")
        FK2.calculate_eigv()
        E2 = FK2.get_F(T=1, cp=1)
        assert E1 < E2

    def test_hamiltonian_energy_2(self):
        l1 = pymc.GrapheneLattice(10)
        FK1 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK1.calculate_eigv()
        E1 = FK1.get_F(T=1, cp=1)

        FK2 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK2.T = 1
        FK2.cp = 1
        FK2.calculate_eigv()
        E2 = FK2.get_F()
        np.testing.assert_almost_equal(E1, E2)

    def test_hamiltonian_energy_3(self):
        l1 = pymc.GrapheneLattice(10)
        FK1 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK1.calculate_eigv()
        E1 = FK1.get_F(T=1, cp=1)

        FK2 = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK2.calculate_eigv(temp=True)
        E2 = FK2.get_F(temp=True, T=1, cp=1)
        np.testing.assert_almost_equal(E1, E2)
