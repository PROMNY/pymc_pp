import numpy as np
import pymc


class TestObsDelta():

    def prepare_model(self, n, nad, order):
        l1 = pymc.GrapheneLattice(n)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        o_delta = pymc.DeltaObs(FK)
        for _ in range(100):
            FK.put_adatoms(nad, order)
            o_delta.calculate()
        return o_delta.get_result()

    def test_obs_delta_1(self):
        d = self.prepare_model(10, 50, "sublattice")
        np.testing.assert_almost_equal(d, 1.0)

    def test_obs_delta_2(self):
        d = self.prepare_model(10, 10, "sublattice")
        np.testing.assert_almost_equal(d, 1.0)

    def test_obs_delta_3(self):
        d = self.prepare_model(10, 90, "sublattice")
        np.testing.assert_almost_equal(d, 1.0)

    def test_obs_delta_4(self):
        d = self.prepare_model(10, 50, "separation")
        np.testing.assert_almost_equal(d, 0)

    def test_obs_delta_5(self):
        d = self.prepare_model(10, 10, "separation")
        np.testing.assert_almost_equal(d, 0)

    def test_obs_delta_6(self):
        d = self.prepare_model(10, 90, "separation")
        np.testing.assert_almost_equal(d, 0)

    def test_obs_delta_7(self):
        l1 = pymc.GrapheneLattice(10)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        o_delta = pymc.DeltaObs(FK)
        for _ in range(100):
            FK.put_adatoms(50, "random")
            o_delta.calculate()
        d = o_delta.get_result()

        assert len(o_delta.value_list) == 100
        np.testing.assert_almost_equal(d, 0, decimal=1)
