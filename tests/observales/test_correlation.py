import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../pymc')))
import lattice
import model
import observables

class TestObsCorrelation():

    def prepare_model(self, n, nad, order):
        l1 = lattice.GrapheneLattice(n)
        FK = model.Hamiltonian(lattice=l1, t=-1, U=2)
        o_C = observables.CorrelationObs(FK)
        for _ in range(100):
            FK.put_adatoms(nad, order)
            o_C.calculate()
        return o_C.get_result()
    
    def test_obs_correlation_1(self):
        d = self.prepare_model(10, 50, "sublattice")
        np.testing.assert_almost_equal(d, 1.0)

    def test_obs_correlation_2(self):
        d = self.prepare_model(10, 90, "sublattice")
        np.testing.assert_almost_equal(d, 1.0)

    def test_obs_correlation_3(self):
        d = self.prepare_model(10, 50, "separation")
        assert d < -0.5

    def test_obs_correlation_4(self):
        d = self.prepare_model(10, 10, "separation")
        assert d < -0.5

    def test_obs_correlation_5(self):
        d = self.prepare_model(10, 90, "separation")
        assert d < -0.5

    def test_obs_correlation_6(self):
        l1 = lattice.GrapheneLattice(10)
        FK = model.Hamiltonian(lattice=l1, t=-1, U=2)
        o_C = observables.CorrelationObs(FK)
        for _ in range(100):
            FK.put_adatoms(50, "random")
            o_C.calculate()
        d = o_C.get_result()

        assert len(o_C.value_list) == 100
        np.testing.assert_almost_equal(abs(d), 0, decimal=1)
