import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../pymc')))
import lattice
import model
import observables

class TestObsCV():

    def prepare_model_E(self, n, nad, order):
        l1 = lattice.GrapheneLattice(n)
        FK = model.Hamiltonian(lattice=l1, t=-1, U=2, T=0.01)
        o_energy = observables.EnergyObs(FK)
        for _ in range(50):
            FK.put_adatoms(nad, order)
            FK.calculate_eigv()
            o_energy.calculate()
        return o_energy.get_result()

    def prepare_model_cv(self, n, nad, order, T):
        l1 = lattice.GrapheneLattice(n)
        FK = model.Hamiltonian(lattice=l1, t=-1, U=2, T=T)
        o_cv = observables.CVObs(FK)
        for _ in range(50):
            FK.put_adatoms(nad, order)
            FK.calculate_eigv()
            o_cv.calculate()
        return o_cv.get_result()
    
    def test_obs_energy_1(self):
        E1 = self.prepare_model_E(10, 50, "sublattice")
        E2 = self.prepare_model_E(10, 50, "random")
        assert E1 < E2

    def test_obs_cv_1(self):
        cv1 = self.prepare_model_cv(10, 50, "sublattice", T=0.1)
        cv2 = self.prepare_model_cv(10, 50, "random", T=0.1)
        assert cv1 < cv2
