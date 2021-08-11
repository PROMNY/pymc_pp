import pymc
import numpy as np
import pytest


class TestHalfFilling1():

    @pytest.fixture(scope='class')
    def get_results(self):
        lattice = pymc.GrapheneLattice(6)
        FK = pymc.Hamiltonian(lattice, t=-1, U=2, cp=1, T=0.2)
        FK.put_adatoms(18, "random")

        obs = pymc.ObsList([pymc.DeltaObs(FK),
                            pymc.EnergyObs(FK), pymc.CVObs(FK),
                            pymc.CorrelationObs(FK), pymc.NeObs(FK)])
        obs_conv = pymc.ObsList(
            [pymc.DeltaObs(FK), pymc.EnergyObs(FK), pymc.CorrelationObs(FK)])

        series = pymc.ObsSeries(obs, ["T"])
        sym = pymc.Simulator(FK, pymc.metropolis_numpy, obs, obs_conv)

        T_range = [0.2, 0.18, 0.16, 0.14, 0.12,
                   0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]

        for T in T_range:
            FK.T = T
            sym.run_termalization(10**2)
            res = sym.run_measurements(10**2)
            series.add(res, [T])

        expected = np.loadtxt(
            "tests/simulate/half_filling_1.csv", delimiter=",")

        res = series.get_df().values
        return (expected, res)

    @pytest.mark.long
    def test_half_filling_1(self, get_results):
        # tempertature
        expected, res = get_results
        np.testing.assert_array_equal(res[:, 0], expected[:, 0])

    @pytest.mark.long
    def test_half_filling_2(self, get_results):
        # delta
        expected, res = get_results
        np.testing.assert_array_almost_equal(
            res[:, 1], expected[:, 1], verbose=True, decimal=1)

    @pytest.mark.long
    def test_half_filling_3(self, get_results):
        # free energy
        expected, res = get_results
        np.testing.assert_array_almost_equal(
            res[:, 2], expected[:, 2], verbose=True, decimal=0)

    @pytest.mark.long
    def test_half_filling_4(self, get_results):
        # C1
        expected, res = get_results
        np.testing.assert_array_almost_equal(
            res[:, 4], expected[:, 4], verbose=True, decimal=1)

    @pytest.mark.long
    def test_half_filling_5(self, get_results):
        # Ne
        expected, res = get_results
        np.testing.assert_array_almost_equal(
            res[:, 5], expected[:, 5], verbose=True, decimal=1)
