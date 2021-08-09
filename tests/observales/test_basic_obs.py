import numpy as np
import pymc


class TestBasicObs():

    def test_converge_1(self):
        obs = pymc.BasicObs(None)
        obs.value_list = np.ones(200)
        assert obs.has_converged()

    def test_converge_2(self):
        obs = pymc.BasicObs(None)
        obs.value_list = np.ones(80)
        assert not obs.has_converged()

    def test_converge_3(self):
        obs = pymc.BasicObs(None)
        obs.value_list = np.concatenate((np.ones(100), np.zeros(100)))
        assert not obs.has_converged(200)
