import numpy as np
import pymc


class TestAdatoms():

    def test_adatoms_sums(self):
        l1 = pymc.lattice.GrapheneLattice(4)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        for mode in ["random", "sublattice", "separation"]:
            FK.put_adatoms(6, order=mode)
            assert np.trace(FK.H) == 12
            assert len(FK.filled_sites) == 6
            assert len(FK.empty_sites) == 10

    def test_adatoms_sublatice(self):
        l1 = pymc.lattice.GrapheneLattice(4)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK.put_adatoms(6, order="sublattice")

        for i in FK.filled_sites:
            assert i in FK.lattice.sub_matrix[1]
            assert FK.H[i, i] == 2.0

    def test_swap_in_temp_H(self):
        l1 = pymc.lattice.GrapheneLattice(4)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK.put_adatoms(6)

        ch_empty = np.random.choice(FK.empty_sites)
        ch_filled = np.random.choice(FK.filled_sites)

        assert FK.H[ch_empty, ch_empty] == 0
        assert FK.H[ch_filled, ch_filled] == 2
        assert FK.temp_H[ch_empty, ch_empty] == 0
        assert FK.temp_H[ch_filled, ch_filled] == 2

        FK.swap_in_temp_H(ch_filled, ch_empty)

        assert FK.H[ch_empty, ch_empty] == 0
        assert FK.H[ch_filled, ch_filled] == 2
        assert FK.temp_H[ch_empty, ch_empty] == 2
        assert FK.temp_H[ch_filled, ch_filled] == 0

        FK.un_swap_in_temp_H(ch_filled, ch_empty)

        assert FK.H[ch_empty, ch_empty] == 0
        assert FK.H[ch_filled, ch_filled] == 2
        assert FK.temp_H[ch_empty, ch_empty] == 0
        assert FK.temp_H[ch_filled, ch_filled] == 2

    def test_swap_in_H(self):
        l1 = pymc.lattice.GrapheneLattice(4)
        FK = pymc.Hamiltonian(lattice=l1, t=-1, U=2)
        FK.put_adatoms(6)

        ch_empty = np.random.choice(FK.empty_sites)
        ch_filled = np.random.choice(FK.filled_sites)

        assert FK.H[ch_empty, ch_empty] == 0
        assert FK.H[ch_filled, ch_filled] == 2
        assert FK.temp_H[ch_empty, ch_empty] == 0
        assert FK.temp_H[ch_filled, ch_filled] == 2

        FK.swap_in_H(ch_filled, ch_empty)

        assert FK.H[ch_empty, ch_empty] == 2
        assert FK.H[ch_filled, ch_filled] == 0
        assert FK.temp_H[ch_empty, ch_empty] == 2
        assert FK.temp_H[ch_filled, ch_filled] == 0

        assert ch_empty in FK.filled_sites
        assert ch_empty not in FK.empty_sites
        assert ch_filled in FK.empty_sites
        assert ch_filled not in FK.filled_sites
