import numpy as np
from .fkham import Hamiltonian
from .obs import Obs
from .opt import Opt


class System(Hamiltonian, Obs, Opt):
    "Class of possible raunners"

    def __init__(self, n, lattice_type, is_periodic=True, t=-1, U=0):
        assert (lattice_type in ["square", "graphene"]), "Unknown lattice type"
        assert (n % 2 == 0), "n must be even"
        assert (U >= 0), "U is equal to zero"

        Obs.__init__(self)
        Hamiltonian.__init__(self, n, lattice_type, is_periodic, t, U)

        self.energy = None
