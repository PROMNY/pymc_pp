# import numpy as np
# import matplotlib.pyplot as plt
# import itertools


class Lattice:
    from .lattice_plot_methods import plot_hopping, plot
    from .lattice_fill_methods import (fill_sub_matrix, fill_adj_matrix_square,
                                       fill_pos_matrix_graphene,
                                       fill_pos_matrix_square,
                                       fill_adj_matrix_graphene)

    def __init__(self, n, lattice_type, is_periodic=True,):
        """Initialize new lattice with given parameters:
        n - linear lattice size
        lattice_type - sqrare or graphene
        is_periodic=True pbc"""

        assert (lattice_type in ["square", "graphene"]), "Unknown lattice type"
        assert (n % 2 == 0), "n must be even"

        self.n = n
        self.lattice_type = lattice_type
        self.is_periodic = is_periodic

        if self.lattice_type == "graphene":
            assert (n >= 4), "for graphen lattice n grater or equal 4"
            self.fill_adj_matrix_graphene()
            self.fill_sub_matrix()
            self.fill_pos_matrix_graphene()

        elif self.lattice_type == "square":
            self.fill_adj_matrix_square()
            self.fill_sub_matrix()
            self.fill_pos_matrix_square()
