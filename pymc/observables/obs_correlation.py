from itertools import product
from .basic_obs import BasicObs


class CorrelationObs(BasicObs):

    def __init__(self, model):
        BasicObs.__init__(self, model)
        self.sublattice_A_set = set(self.model.lattice.sub_matrix[0])
        self.sublattice_B_set = set(self.model.lattice.sub_matrix[1])
 

    def calculate(self, add_result=True):
        filled_set = set(self.model.filled_sites)
        sites_A = list(self.sublattice_A_set.intersection(filled_set))
        sites_B = list(self.sublattice_B_set.intersection(filled_set))

        pairs = product(sites_A, sites_B)

        c1_filter = filter(
            (lambda x: self.model.lattice.adj_matrix[x[0], x[1]]), pairs)
        c = len(list(c1_filter)) / (self.model.n**2 * 3)
        c = self.model.nad_norm**2 - 2*c 
        
        if c < 0:
            c /= (self.model.nad_norm - self.model.nad_norm**2)
        else:
            if self.model.nad_norm <= 0.5:
                c /= self.model.nad_norm**2
            else:
                c /= (1 - self.model.nad_norm)**2

        if add_result:
            self.value_list.append(c)
        return c
