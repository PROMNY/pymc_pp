from .basic_obs import BasicObs


class DeltaObs(BasicObs):

    def __init__(self, model):
        BasicObs.__init__(self, model)
        self.sublattice_A_set = set(self.model.lattice.sub_matrix[0])

    def calculate(self, add_result=True):

        filled_set = set(self.model.filled_sites)
        A = len(list(filled_set.intersection(self.sublattice_A_set)))
        d = abs(2*A - self.model.nad) / self.model.n**2

        if self.model.nad_norm <= 0.5:
            d /= self.model.nad_norm
        else:
            d /= (1 - self.model.nad_norm)

        if add_result:
            self.value_list.append(d)
        return d
