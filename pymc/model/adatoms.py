import numpy as np


class AdatomsMixin():

    def put_adatoms(self, nad, order="random"):
        assert (self.U != 0), "U is equal to zero, adatoms can not be put"
        assert (nad < self.n**2), "too many adatoms"
        assert (order in ["random", "sublattice", "separation"])

        self.nad = nad
        self.nad_norm = nad / self.n**2
        self.nad_to_put = nad

        for i in range(self.n**2):
            self.H[i, i] = 0.0

        if order == "random":
            self.put_adatoms_order_random()
        elif order == "sublattice":
            self.put_adatoms_order_sublattice()
        elif order == "separation":
            self.put_adatoms_order_separation()

        self.filled_sites = [i for i in range(self.n**2) if self.H[i, i] != 0.0]
        self.empty_sites = [i for i in range(self.n**2) if i not in self.filled_sites]
        
        self.temp_H = np.copy(self.H)

    def put_adatoms_order_random(self):
        index = np.random.choice(self.n**2, self.nad_to_put, replace=False)
        for i in index:
            self.H[i, i] = self.U

    def put_adatoms_order_separation(self):
        i = 0
        while i < self.nad_to_put:
            x_index = i % self.n
            y_index = i // self.n
            k = x_index * self.n + y_index
            self.H[k, k] = self.U
            i += 1

    def put_adatoms_order_sublattice(self):

        if self.nad_to_put >= self.n**2 // 2:
            for i in self.lattice.sub_matrix[0]:
                self.H[i, i] = self.U
            self.nad_to_put -= self.n**2 // 2
        
        if self.nad_to_put > 0:
            index = np.random.choice(self.lattice.sub_matrix[1],
                                    self.nad_to_put, replace=False)
            for i in index:
                self.H[i, i] = self.U

    def swap_in_temp_H(self, i, j):
        self.temp_H[i, i] = 0.0
        self.temp_H[j, j] =  self.U

    def swap_in_H(self, i, j):
        self.H[i, i] = 0.0
        self.H[j, j] =  self.U
        
        self.filled_sites.remove(i)
        self.empty_sites.remove(j)
        
        self.filled_sites.append(j)
        self.empty_sites.append(i)

        self.temp_H = np.copy(self.H)

    def un_swap_in_temp_H(self, i, j):
        self.temp_H[j, j] = 0.0
        self.temp_H[i, i] =  self.U    
