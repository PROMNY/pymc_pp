from lattice import Lattice
from fkham import Hamiltonian

lat = Lattice(4, "graphene")
print(lat.sub_matrix)
lat.plot()
