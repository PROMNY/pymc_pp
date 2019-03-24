from lattice import lattice
from fkham import fkham

lat = lattice(4, "graphene")
print(lat.sub_matrix)
lat.plot()
