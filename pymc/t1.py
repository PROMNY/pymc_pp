from lattice import lattice
from fkham import fkham

lat = lattice(4, "graphene")
# lat.plot()
ham = fkham(lat)
print(len(lat.sub_matrix[0]), len(lat.sub_matrix[1]))
