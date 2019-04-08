from lattice import Lattice
from fkham import Hamiltonian

lat = Lattice(4, "graphene")
ham = Hamiltonian(lat, 1)
ham.put_adatoms(15, "sublattice")
ham.plot()
print(ham.H)
