import pymc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

lattice = pymc.GrapheneLattice(10)
FK = pymc.Hamiltonian(lattice, t=-1, U=2, cp=0, T=0.2)
FK.put_adatoms(50, "random")
o_delta = pymc.CorrelationObs(FK)
o_energy = pymc.EnergyObs(FK)

energy = []
c1 = []
T_range = [0.2, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
for T in T_range:
    
    FK.T = T
    o_energy.reset()
    o_delta.reset()

    for _ in tqdm(range(500)):
        pymc.metropolis_numpy(FK, 100, show_tqdm=False)
        o_delta.calculate()
        o_energy.calculate()
        if o_energy.has_converged(min_len=50) and o_delta.has_converged(min_len=50):
            break
    
    o_energy.reset()
    o_delta.reset()

    for _ in tqdm(range(50)):
        pymc.metropolis_numpy(FK, 100, show_tqdm=False)
        o_delta.calculate()
        o_energy.calculate()

    energy.append(o_energy.get_result())
    c1.append(o_delta.get_result())


plt.plot(T_range, energy, label="delta")
plt.show()

plt.plot(T_range, c1, label="delta")
plt.show()
