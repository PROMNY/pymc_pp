import pymc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

lattice = pymc.GrapheneLattice(10)
FK = pymc.Hamiltonian(lattice, t=-1, U=2, cp=1, T=0.2)
FK.put_adatoms(50, "random")

o_delta = pymc.DeltaObs(FK)
o_energy = pymc.EnergyObs(FK)
o_cv = pymc.CVObs(FK)
o_cor = pymc.CorrelationObs(FK)

energy = []
cor = []
cv = []
delta = []

T_range = [0.2, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
for T in tqdm(T_range):
    FK.T = T
    o_energy.reset()
    o_delta.reset()
    o_cv.reset()
    o_cor.reset()
    print(o_cv.model.T)

    print(f"Running termalization for T={T}")
    for _ in range(500):
        pymc.metropolis_numpy(FK, 100, show_tqdm=False)
        o_delta.calculate()
        o_energy.calculate()
        
        if o_energy.has_converged(min_len=100) and o_delta.has_converged(min_len=100):
            break
    
    o_energy.reset()
    o_delta.reset()

    print(f"Running measurements for T={T}")
    for _ in range(200):
        pymc.metropolis_numpy(FK, 100, show_tqdm=False)
        o_delta.calculate()
        o_energy.calculate()
        o_cv.calculate()
        o_cor.calculate()

    energy.append(o_energy.get_result())
    delta.append(o_delta.get_result())
    cv.append(o_cv.get_result())
    cor.append(o_cor.get_result())

print(o_cv.model.T)
print(np.amax(cv))    
print(cv)
cv /= np.amax(cv)
print(np.amax(cv))
print(cv)

#plt.show()
plt.plot(T_range, cv, label="CV")
plt.plot(T_range, delta, label="delta")
plt.plot(T_range, cor, label="C1")
plt.xlabel("T")
#plt.plot(T_range, cv, label="cv")
plt.legend()
plt.show()

