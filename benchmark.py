import pymc
import time
from tqdm import tqdm
import numpy as np

ts_start = time.time()

lattice = pymc.GrapheneLattice(10)
FK = pymc.Hamiltonian(lattice, t=-1, U=2, cp=1, T=0.2)
FK.put_adatoms(6*3, "random")

obs = pymc.ObsList([pymc.DeltaObs(FK),
    pymc.EnergyObs(FK), pymc.CVObs(FK),
    pymc.CorrelationObs(FK), pymc.NeObs(FK)])
series = pymc.ObsSeries(obs, ["T"])

sym = pymc.Simulator(FK, pymc.metropolis_numpy, obs)

T_range = [0.2, 0.10, 0.01]
for T in tqdm(T_range, desc="Main loop"):
    FK.T = T

    res = sym.run_measurements(10**2)
    series.add(res, [T])

ts_end = time.time()


print(f"Took: {ts_end - ts_start}")
print(series.get_df())
