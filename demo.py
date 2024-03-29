import pymc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

lattice = pymc.GrapheneLattice(6)
FK = pymc.Hamiltonian(lattice, t=-1, U=2, cp=-1, T=0.2)
FK.put_adatoms(6*3, "random")

obs = pymc.ObsList([pymc.DeltaObs(FK),
    pymc.EnergyObs(FK), pymc.CVObs(FK),
    pymc.CorrelationObs(FK), pymc.NeObs(FK)])
obs_conv = pymc.ObsList(
    [pymc.DeltaObs(FK), pymc.EnergyObs(FK), pymc.CorrelationObs(FK)])
series = pymc.ObsSeries(obs, ["T"])

sym = pymc.Simulator(FK, pymc.metropolis_numpy, obs, obs_conv)

T_range = [0.2, 0.18, 0.16, 0.14, 0.12,
           0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
for T in tqdm(T_range, desc="Main loop"):
    FK.T = T

    sym.run_termalization(10**4)
    res = sym.run_measurements(10**4)
    series.add(res, [T])

res = series.get_df().values
np.savetxt("half_filling_2.csv", res, delimiter=",")