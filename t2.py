from system import System
import matplotlib.pyplot as mp
from tqdm import tqdm
import numpy as np
from itertools import product

a = System(10, "graphene", U=2)
a.put_adatoms(50, "random")
cp = 1
T = [0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04]
T = np.linspace(0.45, 0.2, 25)
cw = []
delta = []
c1 = []

for t in tqdm(T):
    print(a.metropolis(10**5, t, cp))
    for _ in range(2000):
        print(a.metropolis_while(5, 100, t, cp))
        a.calculate_c1()
        a.calculate_delta()
        a.calculate_E(t, cp)
    delta.append(a.avrg_delta())
    c1.append(a.avrg_c1())
    cw.append(a.avrg_cv(t))
    a.reset_obs()
    print(t, cw[-1], delta[-1], c1[-1])
cw /= np.amax(cw)


mp.plot(T, c1, label="c1")
mp.plot(T, delta, label="d")
mp.plot(T, cw, label="cw")
mp.legend()
mp.show()

print(T)
print(delta)
print(c1)
print(cw)
