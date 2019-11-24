from system import System
import matplotlib.pyplot as mp
from tqdm import tqdm
import numpy as np
a = System(8, "graphene", U=4)
b = System(8, "graphene", U=4)
#print(a.ala)
act =[]
acc =[]

a.put_adatoms(32, "random")
b.put_adatoms(32, "random")
a.calculate_delta()
b.calculate_delta()

for T in tqdm(np.linspace(0.5, 0.3, 5)):
    for _ in range(10):
        a.parallel_drop(10**3, T, cp=2, threads=4)
    a.calculate_delta()

    b.metropolis(4*10**4, T, cp=2)
    b.calculate_delta()



mp.plot(a.delta_list, label="delta")
mp.plot(b.delta_list, label="delta")
# mp.plot(acc, label="acc")
# mp.plot(act, label="act")
mp.legend()
mp.show()
