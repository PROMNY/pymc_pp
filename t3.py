import pymc as p
from tqdm import tqdm
import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as mp


size = [10, 20, 30, 40, 50, 60, 70, 80]

gp = []
cp = []
for n in size:
    a = p.System(n, "graphene", U=2)
    a.put_adatoms(int(0.5*n**2), "random")

    start = time()
    print(a.metropolis(10, 0.2, 1))
    cp.append(time()-start)

    a = p.System(n, "graphene", U=2)
    a.put_adatoms(int(0.5*n**2), "random")

    start = time()
    print(a.metropolis_cp(10, 0.2, 1))
    gp.append(time()-start)

mp.plot(size, cp,"bo", label="CPU")
mp.plot(size, gp,"ro", label="GPU")
mp.legend()
mp.show()
print(size)
print(cp)
print(gp)
