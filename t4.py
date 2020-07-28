import pymc as p
from tqdm import tqdm
import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as mp
n = 10
iter = 10**3
T = 0.08
a = p.System(n, "graphene", U=2)
a.put_adatoms(int(0.5*n**2), "random")

start = time()
a.metropolis(iter=1, T=T, cp=1, runner="cupy")
for _ in range(10):
    a.metropolis(iter=iter, T=T, cp=1, runner="cupy")
print("numpy", time()-start)

a = p.System(n, "graphene", U=2)
a.put_adatoms(int(0.5*n**2), "random")

start = time()
a.metropolis(iter=1, T=T, cp=1, runner="fortran")
for _ in range(10):
    a.metropolis(iter=iter, T=T, cp=1, runner="fortran")
print("numpy", time()-start)


"""
start = time()
print(a.metropolis_cupy(10**5, 0.2, 1))
print(time()-start)
"""
