import pymc as p
from tqdm import tqdm
import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as mp

n=10
a = p.System(n, "graphene", U=2)
a.put_adatoms(int(0.5*n**2), "random")

start = time()
print(a.metropolis(10**5, 0.2, 1))
print(time()-start)

start = time()
print(a.metropolis_cupy(10**5, 0.2, 1))
print(time()-start)
