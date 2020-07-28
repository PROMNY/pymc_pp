import pymc as p
from tqdm import tqdm
import numpy as np
from itertools import product
from time import time
import matplotlib.pyplot as mp
size = range(52, 0, 6)
print(list(size))
tnp = []
tfo = []
iter = 10**3
T = 0.08
"""
for n in size:
    a = p.System(n, "graphene", U=2)
    a.put_adatoms(int(0.5*n**2), "random")

    start = time()
    a.metropolis(iter=iter, T=T, cp=1, runner="cupy")
    tnp.append(time()-start)

    print(tnp)
"""
tnp = np.asarray([1.3919579982757568, 7.4830427169799805, 24.974742889404297, 63.979087114334106, 154.73588275909424, 299.95209670066833, 957.1453492641449])
tfo = np.asarray([0.44379401206970215, 4.707715749740601, 28.637855052947998, 115.35234332084656, 378.84184527397156, 1310.963273525238, 4144.880655527115])
tcu = np.asarray([4.085265159606934, 12.296858072280884, 43.202712535858154, 93.62809014320374, 213.93172073364258, 453.4039671421051, 959.0700483322144])
mp.plot(size, tnp, label="NumPy (~16 cores)")
mp.plot(size, tfo, label="FORTRAN (1 core)")
mp.plot(size, tcu, label="CuPy")
mp.legend()
mp.show()
