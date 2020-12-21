import numpy as np
from tqdm import tqdm
try:
    import cupy
except ImportError:
    print("CuPy is not installed")

# experimental !!!
def metropolis_cupy(model, n_iter, T, cp, show_tqdm=True):
    assert T > 0, "T equal to zero"
    be = 1.0 / T

    if show_tqdm:
        iterable = tqdm(range(iter))
    else:
        iterable = range(iter)

    acc = 0
    act = 0

    temp_H_cupy = cupy.asarray(model.H)
    eigv = cupy.linalg.eigvalsh(temp_H_cupy)
    E0 = -T*cupy.sum(cupy.log(cupy.add(cupy.exp((cp-eigv)*be), 1)))

    temp_empty = list(model.empty_sites)
    temp_filled = list(model.filled_sites)

    for _ in iterable:
        ch_empty = np.random.choice(temp_empty)
        ch_filled = np.random.choice(temp_filled)

        temp_H_cupy[ch_empty, ch_empty] = model.U
        temp_H_cupy[ch_filled, ch_filled] = 0.0

        eigv = cupy.linalg.eigvalsh(temp_H_cupy)
        E = -T*cupy.sum(cupy.log(cupy.add(cupy.exp((cp-eigv)*be), 1)))

        if E <= E0:
            E0 = E
            temp_filled.remove(ch_filled)
            temp_empty.remove(ch_empty)
            temp_empty.append(ch_filled)
            temp_filled.append(ch_empty)
            acc += 1

        elif np.exp((E0-E)*be) >= np.random.random_sample():
            E0 = E
            temp_filled.remove(ch_filled)
            temp_empty.remove(ch_empty)
            temp_empty.append(ch_filled)
            temp_filled.append(ch_empty)
            act += 1
        else:
            temp_H_cupy[ch_empty, ch_empty] = 0.0
            temp_H_cupy[ch_filled, ch_filled] = model.U

    model.empty_sites = np.asarray(temp_empty)
    model.filled_sites = np.asarray(temp_filled)

    model.H = np.asarray(temp_H_cupy.get())
    return {"E": E0, "acc": acc/iter, "act": act/iter}
