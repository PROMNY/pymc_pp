import numpy as np
from tqdm import tqdm


def metropolis_numpy(model, n_iter, T, cp, show_tqdm=True):
    assert T > 0, "T equal to zero"
    be = 1.0 / T
    
    if show_tqdm:
        iterable = tqdm(range(iter))
    else:
        iterable = range(iter)

    acc = 0
    act = 0

    model.cp = cp
    model.T = T

    E0 = model.get_F()

    for _ in iterable:
        ch_empty = np.random.choice(model.empty_sites)
        ch_filled = np.random.choice(model.filled_sites)

        model.swap_in_temp_H(ch_filled, ch_empty)
        model.calculate_eigv(temp=True)
        E = model.get_F(temp=True)

        if E <= E0:
            E0 = E
            model.swap_in_H(ch_filled, ch_empty)
            acc += 1
        elif np.exp((E0-E)*be) > np.random.random_sample():
            E0 = E
            model.swap_in_H(ch_filled, ch_empty)
            act += 1
        else:
            model.un_swap_in_temp_H(ch_filled, ch_empty)

    return {"E": E0, "acc": acc/iter, "act": act/iter}
