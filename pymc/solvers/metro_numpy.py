import numpy as np
from tqdm import tqdm


def metropolis_numpy(model, n_iter, show_tqdm=True):
    assert model.T > 0, "T equal to zero"
    be = 1.0 / model.T
    
    if show_tqdm:
        iterable = tqdm(range(n_iter))
    else:
        iterable = range(n_iter)

    acc = 0
    act = 0
    model.calculate_eigv()
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

    return {"E": E0, "acc": acc/n_iter, "act": act/n_iter}
