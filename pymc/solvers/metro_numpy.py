import numpy as np
from tqdm import tqdm


class MetroNumpy(object):
    def metropolis_numpy(self, iter, T, cp, show_tqdm):
        assert T > 0, "T equal to zero"
        be = 1.0 / T
        if show_tqdm:
            iterable = tqdm(range(iter))
        else:
            iterable = range(iter)

        acc = 0
        act = 0
        np.random.random()

        temp_empty = list(self.empty_sites)
        temp_filled = list(self.filled_sites)

        E0 = self.get_F(T, cp)

        for _ in iterable:
            ch_empty = np.random.choice(temp_empty)
            ch_filled = np.random.choice(temp_filled)

            self.H[ch_empty, ch_empty] = self.U
            self.H[ch_filled, ch_filled] = 0.0

            E = self.get_F(T, cp)

            if E <= E0:
                E0 = E
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                acc += 1
            elif np.exp((E0-E)*be) > np.random.random_sample():
                E0 = E
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                act += 1
            else:
                self.H[ch_empty, ch_empty] = 0.0
                self.H[ch_filled, ch_filled] = self.U

        self.empty_sites = set(temp_empty)
        self.filled_sites = set(temp_filled)

        print(self.get_F(T, cp))
        return {"E": E0, "acc": acc/iter, "act": act/iter}
