import numpy as np
from tqdm import tqdm


class Opt(object):

    def metropolis(self, iter, T, cp, show_tqdm=False):
        assert T > 0, "T equal to zero"

        eigv = np.linalg.eigvalsh(self.H)
        E0 = -sum(np.log(np.exp((cp-eigv)/T) + 1))
        if show_tqdm:
            iterable = tqdm(range(iter))
        else:
            iterable = range(iter)

        acc = 0
        act = 0

        for _ in iterable:
            ch_empty = np.random.choice(list(self.empty_sites))
            ch_filled = np.random.choice(list(self.filled_sites))

            temp_H = np.copy(self.H)
            temp_H[ch_empty, ch_empty] = self.U
            temp_H[ch_filled, ch_filled] = 0.0

            eigv = np.linalg.eigvalsh(temp_H)
            E = -sum(np.log(np.exp((cp-eigv)/T) + 1))

            if E <= E0:
                E0 = E
                self.H = temp_H
                self.filled_sites.remove(ch_filled)
                self.empty_sites.remove(ch_empty)
                self.empty_sites.add(ch_filled)
                self.filled_sites.add(ch_empty)
                acc += 1

            elif np.exp((E0-E)/T) >= np.random.random():
                E0 = E
                self.H = temp_H
                self.filled_sites.remove(ch_filled)
                self.empty_sites.remove(ch_empty)
                self.empty_sites.add(ch_filled)
                self.filled_sites.add(ch_empty)
                act += 1

        return {"E": E0*T, "acc": acc/iter, "act": act/iter}

    def metropolis_while(self, new_conf, max_iter, T, cp):
        assert T > 0, "T equal to zero"

        eigv = np.linalg.eigvalsh(self.H)
        E0 = -sum(np.log(np.exp((cp-eigv)/T) + 1))

        acc = 0
        act = 0
        iter = 0

        while acc+act < new_conf and iter < max_iter:
            iter += 1
            ch_empty = np.random.choice(list(self.empty_sites))
            ch_filled = np.random.choice(list(self.filled_sites))

            temp_H = np.copy(self.H)
            temp_H[ch_empty, ch_empty] = self.U
            temp_H[ch_filled, ch_filled] = 0.0

            eigv = np.linalg.eigvalsh(temp_H)
            E = -sum(np.log(np.exp((cp-eigv)/T) + 1))

            if E <= E0:
                E0 = E
                self.H = temp_H
                self.filled_sites.remove(ch_filled)
                self.empty_sites.remove(ch_empty)
                self.empty_sites.add(ch_filled)
                self.filled_sites.add(ch_empty)
                acc += 1

            elif np.exp((E0-E)/T) >= np.random.random():
                E0 = E
                self.H = temp_H
                self.filled_sites.remove(ch_filled)
                self.empty_sites.remove(ch_empty)
                self.empty_sites.add(ch_filled)
                self.filled_sites.add(ch_empty)
                act += 1

        return {"E": E0*T, "acc": acc/iter, "act": act/iter, "iter": iter}
