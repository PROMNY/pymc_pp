import numpy as np
from tqdm import tqdm
import cupy


class Opt(object):

    def metropolis(self, iter, T, cp, show_tqdm=False):
        assert T > 0, "T equal to zero"

        if show_tqdm:
            iterable = tqdm(range(iter))
        else:
            iterable = range(iter)

        acc = 0
        act = 0

        temp_empty = list(self.empty_sites)
        temp_filled = list(self.filled_sites)
        temp_H = np.copy(self.H)

        eigv = np.linalg.eigvalsh(temp_H)
        E0 = -sum(np.log(np.exp((cp-eigv)/T) + 1))

        for _ in iterable:
            ch_empty = np.random.choice(temp_empty)
            ch_filled = np.random.choice(temp_filled)

            temp_H[ch_empty, ch_empty] = self.U
            temp_H[ch_filled, ch_filled] = 0.0

            eigv = np.linalg.eigvalsh(temp_H)
            E = -sum(np.log(np.exp((cp-eigv)/T) + 1))

            if E <= E0:
                E0 = E
                self.H = temp_H
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                acc += 1

            elif np.exp((E0-E)/T) >= np.random.random():
                E0 = E
                self.H = temp_H
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                act += 1
            else:
                temp_H[ch_empty, ch_empty] = 0.0
                temp_H[ch_filled, ch_filled] = self.U

        return {"E": E0*T, "acc": acc/iter, "act": act/iter}

    def metropolis_cupy(self, iter, T, cp, show_tqdm=False):
        assert T > 0, "T equal to zero"

        if show_tqdm:
            iterable = tqdm(range(iter))
        else:
            iterable = range(iter)

        acc = 0
        act = 0

        temp_H_cupy = cupy.asarray(self.H)
        eigv = cupy.linalg.eigvalsh(temp_H_cupy)
        E0 = -cupy.sum(cupy.log(cupy.add(cupy.exp((cp-eigv)/T), 1)))

        temp_empty = list(self.empty_sites)
        temp_filled = list(self.filled_sites)

        for _ in iterable:
            ch_empty = np.random.choice(temp_empty)
            ch_filled = np.random.choice(temp_filled)

            temp_H_cupy[ch_empty, ch_empty] = self.U
            temp_H_cupy[ch_filled, ch_filled] = 0.0

            eigv = cupy.linalg.eigvalsh(temp_H_cupy)
            E = -cupy.sum(cupy.log(cupy.add(cupy.exp((cp-eigv)/T), 1)))

            if E <= E0:
                E0 = E
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                acc += 1

            elif np.exp((E0-E)/T) >= np.random.random():
                E0 = E
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                act += 1
            else:
                temp_H_cupy[ch_empty, ch_empty] = 0.0
                temp_H_cupy[ch_filled, ch_filled] = self.U

        self.H = temp_H_cupy.get()
        return {"E": E0*T, "acc": acc/iter, "act": act/iter}

    def metropolis_while(self, new_conf, max_iter, T, cp):
        assert T > 0, "T equal to zero"

        acc = 0
        act = 0
        iter = 0

        temp_empty = list(self.empty_sites)
        temp_filled = list(self.filled_sites)
        temp_H = np.copy(self.H)

        eigv = np.linalg.eigvalsh(temp_H)
        E0 = -sum(np.log(np.exp((cp-eigv)/T) + 1))

        while acc+act < new_conf and iter < max_iter:
            iter += 1
            ch_empty = np.random.choice(temp_empty)
            ch_filled = np.random.choice(temp_filled)

            temp_H[ch_empty, ch_empty] = self.U
            temp_H[ch_filled, ch_filled] = 0.0

            eigv = np.linalg.eigvalsh(temp_H)
            E = -sum(np.log(np.exp((cp-eigv)/T) + 1))

            if E <= E0:
                E0 = E
                self.H = temp_H
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                acc += 1

            elif np.exp((E0-E)/T) >= np.random.random():
                E0 = E
                self.H = temp_H
                temp_filled.remove(ch_filled)
                temp_empty.remove(ch_empty)
                temp_empty.append(ch_filled)
                temp_filled.append(ch_empty)
                act += 1
            else:
                temp_H[ch_empty, ch_empty] = 0.0
                temp_H[ch_filled, ch_filled] = self.U

        return {"E": E0*T, "acc": acc/iter, "act": act/iter, "iter": iter}
