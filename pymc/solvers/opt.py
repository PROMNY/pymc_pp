import numpy as np
from tqdm import tqdm
from .metro_numpy import MetroNumpy
from .metro_fort import MetroFort
from .metro_cupy import MetroCupy


class Opt(MetroNumpy, MetroFort, MetroCupy):
    def metropolis(self, iter, T, cp, runner, show_tqdm=False):
        assert T > 0, "T equal to zero"
        if runner == "numpy":
            self.metropolis_numpy(iter, T, cp, show_tqdm)
        elif runner == "fortran":
            self.metropolis_fort(iter, T, cp, show_tqdm)
        elif runner == "cupy":
            self.metropolis_cupy(iter, T, cp, show_tqdm)

    """
    def metropolis_cupy(self, iter, T, cp, show_tqdm=False):



    """

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
