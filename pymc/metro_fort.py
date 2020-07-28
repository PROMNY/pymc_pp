from .METRO import metro
import numpy as np


class MetroFort(object):
    def metropolis_fort(self, iter, T, cp, show_tqdm):
        assert T > 0, "T equal to zero"
        self.H = np.asarray(metro(h=self.H, n=self.n**2, cp=cp, t=T,
                                  u=self.U, iter=iter))
        print(self.get_F(T, cp))

#        return {"E": E0*T, "acc": acc/iter, "act": act/iter}
