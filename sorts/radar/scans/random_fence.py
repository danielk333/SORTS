#!/usr/bin/env python

'''

'''

import numpy as np

from .fence import Fence

class RandomFence(Fence):
    '''General fence scan.
    '''

    def pointing(self, t):
        raise NotImplementedError()

        ind = (np.mod(t/self.cycle(), 1)*self.num).astype(np.int)
        if isinstance(t, float) or isinstance(t, int):
            shape = (3, )
        else:
            shape = (3, len(t))

        azelr = np.empty(shape, dtype=np.float64)
        azelr[0,...] = self._az[ind]
        azelr[1,...] = self._el[ind]
        azelr[2,...] = 1.0
        return azelr