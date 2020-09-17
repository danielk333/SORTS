#!/usr/bin/env python

'''Interpolation functions.

'''

from abc import ABC, abstractmethod

import numpy as np

class Interpolator(ABC):

    def __init__(self, states, t):
        self.states = states
        self.t = t


    @abstractmethod
    def get_state(self, t, **kwargs):
        pass



class Legendre8(Interpolator):
    '''Order-8 Legendre polynomial interpolation of uniformly distributed states.
    '''

    def __init__(self, states, t):
        if len(t) < 9:
            raise ValueError(f'Cannot performance 8-degree interpolation with {len(t)} points')
        super().__init__(states, t)

    def get_state(self, t):
        intep_states = legendre8(self.states.T, self.t.min(), self.t.max(), t, ti=None)
        return intep_states.T





def legendre8(table, t1, tN, t, ti=None):
    """Order-8 Legendre polynomial interpolation

    Code adapted from the `gdar` system developed by NORCE Norwegian Research Centre AS, used with permission.

    Parameters:
      table: M vectors (M >= 9) to interpolate between, each containing N
          values, e.g. for a position vector N=3 (x, y, z) table.shape = (M, N)
      t1: time corresponding to M=0
      tN: time corresponding to M=N-1
      t: times at which to provide N-dimensional answer.
      ti: indices which sort t in monotonic order

      This version requires t to be sorted, which enables optimization
      by interpolating all t in an interval between two nodes as a single
      expression. For the typical case where there are many fewer intervals
      to consider than distinct values to interpolate, this gives much
      improved performance."""

    M, N = table.shape
    t = np.atleast_1d(t)

    tdiff = t[1:]-t[:-1]
    tzero = 0*tdiff

    if ti is None:
        if np.any(tdiff < tzero) and np.any(tdiff > tzero):
            # Input must be monotonically increasing or decreasing
            # raise RuntimeError("Input t must be sorted")
            # print("Sorting inputs ...")
            ti = np.argsort(t)
            # print("done")

    if ti is not None:
        t = t[ti]

    rval = np.zeros((len(t), N))

    den = np.array([40320., -5040., 1440., -720., 576.,
                    -720., 1440., -5040., 40320.])
    trel = (t-t1)/(tN-t1)*(M-1)
    tind = np.clip(np.round(trel-4),  0, M-9).astype(int)

    u, ix = np.unique(tind, return_index=True)
    counts = np.r_[ix[1:], len(tind)] - ix

    err_save = np.seterr(invalid='ignore')
    for val, ix0, count in zip(u, ix, counts):
        six = slice(ix0, ix0+count)
        xx = (trel[six]-val)[np.newaxis, :] - np.arange(9)[:, np.newaxis]       # {9, count}
        num = np.prod(xx, 0)                                    # {count}

        rval[six] = np.dot((num[np.newaxis, :]/den[:, np.newaxis]/xx).T, table[val:val+9])

        zz = np.where(num == 0)[0]
        if len(zz):
            rval[six][zz] = table[np.round(trel[six][zz]-val).astype(int) +
                                  val]
    np.seterr(**err_save)

    if ti is not None:
        # Restore input order
        rval, tmp = np.zeros_like(rval), rval
        rval[ti] = tmp
    return rval


def legendre8_loop(table, t1, tN, t):
    """Order-8 Legendre polynomial interpolation

    Code adapted from the `gdar` system developed by NORCE Norwegian Research Centre AS, used with permission.

    Parameters:
      table: M vectors (M >= 9) to interpolate between, each containing N
          values, e.g. for a position vector N=3 (x, y, z) table.shape = (M, N)
      t1: time corresponding to M=0
      tN: time corresponding to M=N-1
      t: times at which to provide N-dimensional answer.

      This version loops over all interpolation instants t.
      Direct port of IDL, which was directly ported from Fortran."""

    # Return value: shape (P, N)
    M, N = table.shape
    t = np.atleast_1d(t)
    P = len(t)

    rval = np.zeros((P, N))

    # denominators are -1^n*n!*(8-n)! for n in [0, ..., 8]
    # (https://oeis.org/A098361)
    den = np.array([40320., -5040., 1440., -720., 576.,
                    -720., 1440., -5040., 40320.])

    for i in range(P):
        trel = (t[i]-t1)/(tN-t1)*(M-1)
        tind = np.clip(np.round(trel-4), 0, M-9).astype(int)

        x = trel-tind
        num = np.prod(x-np.arange(9))   # x*(x-1)*(x-2)* ... * (x-8)

        if num == 0:
            rval[i] = table[tind + np.rint(x).astype(int)]
        else:
            rval[i] = np.dot(table[tind:tind+9].T, num/den/(x-np.arange(9)))

    return rval


