#!/usr/bin/env python

''' Interpolation functions. '''
from abc import ABC, abstractmethod

import numpy as np


class Interpolator(ABC):
    ''' Base Interpolation class that mimics the behavior of :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>` so that a 
    :class:`Interpolator` instance can be used instead.

    To create a Interpolator one must define the :code:`get_state` method. to return interpolated
    This method should return states based on the data contained in the instance.
    This data is preferably internalized at instantiation.
    
    Parameters 
    ----------
    states : numpy.ndarray (6, N) 
        Array of space object states to interpolate between.
    t : numpy.ndarray (N,) 
        Vector of time points corresponding to the states.
    '''

    def __init__(self, states, t):
        ''' Default class constructor. '''
        self.states = states
        ''' Array of space object states to interpolate between. '''
        self.t = np.atleast_1d(t)
        ''' Vector of time points corresponding to the states (in seconds). '''


    @abstractmethod
    def get_state(self, t, **kwargs):
        ''' Interpolates the states at each given time point.

        Parameters
        ----------
        t : float / numpy.ndarray (N,)
            Time point where the states are to be interpolated (in seconds) relative 
            to the space object reference epoch.

        Returns
        -------
        intep_states : numpy.ndarray (6, N)
            Interpolated space object states.
        '''
        pass



class Legendre8(Interpolator):
    '''Order-8 Legendre polynomial interpolation of uniformly distributed states.

    The number of states used to define the intepolation table must be greater or equal
    to 9.

    Parameters 
    ----------
    states : numpy.ndarray (6, N) 
        Array of space object states to interpolate between.
    t : numpy.ndarray (N,) 
        Vector of time points corresponding to the states.
    '''

    def __init__(self, states, t):
        ''' Default class constructor. '''
        if len(t) < 9:
            raise ValueError(f'Cannot performance 8-degree interpolation with {len(t)} points')
        super().__init__(states, t)


    def get_state(self, t):
        ''' Interpolates the states at each given time point.

        Parameters
        ----------
        t : float / numpy.ndarray (N,)
            Time point where the states are to be interpolated (in seconds) relative 
            to the space object reference epoch.

        Returns
        -------
        intep_states : numpy.ndarray (6, N)
            Interpolated space object states.
        '''
        intep_states = legendre8(self.states.T, self.t.min(), self.t.max(), t, ti=None)
        return intep_states.T




class Linear(Interpolator):
    ''' Linear interpolation between states.

    This class defines the default linear interpolation scheme. It can be used to perform 
    a linear interpolation of vectors.

    Parameters 
    ----------
    states : numpy.ndarray (6, N) 
        Array of space object states to interpolate between.
    t : numpy.ndarray (N,) 
        Vector of time points corresponding to the states.
    '''

    def __init__(self, states, t):
        ''' Default class constructor. '''
        super().__init__(states, t)

        self.t_diffs = np.diff(t)


    def get_state(self, t):
        ''' Interpolates the states at each given time point.

        Parameters
        ----------
        t : float / numpy.ndarray (N,)
            Time point where the states are to be interpolated (in seconds) relative 
            to the space object reference epoch.

        Returns
        -------
        intep_states : numpy.ndarray (6, N)
            Interpolated space object states.
        '''
        st_t = self.t.flatten()
        in_t = np.atleast_1d(t).flatten()
        t_mat = st_t[:,None] - in_t[None,:]

        inds = np.argmax(t_mat > 0, axis=0) - 1

        dts = -t_mat[inds,np.arange(len(t))]
        frac = dts/self.t_diffs[inds]
        
        return self.states[:,inds]*(1 - frac) + self.states[:,inds+1]*frac





def legendre8(table, t1, tN, t, ti=None):
    """ Order-8 Legendre polynomial interpolation

    Code adapted from the `gdar` system developed by NORCE Norwegian Research Centre AS, used with permission.

    .. note::
        the state array must possess at least 9 states to be able to perform the interpolation.

    This version requires t to be sorted, which enables optimization
    by interpolating all t in an interval between two nodes as a single
    expression. For the typical case where there are many fewer intervals
    to consider than distinct values to interpolate, this gives much
    improved performance.

    Parameters
    ----------
    table : np.ndarray (M, N)
        M vectors (M >= 9) to interpolate between, each containing N
        values, e.g. for a position vector N=3 (x, y, z).
    t1 : float
        Time point corresponding to ``M = 0``.
    tN: float
        Time point corresponding to ``M = M_max-1``.
    t: np.ndarray (n,)
        Times at which to provide n-dimensional answer.
    ti : np.ndarray (n,)
        Indices which sort t in monotonic order.

    Returns
    -------
    rval : numpy.ndarray (n,)
        Interpolated vectors at each time point t.
    """

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

    den = np.array([40320., -5040., 1440., -720., 576., -720., 1440., -5040., 40320.])
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
    """ Order-8 Legendre polynomial interpolation.

    Code adapted from the `gdar` system developed by NORCE Norwegian Research Centre AS, used with permission.
    This version loops over all interpolation instants t. Direct port of IDL, which was directly ported from Fortran.

    Parameters
    ----------
    table : np.ndarray (M, N)
        M vectors (M >= 9) to interpolate between, each containing N
        values, e.g. for a position vector N=3 (x, y, z).
    t1 : float
        Time point corresponding to ``M = 0``.
    tN: float
        Time point corresponding to ``M = M_max-1``.
    t: np.ndarray (n,)
        Times at which to provide n-dimensional answer.

    Returns
    -------
    rval : numpy.ndarray (n,)
        Interpolated vectors at each time point t.
    """

    # Return value: shape (P, N)
    M, N = table.shape
    t = np.atleast_1d(t)
    P = len(t)

    rval = np.zeros((P, N))

    # denominators are -1^n*n!*(8-n)! for n in [0, ..., 8]
    # (https://oeis.org/A098361)
    den = np.array([40320., -5040., 1440., -720., 576., -720., 1440., -5040., 40320.])

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