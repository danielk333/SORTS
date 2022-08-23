#!/usr/bin/env python

'''Miscellaneous functions

'''

import numpy as np
import scipy.constants


def signal_delay(st1, st2, ecef):
    ''' Compoutes the radar signal delay due to speed of light between station-1 to 
    ecef position to station-2.

    The computation of the time delay takes advantage of the *Time-of-Flight* principle

    .. math:: \\Delta_t = \\frac{(r_1 + r_2)}{c}

    Parameters
    ----------
    st1 : :class:`sorts.Station<sorts.radar.system.radar.station.Station>`
        First station transmitting the signal.
    st2 : :class:`sorts.Station<sorts.radar.system.radar.station.Station>`
        Second station receiving the signal.
    ecef : np.ndarray (3, N)

    Returns
    -------
    dt : float / np.ndarray (N,)
    '''
    # compute distances from points to stations
    r1 = np.linalg.norm(ecef - st1.ecef[:,None], axis=0)
    r2 = np.linalg.norm(ecef - st1.ecef[:,None], axis=0)

    # compute time delay
    dt = (r1 + r2)/scipy.constants.c

    return dt


def instantaneous_to_coherent(gain, groups, N_IPP, IPP_scale=1.0, units = 'dB'):
    ''' Converts from instantaneous gain to coherently integrated gain using pulse encoding schema, 
    subgroup setup and coherent integration setup. 
    
    Parameters
    ----------
    gain : float
        Instantaneous gain, linear units or in dB.
    groups : int
        Number of subgroups from witch signals are coherently combined, assumes subgroups are identical.
    N_IPP : int
        Number of pulses to coherently integrate.
    IPP_scale : float
        Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    units : str
        If string equals 'dB', assume input and output units should be dB, else use linear scale.
    
    Returns
    -------
    gain_coh : float
        Gain after coherent integration, linear units or in dB.
    '''
    if units == 'dB':
        return gain + 10.0*np.log10( groups*N_IPP*IPP_scale)
    else:
        return gain*(groups*N_IPP*IPP_scale)


def coherent_to_instantaneous(gain,groups,N_IPP,IPP_scale=1.0,units = 'dB'):
    ''' Convert from coherently integrated gain to instantaneous gain using pulse encoding schema, 
    subgroup setup and coherent integration setup.
    
    gain : float
        Coherently integrated gain, linear units or in dB.
    groups : int
        Number of subgroups from witch signals are coherently combined, assumes subgroups are identical.
    N_IPP : int
        Number of pulses to coherently integrate.
    IPP_scale : float
        Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    units : str
        If string equals 'dB', assume input and output units should be dB, else use linear scale.
    
    Returns
    -------
    gain_inst : float
        Instantaneous gain, linear units or in dB.
    '''
    if units == 'dB':
        return gain - 10.0*np.log10( groups*N_IPP*IPP_scale )
    else:
        return gain/(groups*N_IPP*IPP_scale)
