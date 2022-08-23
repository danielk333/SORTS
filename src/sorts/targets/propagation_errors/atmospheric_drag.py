#!/usr/bin/env python

'''
   Monte-Carlo sampling of errors due to atmospheric drag force uncertainty. Estimate a power-law 
   model of error standard deviation in along-track direction (largest error).
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ..space_object import SpaceObject as so


def get_inertial_basis(initial_ecef0, initial_ecef0_dt):
    """ Calculates the unit vectors for along track, normal (towards center of Earth), and cross-track 
    directions given the position vector at time, and the position vector at a small positive time 
    offset.

    Parameters
    ----------
    initial_ecef0 : numpy.ndarray (3/6, N)
        Position vector o fthe object at time *t*.
    initial_ecef0_dt : numpy.ndarray (3/6, N)
        Position vector o fthe object at time *t + dt*.

    Returns
    -------
    along_track : numpy.ndarray (3,)
        Unit vector in the along track direction of the trajectory.
    normal : numpy.ndarray (3,)
        Unit vector in the normal direction of the trajectory.
    cross_track : numpy.ndarray (3,)
        Unit vector in the cross track direction of the trajectory.
    """
    along_track = initial_ecef0_dt - initial_ecef0
    along_track = along_track/np.sqrt(along_track[0,:]**2.0 + along_track[1,:]**2.0 + along_track[2,:]**2.0)

    normal = initial_ecef0/np.sqrt(initial_ecef0[0,:]**2.0 + initial_ecef0[1,:]**2.0 + initial_ecef0[2,:]**2.0)

    cross_track = np.copy(normal)
    cross_track[:,:] = 0.0
    cross_track[0,:] = along_track[1,:]*normal[2,:] - along_track[2,:]*normal[1,:]
    cross_track[1,:] = along_track[2,:]*normal[0,:] - along_track[0,:]*normal[2,:]
    cross_track[2,:] = along_track[0,:]*normal[1,:] - along_track[1,:]*normal[0,:]

    cross_track = cross_track/np.sqrt(cross_track[0,:]**2.0 + cross_track[1,:]**2.0 + cross_track[2,:]**2.0)

    return (along_track, normal, cross_track)


def atmospheric_errors(space_object, atm_error_std=0.01, N_samps=100, plot=False, threshold_error=100.0, res=500):
    """ Estimates position errors as a function of time, assuming a certain error in atmospheric drag.

    This function performs a direct Monte-Carlo simulation to propagate the uncertainties in the atmospheric
    drag coefficient into space object time and position errors.

    The error is approximated by the power law

    .. math:: \\frac{\\sigma}{\\sigma_1} = (\\frac{t}{t_1})^\\alpha 

    The function estimates the power law parameters :math:`\\alpha`, :math:`\\sigma_0` from simulations
    (:math:`t_1` is taken to be the value `t[46]`)

    Parameters
    ----------
    space_object : :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
        Space object which position error we want to estimate.
    atm_pos_error_std : float, default =0.01
        Atmospheric drag coefficient error standard deviation.
    N_samps: int, default=100 
        Number of samples used in the direct MC error estimation algrithm.
    plot : bool, default=False
        If *True*, the error estimation results will be plotted as a function of time.
    threshold_pos_error : float, default=100.0
        If at least one along track position error is greater than this threshold, only the values higher
        than the threshold will be considered.
    res : float, default=500
        # not used - TODO

    Returns
    -------
    hour0 : float
        Simulation end time (in hours) or time points where the error is greater than the threshold value ``threshold_pos_error``.
    offset : float
        Value of the power law offset :math:`\\log(\\sigma_0)`.
    t_eval : float
        Power law evaluation point. Used to estimate the error as a function of time as
    alpha : float
        Power law exponent.
    """


    t = 10**(np.linspace(2, 6.2, num=100))
    t_dt = np.copy(t) + 1.0     
    initial_ecef = space_object.get_state(t)

    print("n_days %d"%(np.max(t)/24.0/3600.0))

    C_D0 = space_object.parameters['C_D']
    pos_error = np.copy(initial_ecef)
    pos_error[:, :] = 0.0

    t0 = time.time()
    
    for i in range(N_samps):
        space_object_perturbed = space_object.copy()
        space_object_perturbed.mu0 = np.random.rand(1)*360.0

        initial_ecef = space_object_perturbed.get_state(t)
        initial_ecef_dt = space_object_perturbed.get_state(t_dt)                
        k_along_track, k_normal, k_cross_track = get_inertial_basis(initial_ecef, initial_ecef_dt)
        
        C_D = C_D0 + C_D0*np.random.randn(1)[0]*atm_error_std
        space_object_perturbed.parameters['C_D'] = C_D
        
        perturbed_initial_ecef = space_object_perturbed.get_state(t)
        
        pos_error_now = (perturbed_initial_ecef - initial_ecef)
        pos_error[0,:] += np.abs(pos_error_now[0,:]*k_along_track[0,:] + pos_error_now[1,:]*k_along_track[1,:] + pos_error_now[2,:]*k_along_track[2,:])**2.0

        # difference in radius is the best estimate for radial distance error.
        pos_error[1,:] += np.abs(np.sqrt(initial_ecef[0,:]**2.0 + initial_ecef[1,:]**2.0 + initial_ecef[2,:]**2.0) - np.sqrt(perturbed_initial_ecef[0,:]**2.0 + perturbed_initial_ecef[1,:]**2.0 + perturbed_initial_ecef[2,:]**2.0))
        # and not this:  pos_error[1,:] += np.abs(pos_error_now[0,:]*norm[0,:] + pos_error_now[1,:]*norm[1,:] + pos_error_now[2,:]*norm[2,:])**2.0        
        pos_error[2,:] += np.abs(pos_error_now[0,:]*k_cross_track[0,:] + pos_error_now[1,:]*k_cross_track[1,:] + pos_error_now[2,:]*k_cross_track[2,:])**2.0

        elapsed_time = time.time() - t0
        print('{}/{} done - time elapsed {:<5.2f} h | estimated time remaining {:<5.2f}'.format(
            i+1,
            N_samps,
            elapsed_time/3600.0,
            elapsed_time/float(i + 1)*float(N_samps - i - 1)/3600.0,
        ))

    # check if threshold crossed
    along_track_pos_error = np.sqrt(pos_error[0,:]/N_samps)
    if np.max(along_track_pos_error) > threshold_pos_error:
        inds = np.where(along_track_pos_error > threshold_pos_error)[0][0]
        days = t/24.0/3600.0
        hour0 = 24.0*days[inds]
    else:
        hour0 = np.max(t/3600.0)

    # compute power law exponent, ofset and variance
    alpha = (np.log(pos_error[0, -1]/N_samps) - np.log(pos_error[0, 46]/N_samps))/(np.log(t[-1]) - np.log(t[46])) # (np.log(t[-1])-np.log(t[0]))*alpha=np.log(pos_error[0,-1]/N_samps) 
    offset = np.log(pos_error[0, 46]/N_samps)
    t_eval = t[46]
    var = np.exp((np.log(t) - np.log(t[46]))*alpha + offset)
    
    if plot:
        plt.loglog(t/24.0/3600.0, np.sqrt(pos_error[0,:]/N_samps), label="Along track")
        plt.loglog(t/24.0/3600.0, np.sqrt(var), label="Fit", alpha=0.5)
        plt.loglog(t/24.0/3600.0, np.sqrt(pos_error[1,:]/N_samps), label="Radial")
        plt.loglog(t/24.0/3600.0, np.sqrt(pos_error[2,:]/N_samps), label="Cross-track")

        if np.max(along_track_pos_error) > threshold_pos_error:    
            plt.axvline(days[inds])
            plt.text(days[inds], threshold_pos_error, "$\\tau = %1.1f$ hours" % (24*days[inds]))

        plt.grid()
        plt.axvline(np.max(t)/24.0/3600.0)
        plt.xlim([0, np.max(t)/24.0/3600.0])
        plt.legend()
        plt.ylabel("Cartesian position error (m)")
        plt.xlabel("Time (days)")
        #plt.title("Atmospheric drag uncertainty related errors"%(alpha))
        plt.title("a %1.0f (km) e %1.2f i %1.0f (deg) aop %1.0f (deg) raan %1.0f (deg)\nA %1.2f$\pm$ %d%% (m$^2$) mass %1.2f (kg)\n$\\alpha=%1.1f$ $t_1=%1.1f$ $\\beta=%1.1f$" 
            % (
                space_object.orbit.a,
                space_object.orbit.e,
                space_object.orbit.i,
                space_object.orbit.omega,
                space_object.orbit.Omega,
                space_object.parameters['A'],
                int(atm_pos_error_std*100.0),
                space_object.parameters['m'],
                alpha,
                t_eval,
                offset)
            )

    return (hour0, offset, t_eval, alpha)
