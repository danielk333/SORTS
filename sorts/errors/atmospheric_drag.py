#!/usr/bin/env python

'''
   Monte-Carlo sampling of errors due to atmospheric drag force uncertainty.

   Estimate a power-law model of error standard deviation in along-track direction (largest error).

   Juha Vierinen
'''
import time
import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ..space_object import SpaceObject as so


def get_inertial_basis(ecef0,ecef0_dt):
    """
    Given pos vector, and pos vector at a small positive time offset,
    calculate unit vectors for along track, normal (towards center of Earth), and cross-track directions
    """
    along_track=ecef0_dt-ecef0
    along_track=along_track/n.sqrt(along_track[0,:]**2.0+along_track[1,:]**2.0+along_track[2,:]**2.0)
    normal = ecef0/n.sqrt(ecef0[0,:]**2.0+ecef0[1,:]**2.0+ecef0[2,:]**2.0)
    cross_track=n.copy(normal)
    cross_track[:,:]=0.0
    cross_track[0,:] = along_track[1,:]*normal[2,:] - along_track[2,:]*normal[1,:]
    cross_track[1,:] = along_track[2,:]*normal[0,:] - along_track[0,:]*normal[2,:]
    cross_track[2,:] = along_track[0,:]*normal[1,:] - along_track[1,:]*normal[0,:]
    cross_track=cross_track/n.sqrt(cross_track[0,:]**2.0+cross_track[1,:]**2.0+cross_track[2,:]**2.0)
    return(along_track,normal,cross_track)


def atmospheric_errors(o,a_err_std=0.01,N_samps=100,plot=False,threshold_error=100.0, res = 500):
    """
    Estimate position errors as a function of time, assuming
    a certain error in atmospheric drag.
    """


    t=10**(n.linspace(2,6.2,num=100))
    t_dt=n.copy(t)+1.0     
    ecef=o.get_state(t)

    print("n_days %d"%(n.max(t)/24.0/3600.0))
    C_D0=o.parameters['C_D']
    err=n.copy(ecef)
    err[:,:]=0.0

    t0 = time.time()
    
    for i in range(N_samps):
        o1=o.copy()
        o1.mu0=n.random.rand(1)*360.0

        ecef=o1.get_state(t)
        ecef_dt=o1.get_state(t_dt)                
        at,norm,ct=get_inertial_basis(ecef,ecef_dt)
        
        C_D=C_D0 + C_D0*n.random.randn(1)[0]*a_err_std
        o1.parameters['C_D']=C_D
        
        ecef1=o1.get_state(t)
        
        err_now=(ecef1-ecef)
        err[0,:]+=n.abs(err_now[0,:]*at[0,:]+err_now[1,:]*at[1,:]+err_now[2,:]*at[2,:])**2.0

        # difference in radius is the best estimate for radial distance error.
        err[1,:]+=n.abs(n.sqrt(ecef[0,:]**2.0+ecef[1,:]**2.0+ecef[2,:]**2.0) - n.sqrt(ecef1[0,:]**2.0+ecef1[1,:]**2.0+ecef1[2,:]**2.0))
#       and not this:  err[1,:]+=n.abs(err_now[0,:]*norm[0,:]+err_now[1,:]*norm[1,:]+err_now[2,:]*norm[2,:])**2.0        
        err[2,:]+=n.abs(err_now[0,:]*ct[0,:]+err_now[1,:]*ct[1,:]+err_now[2,:]*ct[2,:])**2.0

        elps = time.time() - t0
        print('{}/{} done - time elapsed {:<5.2f} h | estimated time remaining {:<5.2f}'.format(
            i+1,
            N_samps,
            elps/3600.0,
            elps/float(i+1)*float(N_samps - i - 1)/3600.0,
        ))

    ate=n.sqrt(err[0,:]/N_samps)
    if n.max(ate) > threshold_error:
        idx0=n.where(ate > threshold_error)[0][0]
        days=t/24.0/3600.0
        hour0=24.0*days[idx0]
    else:
        hour0=n.max(t/3600.0)

    alpha=(n.log(err[0,-1]/N_samps)-n.log(err[0,46]/N_samps))/(n.log(t[-1])-n.log(t[46]))
    #(n.log(t[-1])-n.log(t[0]))*alpha=n.log(err[0,-1]/N_samps) 

    offset=n.log(err[0,46]/N_samps)
    t1 = t[46]
    var=n.exp((n.log(t)-n.log(t[46]))*alpha + offset)
    
    if plot:
        plt.loglog(t/24.0/3600.0,n.sqrt(err[0,:]/N_samps),label="Along track")
        plt.loglog(t/24.0/3600.0,n.sqrt(var),label="Fit",alpha=0.5)
        plt.loglog(t/24.0/3600.0,n.sqrt(err[1,:]/N_samps),label="Radial")
        plt.loglog(t/24.0/3600.0,n.sqrt(err[2,:]/N_samps),label="Cross-track")
        if n.max(ate) > threshold_error:    
            plt.axvline(days[idx0])
            plt.text(days[idx0],threshold_error,"$\\tau=%1.1f$ hours"%(24*days[idx0]))        
        plt.grid()
        plt.axvline(n.max(t)/24.0/3600.0)
        plt.xlim([0,n.max(t)/24.0/3600.0])
        plt.legend()
        plt.ylabel("Cartesian position error (m)")
        plt.xlabel("Time (days)")
        #plt.title("Atmospheric drag uncertainty related errors"%(alpha))
        plt.title("a %1.0f (km) e %1.2f i %1.0f (deg) aop %1.0f (deg) raan %1.0f (deg)\nA %1.2f$\pm$ %d%% (m$^2$) mass %1.2f (kg)\n$\\alpha=%1.1f$ $t_1=%1.1f$ $\\beta=%1.1f$"%(o.orbit.a,o.orbit.e,o.orbit.i,o.orbit.omega,o.orbit.Omega,o.parameters['A'],int(a_err_std*100.0),o.parameters['m'],alpha,t1,offset))

    return(hour0,offset,t1,alpha)
