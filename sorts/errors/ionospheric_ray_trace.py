#!/usr/bin/env python

'''Ionospheric radio propagation effects studied using ray-tracing.

2016-2018 Juha Vierinen
2018-2020 Juha Vierinen, Daniel Kastinen

'''

try:
    from pyglow.pyglow import Point
    from pyglow import coord as gcoord
except ImportError:
    Point = None
    gcoord = None

from astropy.time import Time

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.constants as constants

from .. import frames
from .errors import Errors

def calculate_delay(
        time,
        lat,
        lon,
        frequency,
        elevation,
    ):
    '''TODO: Docstring
    '''
    

    if Point is None or gcoord is None:
        raise ImportError('pyglow must be installed to calculate delay')

    if not isinstance(time, Time):
        dn = time
    else:
        dn = time.tt.datetime

    num=500
    alts=np.linspace(0,4000,num=num)
    distance=np.linspace(0,4000,num=num)
    ne=np.zeros(num)
    xyz_prev=0.0
    for ai,a in enumerate(alts):
        llh=coord.az_el_r2geodetic(lat, lon, 0, 180.0, elevation, a*1e3)

        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            ne[ai]=pt.ne*1e6
        else:
            ne[ai]=0.0
        xyz=gcoord.lla2ecef(np.array([lat,lon,a]))[0]
        if ai==0:
            distance[ai]=0.0
        else:
            distance[ai]=np.sqrt(np.dot(xyz-xyz_prev,xyz-xyz_prev))+distance[ai-1]
        xyz_prev=xyz
        
    f_p=8.98*np.sqrt(ne)
    v_g = constants.c*np.sqrt(1-(f_p/frequency)**2.0)

    dt2=integrate.simps(1.0 - 1.0/(np.sqrt(1-(f_p/frequency)**2.0)),distance*1e3)/constants.c

    return dt2, ne, distance 


def ray_trace(
        time,
        lat,
        lon,
        frequency,
        elevation,
        azimuth,
    ):
    '''TODO: Docstring
    '''
    
    
    if Point is None or gcoord is None:
        raise ImportError('pyglow must be installed to ray trace')

    if not isinstance(time, Time):
        dn = time
    else:
        dn = time.tt.datetime

    num=1000
    alts=np.linspace(0,4000,num=num)
    distance=np.linspace(0,4000,num=num)
    ne=np.zeros(num)
    ne2=np.zeros(num)
    dnex=np.zeros(num)
    dtheta=np.zeros(num)
    dalt=np.zeros(num)
    dney=np.zeros(num)
    dnez=np.zeros(num)
    xyz_prev=0.0
    px=np.zeros(num)
    dk=np.zeros(num)
    py=np.zeros(num)
    pz=np.zeros(num)
    p0x=np.zeros(num)
    p0y=np.zeros(num)
    p0z=np.zeros(num)

    # initial direction and position
    k=frames.azel_to_ecef(lat, lon, 10e3, azimuth, elevation)
    k0=k
    p=frames.geodetic_to_ITRS(lat, lon, 10e3)
    pe=frames.geodetic_to_ITRS(lat, lon, 10e3)
    p0=frames.geodetic_to_ITRS(lat, lon, 10e3)
    dh=4e3
    vg=1.0

    p_orig=p
    ray_time=0.0
    
    for ai,a in enumerate(alts):
        p=p+k*dh*vg
        p0=p0+k0*dh  
        ray_time+=dh/constants.c
        
        dpx=p+np.array([1.0,0.0,0.0])*dh
        dpy=p+np.array([0.0,1.0,0.0])*dh
        dpz=p+np.array([0.0,0.0,1.0])*dh

        llh=frames.ITRS_to_geodetic(p[0],p[1],p[2])
        llh_1=frames.ITRS_to_geodetic(p0[0],p0[1],p0[2])
        dalt[ai]=llh_1[2]-llh[2]
        
        if llh[2]/1e3 > 1900:
            break
        alts[ai]=llh[2]/1e3
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            ne[ai]=pt.ne*1e6
            f_p=8.98*np.sqrt(ne[ai])
            v_g = np.sqrt(1.0-(f_p/frequency)**2.0)
        else:
            ne[ai]=0.0
            
        llh=frames.ITRS_to_geodetic(dpx[0],dpx[1],dpx[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            dnex[ai]=(ne[ai]-pt.ne*1e6)/dh
        else:
            dnex[ai]=0.0

        llh=frames.ITRS_to_geodetic(dpy[0],dpy[1],dpy[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            dney[ai]=(ne[ai]-pt.ne*1e6)/dh
        else:
            dney[ai]=0.0
            
        llh=frames.ITRS_to_geodetic(dpz[0],dpz[1],dpz[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            dnez[ai]=(ne[ai]-pt.ne*1e6)/dh
        else:
            dnez[ai]=0.0
        grad=np.array([dnex[ai],dney[ai],dnez[ai]])
        px[ai]=p[0]
        py[ai]=p[1]
        pz[ai]=p[2]
        p0x[ai]=p0[0]
        p0y[ai]=p0[1]
        p0z[ai]=p0[2]

        dk[ai]=np.arccos(np.dot(k0,k)/(np.sqrt(np.dot(k0,k0))*np.sqrt(np.dot(k,k))))
        # no bending if gradient too small
        if np.dot(grad,grad) > 100.0:
            grad1=grad/np.sqrt(np.dot(grad,grad))
            
            p2=p+k*dh
            llh=frames.ITRS_to_geodetic(p2[0],p2[1],p2[2])
            pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
            pt.run_iri()
            if pt.ne > 0.0:
                ne2=pt.ne*1e6
            else:
                ne2=0.0
            f0=8.98*np.sqrt(ne[ai])
            n0=np.sqrt(1.0-(f0/frequency)**2.0)
            f1=8.98*np.sqrt(ne2)            
            n1=np.sqrt(1.0-(f1/frequency)**2.0)

            theta0=np.arccos(np.dot(grad,k)/(np.sqrt(np.dot(grad,grad))*np.sqrt(np.dot(k,k))))
            # angle cannot be over 90
            if theta0 > np.pi/2.0:
                theta0=np.pi-theta0
            sin_theta_1=(n0/n1)*np.sin(theta0)
            dtheta[ai]=180.0*np.arcsin(sin_theta_1)/np.pi-180.0*theta0/np.pi
            #print("n0/n1 %1.10f theta0 %1.2f theta1-theta0 %1.10f"%(n0/n1,180.0*theta0/np.pi,dtheta[ai]))
            cos_theta_1=np.sqrt(1.0-sin_theta_1**2.0)
            k_ref=(n0/n1)*k+((n0/n1)*np.cos(theta0)-cos_theta_1)*grad1
            # normalize
            k_ref/np.sqrt(np.dot(k_ref,k_ref))
            k=k_ref
            
            angle=np.arccos(np.dot(grad,k)/np.sqrt(np.dot(grad,grad))*np.sqrt(np.dot(k,k)))

    los_time=np.sqrt(np.dot(p_orig-p,p_orig-p))/constants.c
    excess_ionospheric_delay=ray_time-los_time

    # print("Excess propagation time %1.20f mus"%((1e6*(ray_time-los_time))))
    
    theta=np.arccos(np.dot(k0,k)/(np.sqrt(np.dot(k0,k0))*np.sqrt(np.dot(k,k))))
    
    theta_p=np.arccos(np.dot(p0,p)/(np.sqrt(np.dot(p0,p0))*np.sqrt(np.dot(p,p))))

    llh0=frames.ITRS_to_geodetic(px[ai-2],py[ai-2],pz[ai-2])
    llh1=frames.ITRS_to_geodetic(p0x[ai-2],p0y[ai-2],p0z[ai-2])

    # print("d_coord")
    # print(llh0-llh1)
    
    ret = dict(
        px=px,
        py=py,
        pz=pz,
        p0x=p0x,
        p0y=p0y,
        p0z=p0z,
        ray_bending = dtheta,
        electron_density = ne,
        altitudes = alts,
        altitude_errors = dalt,
        excess_ionospheric_delay = excess_ionospheric_delay,
        total_angle_error = 180.0*theta_p/np.pi,
        p_end = p,
        p0_end = p0,
    )

    return ret


def ray_trace_error(
        time,
        lat,
        lon,
        frequency,
        elevation,
        azimuth,
        ionosphere=False,
        error_std=0.05,
    ):
    '''TODO: Docstring
    '''
    
    if Point is None or gcoord is None:
        raise ImportError('pyglow must be installed to ray trace')

    if not isinstance(time, Time):
        dn = time
    else:
        dn = time.tt.datetime

    num=2000
    alts=np.repeat(1e99,num)
    distance=np.linspace(0,4000,num=num)
    ne=np.zeros(num)
    ne2=np.zeros(num)
    dtheta=np.zeros(num)
    dalt=np.zeros(num)        
    dnex=np.zeros(num)
    dney=np.zeros(num)
    dnez=np.zeros(num)            
    xyz_prev=0.0
    dk=np.zeros(num)        
    px=np.zeros(num)
    py=np.zeros(num)
    pz=np.zeros(num)
    t_vec=np.zeros(num)
    t_i_vec=np.zeros(num)           
    k_vecs=[]
    # initial direction and position
    k=frames.azel_to_ecef(lat, lon, 10e3, az, elevation)
    k0=k
    p=frames.geodetic_to_ITRS(lat, lon, 10e3)
    dh=4e3
    dt=20e-6
    # correlated errors std=1, 100 km correlation length
    scale_length=40.0
    ne_errors_x=np.convolve(np.repeat(1.0/np.sqrt(scale_length),scale_length),np.random.randn(10000))
    ne_errors_y=np.convolve(np.repeat(1.0/np.sqrt(scale_length),scale_length),np.random.randn(10000))
    ne_errors_z=np.convolve(np.repeat(1.0/np.sqrt(scale_length),scale_length),np.random.randn(10000))    
    
    p_orig=p
    ray_time=0.0
    v_c=constants.c
    for ai,a in enumerate(alts):
        # go forward in time
        dhp=v_c*dt
        p=p+k*dhp
        ray_time+=dt
        print(ray_time*1e6)
        t_vec[ai+1]=dt
        k_vecs.append(k)
        
        dpx=p+np.array([1.0,0.0,0.0])*dh
        dpy=p+np.array([0.0,1.0,0.0])*dh
        dpz=p+np.array([0.0,0.0,1.0])*dh

        llh=frames.ITRS_to_geodetic(p[0],p[1],p[2])
        
        if llh[2]/1e3 > 2100:
            break
        alts[ai]=llh[2]/1e3
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            ne[ai]=pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6
            if ionosphere:
                f0=8.98*np.sqrt(ne[ai])            
                f_p=8.98*np.sqrt(ne[ai])
                # update group velocity
                v_c=constants.c*np.sqrt(1.0-(f0/frequency)**2.0)            
        else:
            ne[ai]=0.0
            
        llh=frames.ITRS_to_geodetic(dpx[0],dpx[1],dpx[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            dnex[ai]=(ne[ai]-pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6)/dh            
        else:
            dnex[ai]=0.0

        llh=frames.ITRS_to_geodetic(dpy[0],dpy[1],dpy[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        
        if pt.ne > 0.0:
            dney[ai]=(ne[ai]-pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6)/dh                        
        else:
            dney[ai]=0.0
            
        llh=frames.ITRS_to_geodetic(dpz[0],dpz[1],dpz[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            dnez[ai]=(ne[ai]-pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6)/dh
        else:
            dnez[ai]=0.0
            
        grad=np.array([dnex[ai],dney[ai],dnez[ai]])
        
        px[ai]=p[0]
        py[ai]=p[1]
        pz[ai]=p[2]

        dk[ai]=np.arccos(np.dot(k0,k)/(np.sqrt(np.dot(k0,k0))*np.sqrt(np.dot(k,k))))
        # no bending if gradient too small
        if np.dot(grad,grad) > 100.0 and ionosphere:
            grad1=grad/np.sqrt(np.dot(grad,grad))
            
            p2=p+k*dh
            llh=frames.ITRS_to_geodetic(p2[0],p2[1],p2[2])
            pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
            pt.run_iri()
            if pt.ne > 0.0:
                ne2=pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6
            else:
                ne2=0.0
            f0=8.98*np.sqrt(ne[ai])
            n0=np.sqrt(1.0-(f0/frequency)**2.0)
            f1=8.98*np.sqrt(ne2)            
            n1=np.sqrt(1.0-(f1/frequency)**2.0)

            theta0=np.arccos(np.dot(grad,k)/(np.sqrt(np.dot(grad,grad))*np.sqrt(np.dot(k,k))))
            # angle cannot be over 90
            if theta0 > np.pi/2.0:
                theta0=np.pi-theta0
            sin_theta_1=(n0/n1)*np.sin(theta0)
            dtheta[ai]=180.0*np.arcsin(sin_theta_1)/np.pi-180.0*theta0/np.pi
#            print("n0/n1 %1.10f theta0 %1.2f theta1-theta0 %1.10f"%(n0/n1,180.0*theta0/np.pi,dtheta[ai]))
            cos_theta_1=np.sqrt(1.0-sin_theta_1**2.0)
            k_ref=(n0/n1)*k+((n0/n1)*np.cos(theta0)-cos_theta_1)*grad1
            # normalize
            k_ref/np.sqrt(np.dot(k_ref,k_ref))
            k=k_ref
            
            angle=np.arccos(np.dot(grad,k)/np.sqrt(np.dot(grad,grad))*np.sqrt(np.dot(k,k)))

    return t_vec,px,py,pz,alts,ne,k_vecs



def ionospheric_error(time, elevation=90.0,n_samp=20,frequency=233e6, error_std=0.05):
    '''TODO: Docstring

    # estimate using sampling what the ray-tracing error is
    # return error in microseconds, round-trip time units.
    '''
    prop_errors=np.zeros([n_samp,100])
    prop_error_mean=np.zeros(100)
    prop_error_std=np.zeros(100)        
    prop_alts=np.linspace(0,2000,num=100)

    for i in range(n_samp):
        t_vec_i, px_i, py_i, pz_i, alts_i, ne_i, k_i = ray_trace_error(
            time=time,
            frequency=frequency,
            elevation=elevation,
            ionosphere=True,
            error_std=0.0,
        )
        t_vec_e_i, px_e_i, py_e_i, pz_e_i, alts_e_i, ne_e_i, k_e_i = ray_trace_error(
            time=time,
            frequency=frequency,
            elevation=elevation,
            ionosphere=True,
            error_std=error_std,
        )

        maxi=np.where(alts_i > 2050)[0][0]
        alts_i[0]=0.0

        offsets=np.zeros(maxi)
        for ri in range(maxi):
            # determine what is the additional distance needed to reach target
            p=np.array([px_i[ri],py_i[ri],pz_i[ri]])
            pe=np.array([px_e_i[ri],py_e_i[ri],pz_e_i[ri]])

            offsets[ri] = 1e6*2.0*np.abs(np.dot(p-pe,k_i[ri]/np.sqrt(np.dot(k_i[ri],k_i[ri]))))/constants.c
        pos_err_fun = interpolate.interp1d(alts_i[0:maxi],offsets)
        prop_errors[i,:]=pos_err_fun(prop_alts)

    for ri in range(len(prop_alts)):
        prop_error_mean[ri]=np.mean(prop_errors[:,ri])
        prop_error_std[ri]=np.std(prop_errors[:,ri])

    return prop_error_mean, prop_error_std/np.sqrt(float(n_samp)), prop_alts, ne_i[0:maxi], ne_e_i[0:maxi], alts_i[0:maxi]
    

class IonosphericRayTrace(Errors):
    '''
     

    '''
    VARIABLES = [
        'range', 
        'k',
        't',
    ]

    def __init__(self, station, seed=None, electron_density_std=0.0):
        super().__init__(seed=seed)

        self.electron_density_std = electron_density_std
        self.station = station

    def k(self, data, time, azimuth, elevation):
        raise NotImplementedError('')

    def t(self, data, time, azimuth, elevation):
        raise NotImplementedError('')

    def range(self, data, time, azimuth, elevation):
        raise NotImplementedError('')
        ray_trace_error(
                time,
                self.station.lat,
                self.station.lon,
                self.station.beam.frequency,
                elevation,
                azimuth,
                ionosphere=True,
                error_std=self.electron_density_std,
            )
