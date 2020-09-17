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

from datetime import datetime
import matplotlib.pyplot as plt

import numpy as n
import scipy.integrate as si
import scipy.interpolate as sint
import scipy.constants as c
import coord

from mpl_toolkits.mplot3d import Axes3D


def get_delay(dn=datetime(2016, 3, 23, 00, 00),
              f=233e6,
              lat=e3d._tx[0].lat,
              lon=e3d._tx[0].lon,
              elevation=30.0,
              plot=False):
    raise NotImplementedError()
    np=500
    alts=n.linspace(0,4000,num=np)
    distance=n.linspace(0,4000,num=np)
    ne=n.zeros(np)
    xyz_prev=0.0
    for ai,a in enumerate(alts):
        llh=coord.az_el_r2geodetic(lat, lon, 0, 180.0, elevation, a*1e3)
#        print(llh[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            ne[ai]=pt.ne*1e6
        else:
            ne[ai]=0.0
        xyz=gcoord.lla2ecef(n.array([lat,lon,a]))[0]
        if ai==0:
            distance[ai]=0.0
        else:
            distance[ai]=n.sqrt(n.dot(xyz-xyz_prev,xyz-xyz_prev))+distance[ai-1]
        xyz_prev=xyz
        
    f_p=8.98*n.sqrt(ne)
    v_g = c.c*n.sqrt(1-(f_p/f)**2.0)

    dt2=si.simps(1.0 - 1.0/(n.sqrt(1-(f_p/f)**2.0)),distance*1e3)/c.c

    if plot:
        print("ionospheric delay (s)")
        print(dt2)
        plt.plot(ne,distance)
        plt.ylabel("Distance (km)")
        plt.xlabel("$N_e$ ($m^{-3}$)")
        plt.show()
    return(dt2,ne,distance)


def ray_trace(dn=datetime(2016, 6, 21, 12, 00),
              f=233e6,
              lat=e3d._tx[0].lat,
              lon=e3d._tx[0].lon,
              elevation=30.0,
              az=180.0,
              fpref="",
              plot=False):

    raise NotImplementedError()
    np=1000
    alts=n.linspace(0,4000,num=np)
    distance=n.linspace(0,4000,num=np)
    ne=n.zeros(np)
    ne2=n.zeros(np)    
    dnex=n.zeros(np)
    dtheta=n.zeros(np)
    dalt=n.zeros(np)    
    dney=n.zeros(np)
    dnez=n.zeros(np)            
    xyz_prev=0.0
    px=n.zeros(np)
    dk=n.zeros(np)    
    py=n.zeros(np)
    pz=n.zeros(np)       
    p0x=n.zeros(np)
    p0y=n.zeros(np)
    p0z=n.zeros(np)
    
    # initial direction and position
    k=coord.azel_ecef(lat, lon, 10e3, az, elevation)
    k0=k
    p=coord.geodetic2ecef(lat, lon, 10e3)
    pe=coord.geodetic2ecef(lat, lon, 10e3)    
    p0=coord.geodetic2ecef(lat, lon, 10e3)    
    dh=4e3
    vg=1.0

    p_orig=p
    ray_time=0.0
    
    for ai,a in enumerate(alts):
        p=p+k*dh*vg
        p0=p0+k0*dh  
        ray_time+=dh/c.c
        
        dpx=p+n.array([1.0,0.0,0.0])*dh
        dpy=p+n.array([0.0,1.0,0.0])*dh
        dpz=p+n.array([0.0,0.0,1.0])*dh

        llh=coord.ecef2geodetic(p[0],p[1],p[2])
        llh_1=coord.ecef2geodetic(p0[0],p0[1],p0[2])
        dalt[ai]=llh_1[2]-llh[2]
        
        if llh[2]/1e3 > 1900:
            break
        alts[ai]=llh[2]/1e3
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            ne[ai]=pt.ne*1e6
            f_p=8.98*n.sqrt(ne[ai])
            v_g = n.sqrt(1.0-(f_p/f)**2.0)
        else:
            ne[ai]=0.0
            
        llh=coord.ecef2geodetic(dpx[0],dpx[1],dpx[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            dnex[ai]=(ne[ai]-pt.ne*1e6)/dh
        else:
            dnex[ai]=0.0

        llh=coord.ecef2geodetic(dpy[0],dpy[1],dpy[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            dney[ai]=(ne[ai]-pt.ne*1e6)/dh
        else:
            dney[ai]=0.0
            
        llh=coord.ecef2geodetic(dpz[0],dpz[1],dpz[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        if pt.ne > 0.0:
            dnez[ai]=(ne[ai]-pt.ne*1e6)/dh
        else:
            dnez[ai]=0.0
        grad=n.array([dnex[ai],dney[ai],dnez[ai]])
        px[ai]=p[0]
        py[ai]=p[1]
        pz[ai]=p[2]
        p0x[ai]=p0[0]
        p0y[ai]=p0[1]
        p0z[ai]=p0[2]
#        print(ai)
        dk[ai]=n.arccos(n.dot(k0,k)/(n.sqrt(n.dot(k0,k0))*n.sqrt(n.dot(k,k))))
        # no bending if gradient too small
        if n.dot(grad,grad) > 100.0:
            grad1=grad/n.sqrt(n.dot(grad,grad))
            
            p2=p+k*dh
            llh=coord.ecef2geodetic(p2[0],p2[1],p2[2])
            pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
            pt.run_iri()
            if pt.ne > 0.0:
                ne2=pt.ne*1e6
            else:
                ne2=0.0
            f0=8.98*n.sqrt(ne[ai])
            n0=n.sqrt(1.0-(f0/f)**2.0)
            f1=8.98*n.sqrt(ne2)            
            n1=n.sqrt(1.0-(f1/f)**2.0)

            theta0=n.arccos(n.dot(grad,k)/(n.sqrt(n.dot(grad,grad))*n.sqrt(n.dot(k,k))))
            # angle cannot be over 90
            if theta0 > n.pi/2.0:
                theta0=n.pi-theta0
            sin_theta_1=(n0/n1)*n.sin(theta0)
            dtheta[ai]=180.0*n.arcsin(sin_theta_1)/n.pi-180.0*theta0/n.pi
#            print("n0/n1 %1.10f theta0 %1.2f theta1-theta0 %1.10f"%(n0/n1,180.0*theta0/n.pi,dtheta[ai]))
            cos_theta_1=n.sqrt(1.0-sin_theta_1**2.0)
            k_ref=(n0/n1)*k+((n0/n1)*n.cos(theta0)-cos_theta_1)*grad1
            # normalize
            k_ref/n.sqrt(n.dot(k_ref,k_ref))
            k=k_ref
            
            angle=n.arccos(n.dot(grad,k)/n.sqrt(n.dot(grad,grad))*n.sqrt(n.dot(k,k)))

    los_time=n.sqrt(n.dot(p_orig-p,p_orig-p))/c.c
    excess_ionospheric_delay=ray_time-los_time
    print("Excess propagation time %1.20f mus"%((1e6*(ray_time-los_time))))
    
    theta=n.arccos(n.dot(k0,k)/(n.sqrt(n.dot(k0,k0))*n.sqrt(n.dot(k,k))))
    
    theta_p=n.arccos(n.dot(p0,p)/(n.sqrt(n.dot(p0,p0))*n.sqrt(n.dot(p,p))))

    llh0=coord.ecef2geodetic(px[ai-2],py[ai-2],pz[ai-2])
    llh1=coord.ecef2geodetic(p0x[ai-2],p0y[ai-2],p0z[ai-2])
    print("d_coord")
    print(llh0-llh1)
    if plot:
        print(p0-p)
        print(180.0*theta_p/n.pi)
        fig=plt.figure(figsize=(14,8))
        plt.clf()
        plt.subplot(131)
        plt.title("Elevation=%1.0f"%(elevation))
        plt.plot(n.sqrt((p0x-px)**2.0 + (p0y-py)**2.0 +(p0z-pz)**2.0),alts,label="Total error")
        plt.plot(dalt,alts,label="Altitude error")
        plt.ylim([0,1900])
#        plt.xlim([-50,800.0])
        plt.grid()
        plt.legend()
        plt.xlabel("Position error (m)")
        plt.ylabel("Altitude km")

        plt.subplot(132)
        plt.plot(dtheta*1e6,alts)
#        plt.plot(1e6*180.0*dk/n.pi,alts)                
        plt.xlabel("Ray-bending ($\mu$deg/km)")
        plt.ylabel("Altitude km")
        plt.title("Total error=%1.2g (deg)"%(180.0*theta_p/n.pi))
        plt.ylim([0,1900])        
        plt.subplot(133)
        plt.plot(ne,alts)
        plt.xlabel("$N_{\mathrm{e}}$ ($\mathrm{m}^{-3}$)")
        plt.ylabel("Altitude km")
        plt.ylim([0,1900])        
        #    ax.plot(px,py,pz)
        plt.tight_layout()
        plt.savefig("ref-%s-%d-%d.png"%(fpref,f/1e6,elevation))
        plt.close()
    
    return(p0,p,180.0*theta_p/n.pi,excess_ionospheric_delay)


def ray_trace_error(dn=datetime(2016, 6, 21, 12, 00),
                    f=233e6,
                    lat=e3d._tx[0].lat,
                    lon=e3d._tx[0].lon,
                    elevation=30.0,
                    az=180.0,
                    fpref="",
                    ionosphere=False,
                    error_std=0.05,
                    plot=False):
    raise NotImplementedError()
    np=2000
    alts=n.repeat(1e99,np)
    distance=n.linspace(0,4000,num=np)
    ne=n.zeros(np)
    ne2=n.zeros(np)
    dtheta=n.zeros(np)
    dalt=n.zeros(np)        
    dnex=n.zeros(np)
    dney=n.zeros(np)
    dnez=n.zeros(np)            
    xyz_prev=0.0
    dk=n.zeros(np)        
    px=n.zeros(np)
    py=n.zeros(np)
    pz=n.zeros(np)
    t_vec=n.zeros(np)
    t_i_vec=n.zeros(np)           
    k_vecs=[]
    # initial direction and position
    k=coord.azel_ecef(lat, lon, 10e3, az, elevation)
    k0=k
    p=coord.geodetic2ecef(lat, lon, 10e3)
    dh=4e3
    dt=20e-6
    # correlated errors std=1, 100 km correlation length
    scale_length=40.0
    ne_errors_x=n.convolve(n.repeat(1.0/n.sqrt(scale_length),scale_length),n.random.randn(10000))
    ne_errors_y=n.convolve(n.repeat(1.0/n.sqrt(scale_length),scale_length),n.random.randn(10000))
    ne_errors_z=n.convolve(n.repeat(1.0/n.sqrt(scale_length),scale_length),n.random.randn(10000))    
    
    p_orig=p
    ray_time=0.0
    v_c=c.c
    for ai,a in enumerate(alts):
        # go forward in time
        dhp=v_c*dt
        p=p+k*dhp
        ray_time+=dt
        print(ray_time*1e6)
        t_vec[ai+1]=dt
        k_vecs.append(k)
        
        dpx=p+n.array([1.0,0.0,0.0])*dh
        dpy=p+n.array([0.0,1.0,0.0])*dh
        dpz=p+n.array([0.0,0.0,1.0])*dh

        llh=coord.ecef2geodetic(p[0],p[1],p[2])
        
        if llh[2]/1e3 > 2100:
            break
        alts[ai]=llh[2]/1e3
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            ne[ai]=pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6
            if ionosphere:
                f0=8.98*n.sqrt(ne[ai])            
                f_p=8.98*n.sqrt(ne[ai])
                # update group velocity
                v_c=c.c*n.sqrt(1.0-(f0/f)**2.0)            
        else:
            ne[ai]=0.0
            
        llh=coord.ecef2geodetic(dpx[0],dpx[1],dpx[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            dnex[ai]=(ne[ai]-pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6)/dh            
        else:
            dnex[ai]=0.0

        llh=coord.ecef2geodetic(dpy[0],dpy[1],dpy[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()
        
        if pt.ne > 0.0:
            dney[ai]=(ne[ai]-pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6)/dh                        
        else:
            dney[ai]=0.0
            
        llh=coord.ecef2geodetic(dpz[0],dpz[1],dpz[2])
        pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
        pt.run_iri()

        if pt.ne > 0.0:
            dnez[ai]=(ne[ai]-pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6)/dh
        else:
            dnez[ai]=0.0
            
        grad=n.array([dnex[ai],dney[ai],dnez[ai]])
        
        px[ai]=p[0]
        py[ai]=p[1]
        pz[ai]=p[2]

        dk[ai]=n.arccos(n.dot(k0,k)/(n.sqrt(n.dot(k0,k0))*n.sqrt(n.dot(k,k))))
        # no bending if gradient too small
        if n.dot(grad,grad) > 100.0 and ionosphere:
            grad1=grad/n.sqrt(n.dot(grad,grad))
            
            p2=p+k*dh
            llh=coord.ecef2geodetic(p2[0],p2[1],p2[2])
            pt = Point(dn, llh[0], llh[1], llh[2]/1e3)
            pt.run_iri()
            if pt.ne > 0.0:
                ne2=pt.ne*(1.0+error_std*ne_errors_x[ai])*1e6
            else:
                ne2=0.0
            f0=8.98*n.sqrt(ne[ai])
            n0=n.sqrt(1.0-(f0/f)**2.0)
            f1=8.98*n.sqrt(ne2)            
            n1=n.sqrt(1.0-(f1/f)**2.0)

            theta0=n.arccos(n.dot(grad,k)/(n.sqrt(n.dot(grad,grad))*n.sqrt(n.dot(k,k))))
            # angle cannot be over 90
            if theta0 > n.pi/2.0:
                theta0=n.pi-theta0
            sin_theta_1=(n0/n1)*n.sin(theta0)
            dtheta[ai]=180.0*n.arcsin(sin_theta_1)/n.pi-180.0*theta0/n.pi
#            print("n0/n1 %1.10f theta0 %1.2f theta1-theta0 %1.10f"%(n0/n1,180.0*theta0/n.pi,dtheta[ai]))
            cos_theta_1=n.sqrt(1.0-sin_theta_1**2.0)
            k_ref=(n0/n1)*k+((n0/n1)*n.cos(theta0)-cos_theta_1)*grad1
            # normalize
            k_ref/n.sqrt(n.dot(k_ref,k_ref))
            k=k_ref
            
            angle=n.arccos(n.dot(grad,k)/n.sqrt(n.dot(grad,grad))*n.sqrt(n.dot(k,k)))

    return(t_vec,px,py,pz,alts,ne,k_vecs)

def ref_delay():
    raise NotImplementedError()
    tau,ne_d,distance=get_delay(dn=datetime(2015, 6, 21, 12, 00),elevation=90.0)
    tau,ne_n,distance=get_delay(dn=datetime(2015, 6, 21, 0, 0),elevation=90.0)    
    plt.plot(ne_d,distance,label="Noon")
    plt.plot(ne_n,distance,label="Midnight")    
    plt.ylim([0,2000])
    plt.legend()
    plt.ylabel("Altitude (km)")
    plt.xlabel("$\mathrm{N_e}$ ($\mathrm{m}^{-3}$)")

    plt.savefig("profile.png")
    plt.show()

    f=233e6
    taus=[]
    hours=[]
    for i in n.arange(0,24):
        print(i)
        p0,p,angle,iono_del=ray_trace(dn=datetime(2015, 12, 16, int(i) , 00),f=f,elevation=90.0,plot=False)
#        tau,ne,distance,iono_del=get_delay(dn=datetime(2015, 12, 16,int(i) , 00),elevation=90.0)
        hours.append(i)
        taus.append(iono_del)
        print(tau)
    plt.plot(hours,n.array(taus)*1e6,label="Winter soltice el=90$^{\circ}$",color="blue")

    taus=[]
    hours=[]
    for i in n.arange(0,24):
        print(i)
        p0,p,angle,iono_del=ray_trace(dn=datetime(2015, 6, 21, int(i) , 00),f=f,elevation=90.0,plot=False)                
#        tau,ne,distance,iono_del=get_delay(dn=datetime(2015, 6, 21,int(i) , 00),elevation=90.0)
        hours.append(i)
        taus.append(iono_del)
        print(tau)
    plt.plot(hours,n.array(taus)*1e6,label="Summer soltice el=90$^{\circ}$",color="red")

    taus=[]
    hours=[]
    for i in n.arange(0,24):
        print(i)
        p0,p,angle,iono_del=ray_trace(dn=datetime(2015, 12, 16, int(i) , 00),f=f,elevation=30.0,plot=False)                        
#        tau,ne,distance,iono_del=get_delay(dn=datetime(2015, 12, 16,int(i) , 00))
        hours.append(i)
        taus.append(iono_del)
        print(tau)
    plt.plot(hours,n.array(taus)*1e6,label="Winter soltice el=30$^{\circ}$",ls="--",color="blue")

    taus=[]
    hours=[]
    for i in n.arange(0,24):
        print(i)
        p0,p,angle,iono_del=ray_trace(dn=datetime(2015, 6, 21, int(i) , 00),f=f,elevation=30.0,plot=False)                                
#        tau,ne,distance,iono_del=get_delay(dn=datetime(2015, 6, 21,int(i) , 00))
        hours.append(i)
        taus.append(iono_del)
        print(tau)
    plt.plot(hours,n.array(taus)*1e6,label="Summer soltice el=30$^{\circ}$",ls="--",color="red")
    
    plt.ylabel("Ionospheric delay ($\mu \mathrm{s}$)")
    plt.xlabel("Time of day (UTC hour)")
    plt.legend()
    plt.savefig("vert_delay-%1.2f.png"%(f/1e6))
    plt.show()

# estimate using sampling what the ray-tracing error is
# return error in microseconds, round-trip time units.
def ionospheric_error(elevation=90.0,n_samp=20,f=233e6,error_std=0.05,dn=datetime(2015, 6, 16, 12 , 00)):
    raise NotImplementedError()
    prop_errors=n.zeros([n_samp,100])
    prop_error_mean=n.zeros(100)
    prop_error_std=n.zeros(100)        
    prop_alts=n.linspace(0,2000,num=100)
    for i in range(n_samp):
        t_vec_i,px_i,py_i,pz_i,alts_i,ne_i,k_i=ray_trace_error(dn=dn,f=f,elevation=elevation,ionosphere=True,error_std=0.0)
        t_vec_e_i,px_e_i,py_e_i,pz_e_i,alts_e_i,ne_e_i,k_e_i=ray_trace_error(dn=dn,f=f,elevation=elevation,ionosphere=True,error_std=error_std)
        maxi=n.where(alts_i > 2050)[0][0]
        alts_i[0]=0.0

        offsets=n.zeros(maxi)
        for ri in range(maxi):
            # determine what is the additional distance needed to reach target
            p=n.array([px_i[ri],py_i[ri],pz_i[ri]])
            pe=n.array([px_e_i[ri],py_e_i[ri],pz_e_i[ri]])
            offsets[ri]=1e6*2.0*n.abs(n.dot(p-pe,k_i[ri]/n.sqrt(n.dot(k_i[ri],k_i[ri]))))/c.c
        pos_err_fun=sint.interp1d(alts_i[0:maxi],offsets)
        prop_errors[i,:]=pos_err_fun(prop_alts)
    for ri in range(len(prop_alts)):
        prop_error_mean[ri]=n.mean(prop_errors[:,ri])
        prop_error_std[ri]=n.std(prop_errors[:,ri])
    return(prop_error_mean,prop_error_std/n.sqrt(float(n_samp)),prop_alts,ne_i[0:maxi],ne_e_i[0:maxi],alts_i[0:maxi])
    
    
if __name__ == "__main__":
    ionospheric_error()
    exit()
    
    nang=10
    angles=n.linspace(30,90,num=nang)
    dang=n.zeros(nang)
    iono_del_w=n.zeros(nang)
    iono_del_s=n.zeros(nang)        
    f=49.92e6
    for ai,a in enumerate(angles):
        p0,p,angle,iono_del=ray_trace(dn=datetime(2015, 12, 16, 0, 00),f=f,elevation=a,plot=True,fpref="winter")
        dang[ai]=angle
        iono_del_w[ai]=iono_del
    plt.clf()
    plt.semilogy(angles,dang,label="Midnight, Winter soltice")
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Ionospheric ray-bending (deg)")
  #  plt.tight_layout()
 #   plt.grid()
#    plt.savefig("angle_error-%1.0f.png"%(f/1e6))

    angles=n.linspace(30,90,num=10)
    dang=n.zeros(10)
    for ai,a in enumerate(angles):
        p0,p,angle,iono_del=ray_trace(dn=datetime(2015, 6, 21, 12, 00),f=f,elevation=a,plot=True,fpref="summer")
        dang[ai]=angle
        iono_del_s[ai]=iono_del
    plt.semilogy(angles,dang,label="Noon, Summer soltice")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("angle_error-%1.0f.png"%(f/1e6))

    plt.clf()
    plt.plot(angles,1e6*iono_del_w,label="Winter")
    plt.plot(angles,1e6*iono_del_s,label="Summer")
    plt.legend()
    plt.grid()
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Ionospheric excess delay ($\mu$s)")
    plt.savefig("iono_del-%1.0f.png"%(f/1e6))
    
    #    plt.show()
#    ref_delay()
    