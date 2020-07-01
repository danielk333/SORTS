#!/usr/bin/env python

'''A collection of functions that return common instances of the :class:`~antenna.BeamPattern` class.

Contains for example:
 * Uniformly filled circular aperture of radius a
 * Cassegrain antenna with radius a0 and subreflector radius a1
 * Planar gaussian illuminated aperture (approximates a phased array)

Reference:
https://www.cv.nrao.edu/course/astr534/2DApertures.html
'''
import os
#import pdb

import numpy as np
import scipy.constants as c
import scipy.special as s
import scipy.interpolate as sio
import h5py

import coord
import antenna
import dpt_tools as dpt



def elliptic_airy(k_in, beam):
    '''# TODO: Descriptive doc string.

     a = radius
     f = frequency
     I_0 = gain at center
    '''
    phi = np.pi*coord.angle_deg(beam.on_axis, beam.plane_normal)/180.0
    #F[ g(ct) ] = G(f/c)/|c|
    theta = np.pi*coord.angle_deg(beam.on_axis,k_in)/180.0
    lam=c.c/beam.f
    
    k=2.0*np.pi/lam
    
    f_circ = lambda r,phi: beam.I_0*((2.0*s.jn(1,k* beam.a*np.sin(theta))/(k*beam.a*np.sin(theta))))**2.0

    return( )


def airy(k_in, beam):
    '''# TODO: Descriptive doc string.

     a = radius
     f = frequency
     I_0 = gain at center
    '''
    theta = np.pi*coord.angle_deg(beam.on_axis,k_in)/180.0
    lam=c.c/beam.f
    
    k=2.0*np.pi/lam
    
    return(beam.I_0*((2.0*s.jn(1,k*beam.a*np.sin(theta))/(k*beam.a*np.sin(theta))))**2.0)


def cassegrain(k_in, beam):
    '''# TODO: Descriptive doc string.

    A better model of the EISCAT UHF antenna
    '''
    theta = np.pi*coord.angle_deg(beam.on_axis,k_in)/180.0

    lam=c.c/beam.f
    k=2.0*np.pi/lam
    
    A=(beam.I_0*((lam/(np.pi*np.sin(theta)))**2.0))/((beam.a0**2.0-beam.a1**2.0)**2.0)
    B=(beam.a0*s.jn(1,beam.a0*np.pi*np.sin(theta)/lam)-beam.a1*s.jn(1,beam.a1*np.pi*np.sin(theta)/lam))**2.0
    A0=(beam.I_0*((lam/(np.pi*np.sin(1e-6)))**2.0))/((beam.a0**2.0-beam.a1**2.0)**2.0)
    B0=(beam.a0*s.jn(1,beam.a0*np.pi*np.sin(1e-6)/lam)-beam.a1*s.jn(1,beam.a1*np.pi*np.sin(1e-6)/lam))**2.0
    const=beam.I_0/(A0*B0)
    return(A*B*const)



def uhf_meas(k_in,beam):
    '''Measured UHF beam pattern

    '''
    theta = coord.angle_deg(beam.on_axis,k_in)
    # scale beam width by frequency
    sf=beam.f/930e6
    
    return(beam.I_0*beam.gf(sf*np.abs(theta)))


def planar(k_in,beam):
    '''Gaussian tapered planar array

    '''
    
    if np.abs(1-np.dot(beam.on_axis,beam.plane_normal)) < 1e-6:
        rd=np.random.randn(3)
        rd=rd/np.sqrt(np.dot(rd,rd))
        ct=np.cross(beam.on_axis,rd)
    else:
        ct=np.cross(beam.on_axis,beam.plane_normal)
        
    ct=ct/np.sqrt(np.dot(ct,ct))
    ht=np.cross(beam.plane_normal,ct)
    ht=ht/np.sqrt(np.dot(ht,ht))
    angle=coord.angle_deg(beam.on_axis,ht)

    ot=np.cross(beam.on_axis,ct)
    ot=ot/np.sqrt(np.dot(ot,ot))

    beam.I_1=np.sin(np.pi*angle/180.0)*beam.I_0
    beam.a0p=np.sin(np.pi*angle/180.0)*beam.a0

    beam.ct=ct
    beam.ht=ht
    beam.ot=ot
    beam.angle=angle

    beam.sigma1=0.7*beam.a0p/beam.lam
    beam.sigma2=0.7*beam.a0/beam.lam

    k0=k_in/np.sqrt(np.dot(k_in,k_in))
    
    A=np.dot(k0,beam.on_axis)
    kda=A*beam.on_axis
    l1=np.dot(k0,beam.ct)
    kdc=l1*beam.ct
    m1=np.dot(k0,beam.ot)
    kdo=m1*beam.ot
    
    l2=l1*l1
    m2=m1*m1
    return beam.I_1*np.exp(-np.pi*m2*2.0*np.pi*beam.sigma1**2.0)*np.exp(-np.pi*l2*2.0*np.pi*beam.sigma2**2.0)


def elliptic(k_in,beam):
    '''# TODO: Description.


     TDB: sqrt(u**2 + c**2 v**2)


     http://www.iue.tuwien.ac.at/phd/minixhofer/node59.html
     https://en.wikipedia.org/wiki/Fraunhofer_diffraction_equation


     x=n.linspace(-2,2,num=1024)
     xx,yy=n.meshgrid(x,x)

     A=n.zeros([1024,1024])
     A[xx**2.0/0.25**2 + yy**2.0/0.0625**2.0 < 1.0]=1.0

     plt.pcolormesh(10.0*n.log10(n.fft.fftshift(n.abs(B))))
     plt.colorbar()
     plt.axis("equal")
     plt.show()


    Variable substitution
    '''
    if np.abs(1-np.dot(beam.on_axis,beam.plane_normal)) < 1e-6:
        rd=np.random.randn(3)
        rd=rd/np.sqrt(np.dot(rd,rd))
        ct=np.cross(beam.on_axis,rd)
    else:
        ct=np.cross(beam.on_axis,beam.plane_normal)
    
    ct=ct/np.sqrt(np.dot(ct,ct))
    ht=np.cross(beam.plane_normal,ct)
    ht=ht/np.sqrt(np.dot(ht,ht))
    angle=coord.angle_deg(beam.on_axis,ht)
    ot=np.cross(beam.on_axis,ct)
    ot=ot/np.sqrt(np.dot(ot,ot))

    beam.I_1=np.sin(np.pi*angle/180.0)*beam.I_0
    beam.a0p=np.sin(np.pi*angle/180.0)*beam.a0
    
    beam.ct=ct
    beam.ht=ht
    beam.ot=ot
    beam.angle=angle
    
    beam.sigma1=0.7*beam.a0p/beam.lam
    beam.sigma2=0.7*beam.a0/beam.lam
    
    k0=k_in/np.sqrt(np.dot(k_in,k_in))
    
    A=np.dot(k0,beam.on_axis)
    kda=A*beam.on_axis
    l1=np.dot(k0,beam.ct)
    kdc=l1*beam.ct
    m1=np.dot(k0,beam.ot)
    kdo=m1*beam.ot
    
    l2=l1*l1
    m2=m1*m1
    return beam.I_1*np.exp(-np.pi*m2*2.0*np.pi*beam.sigma1**2.0)*np.exp(-np.pi*l2*2.0*np.pi*beam.sigma2**2.0)


def plane_wave(k,r,p):
    '''The complex plane wave function.

    :param numpy.ndarray k: Wave-vector (wave propagation direction)
    :param numpy.ndarray r: Spatial location (Antenna position in space)
    :param numpy.ndarray p: Beam-forming direction (antenna array "pointing" direction)
    '''
    return np.exp(1j*np.pi*2.0*np.dot(k-p,r))

def array(k_in,beam):
    '''# TODO: Description.

    '''
    k = k_in/np.linalg.norm(k_in)
    p = beam.on_axis
    G = np.exp(1j)*0.0

    #r in meters, divide by lambda
    for r in beam.antennas:
        G += plane_wave(k,r/(c.c/beam.f),p)

    #Ugly fix: multiply gain by k_z to emulate beam steering loss as a function of elevation
    #should be antenna element gain pattern of k...
    return np.abs(G.conj()*G*beam.I_scale)*p[2]

def array_beam(az0, el0, I_0, f, antennas):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(array, az0, el0, I_0, f, beam_name='Array')
    beam.antennas = antennas
    beam.I_scale = I_0/(antennas.shape[0]**2.0)
    return beam

def e3d_subarray(f):
    '''# TODO: Description.

    '''
    l0 = c.c/f;

    dx = 1.0/np.sqrt(3);
    dy = 0.5;

    xall = []
    yall = []

    x0 = np.array([np.arange(-2.5,-5.5,-.5).tolist() + np.arange(-4.5,-2.0,.5).tolist()])[0]*dx
    y0 = np.arange(-5,6,1)*dy

    for iy in range(11):
        nx = 11-np.abs(iy-5)
        x_now = x0[iy]+np.array(range(nx))*dx
        y_now = y0[iy]+np.array([0.0]*(nx))
        xall += x_now.tolist()
        yall += y_now.tolist()

    x = l0*np.array(xall);
    y = l0*np.array(yall);
    z = x*0.0;

    return x,y,z

def e3d_array(f,fname='data/e3d_array.txt'):
    '''# TODO: Description.

    '''
    dat = []
    with open(fname,'r') as file:
        for line in file:
            dat.append( list(map(lambda x: float(x),line.split() )) )
    dat = np.array(dat)

    sx,sy,sz = e3d_subarray(f)

    antennas = []
    for i in range(dat.shape[0]):
        for j in range(len(sx)):
            antennas.append([ sx[j] + dat[i,0],sy[j] + dat[i,1],sz[j] ])
    return np.array(antennas)

def e3d_array_stage1(f,fname='data/e3d_array.txt',opt='dense'):
    '''# TODO: Description.

    '''
    dat = []
    with open(fname,'r') as file:
        for line in file:
            dat.append( list(map(lambda x: float(x),line.split() ) ) )
    dat = np.array(dat)

    if opt=='dense':
        dat = dat[ ( np.sum(dat**2.0,axis=1) < 27.0**2.0 ) ,: ]
    else:
        dat = dat[ \
        np.logical_or( \
            np.logical_or(\
                np.logical_and( np.sum(dat**2,axis=1) < 10**2 , np.sum(dat**2,axis=1) > 7**2 ), \
                np.logical_and( np.sum(dat**2,axis=1) < 22**2 , np.sum(dat**2,axis=1) > 17**2 )),  \
            np.logical_and( np.sum(dat**2,axis=1) < 36**2 , np.sum(dat**2,axis=1) > 30**2 ) \
        ),: ]

    sx,sy,sz = e3d_subarray(f)

    antennas = []
    for i in range(dat.shape[0]):
        for j in range(len(sx)):
            antennas.append([ sx[j] + dat[i,0],sy[j] + dat[i,1],sz[j] ])
    return np.array(antennas)


def e3d_module_beam(az0=0, el0=90.0, I_0=10**2.2):
    '''# TODO: Description.

    '''
    sx,sy,sz = e3d_subarray(233e6)
    antennas = []
    for j in range(len(sx)):
        antennas.append([ sx[j] ,sy[j] ,sz[j] ])
    antennas = np.array(antennas)

    beam = array_beam(az0, el0, I_0=I_0, f=233e6, antennas = antennas)
    beam.beam_name = 'E3D module'
    beam.antennas_n = antennas.shape[0]
    return beam

def e3d_array_beam(az0=0, el0=90.0, I_0=10**4.5, fname='data/e3d_array.txt'):
    '''# TODO: Description.
    
    45dB peak according to e3d specs: Technical specification and requirements for antenna unit
    '''
    
    antennas = e3d_array(233e6, fname)
    beam = array_beam(az0, el0, I_0=I_0, f=233e6, antennas = antennas)
    beam.beam_name = 'E3D stage 2'
    beam.antennas_n = antennas.shape[0]
    return beam

def e3d_array_beam_stage1(az0=0, el0=90.0, I_0=10**4.2, fname='data/e3d_array.txt', opt='dense'):
    '''# TODO: Description.
    
    45dB-3dB=42dB peak according to e3d specs: Technical specification and requirements for antenna unit
    '''

    antennas = e3d_array_stage1(233e6, fname, opt=opt)
    beam = array_beam(az0, el0, I_0=I_0, f=233e6, antennas = antennas)
    beam.beam_name = 'E3D stage 1 ' + opt
    beam.antennas_n = antennas.shape[0]
    return beam


def _generate_interpolation_beam_data(fname, beam, res = 1000):
    '''Create a grid of wave vector projections and 2d interpolate the gain function.
    '''
    beam.point(az0=0.0, el0=90.0)

    save_raw = fname.split('.')
    save_raw[-2] += '_data'
    save_raw = '.'.join(save_raw)

    if not os.path.isfile(save_raw):

        kx=np.linspace(-1.0, 1.0, num=res)
        ky=np.linspace(-1.0, 1.0, num=res)
        
        S=np.zeros((res,res))
        Xmat=np.zeros((res,res))
        Ymat=np.zeros((res,res))

        cnt = 0
        tot = res**2

        for i,x in enumerate(kx):
            for j,y in enumerate(ky):
                
                if cnt % int(tot/1000) == 0:
                    print('{}/{} Gain done'.format(cnt, tot))
                cnt += 1

                z2 = x**2 + y**2
                if z2 < 1.0:
                    k=np.array([x, y, np.sqrt(1.0 - z2)])
                    S[i,j]=beam.gain(k)
                else:
                    S[i,j] = 0;
                Xmat[i,j]=x
                Ymat[i,j]=y
        np.save(save_raw, S)

    S = np.load(save_raw)

    f = sio.interp2d(kx, ky, S.T, kind='linear')
    np.save(fname, f)


def _rot2d(theta):
    M_rot = np.empty((2,2), dtype=np.float)
    M_rot[0,0] = np.cos(theta)
    M_rot[1,0] = np.sin(theta)
    M_rot[0,1] = -np.sin(theta)
    M_rot[1,1] = np.cos(theta)
    return M_rot

def _scale2d(x,y):
    M_rot = np.zeros((2,2), dtype=np.float)
    M_rot[0,0] = x
    M_rot[1,1] = y
    return M_rot

def _plane_scaling_matrix(vec, factor):
    theta = -np.arctan2(vec[1], vec[0])
    
    M_rot = _rot2d(theta)
    M_scale = _scale2d(factor, 1)
    M_rot_inv = _rot2d(-theta)

    M = M_rot_inv.dot(M_scale.dot(M_rot))

    return M


def interpolated_beam(k_in, beam):
    '''Assume that the interpolated grid at zenith is merely shifted to the pointing direction and scaled by the sine of the elevation angle. 
    '''
    k = k_in/np.linalg.norm(k_in)

    M = _plane_scaling_matrix(beam.on_axis[:2], beam.on_axis[2])
    
    k_trans = np.empty((3,), dtype = np.float)
    k_trans[:2] = M.dot(k[:2] - beam.on_axis[:2])
    k_trans[2] = k[2]

    interp_gain = beam.interp_f(k_trans[0], k_trans[1])[0]
    if interp_gain < 0:
        interp_gain = 0.0
    return interp_gain*beam.I_0*beam.on_axis[2]


def e3d_array_beam_stage1_dense_interp(az0=0, el0=90.0, I_0=10**4.2, fname='data/inerp_e3d_stage1_dense.npy', res = 400):
    if not os.path.isfile(fname):
        _beam = e3d_array_beam_stage1(az0=0, el0=90.0, I_0 = 1.0)
        _generate_interpolation_beam_data(fname, _beam, res = res)
        del _beam

    f_obj = np.load(fname)
    f = f_obj.item()

    beam = antenna.BeamPattern(interpolated_beam, az0, el0, I_0, f, beam_name='E3D stage 1 dense -interpolated-')
    beam.interp_f = f

    return beam

def e3d_array_beam_interp(az0=0, el0=90.0, I_0=10**4.2, fname='data/inerp_e3d.npy', res = 400):
    if not os.path.isfile(fname):
        _beam = e3d_array_beam(az0=0, el0=90.0, I_0 = 1.0)
        _generate_interpolation_beam_data(fname, _beam, res = res)
        del _beam

    f_obj = np.load(fname)
    f = f_obj.item()

    beam = antenna.BeamPattern(interpolated_beam, az0, el0, I_0, f, beam_name='E3D stage 2 -interpolated-')
    beam.interp_f = f

    return beam
    


def airy_beam(az0, el0, I_0, f, a):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(airy, az0, el0, I_0, f, beam_name='Airy')
    beam.a = a
    return beam


def uhf_beam(az0, el0, I_0, f, beam_name='UHF Measured beam'):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(uhf_meas, az0, el0, I_0, f, beam_name=beam_name)

    bmod=np.genfromtxt("data/bp.txt")
    angle=bmod[:,0]
    gain=10**(bmod[:,1]/10.0)
    gf=sio.interp1d(np.abs(angle),gain)
    
    beam.gf = gf
    return beam


def cassegrain_beam(az0, el0, I_0, f, a0, a1, beam_name="Cassegrain"):
    '''# TODO: Description.

    az and el of on-axis
    lat and lon of location
    I_0 gain on-axis
    a0 diameter of main reflector
    a1 diameter of the subreflector
    '''
    beam = antenna.BeamPattern(cassegrain, az0, el0, I_0, f, beam_name=beam_name)
    beam.a0 = a0
    beam.a1 = a1
    return beam

def planar_beam(az0, el0, I_0, f, a0, az1, el1):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(planar, az0, el0, I_0, f, beam_name='Planar')
    beam.a0 = a0
    beam.plane_normal=coord.azel_to_cart(az1, el1, 1.0)
    beam.lam=c.c/f
    beam.point(az0,el0)
    return beam


def unidirectional_broadside_rectangular_array(ar,br,theta,phi):
    # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
    # y = transverse angle, 0 = boresight, radians
    x = ar * np.sin(theta)    # sinc component (longitudinal)
    y = br * np.sin(phi)      # sinc component (transverse)
    z = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!
    z = z*np.cos(phi)         # density (from spherical integration)
    z = z*z                   # sinc^2 fn. (= power)
    return z



def TSR_gain_point(k_in, beam, az, el):

    k = k_in/np.linalg.norm(k_in)
    
    Rz = dpt.rot_mat_z(np.radians(az))
    Rx = dpt.rot_mat_x(np.radians(90.0-el))

    kb = Rx.dot(Rz.dot(k))

    theta = np.arcsin(kb[1])
    phi = np.arcsin(kb[0])
    G = unidirectional_broadside_rectangular_array(beam.ar, beam.br, theta, phi)

    return G*beam.I_0


def TSR_gain(k_in, beam):

    k = k_in/np.linalg.norm(k_in)
    
    Rz = dpt.rot_mat_z(np.radians(beam.az0))
    Rx = dpt.rot_mat_x(np.radians(90.0-beam.el0))

    kb = Rx.dot(Rz.dot(k))

    theta = np.arcsin(kb[1])
    phi = np.arcsin(kb[0])
    G = unidirectional_broadside_rectangular_array(beam.ar, beam.br, theta, phi)

    return G*beam.I_0


def tsr_fence_beam(f = 224.0e6):
    a = 30               # Panel width, metres (30 = 1 panel, 120 = all panels)
    b = 40               # Panel height, metres
    c = 299792458        # Speed of light, m/s
    wavelength = c/f     # Wavelength, metres

    ar = a / wavelength  # Antenna size in wavelengths
    br = b / wavelength  # ditto

    # Make an equirectangular projection mesh (2000 points per axis)
    x = np.linspace(-np.pi/2,np.pi/2,4000)
    y = np.linspace(-np.pi/2,np.pi/2,4000)
    xx,yy = np.meshgrid(x,y)

    # Calclate the beam pattern
    z = unidirectional_broadside_rectangular_array(ar,br,xx,yy)

    # Normalise (4pi steradian * num.pixels / integrated gain / pi^2)
    scale = 4 * np.pi * z.size / np.sum(z)   # Normalise over sphere
    sincint = np.pi*np.pi                    # Integral of the sinc^2()s: -inf:inf

    els = [30.0, 60.0, 90.0, 60.0]
    azs = [0.0, 0.0, 0.0, 180.0]

    def TSR_fence_gain(k_in, beam):
        G = 0.0

        for az, el in zip(azs, els):
            G += TSR_gain_point(k_in, beam, az + beam.az0, el + beam.el0 - 90.0)

        return G

    beam = antenna.BeamPattern(TSR_fence_gain, az0=0.0, el0=90.0, I_0=scale/sincint, f=f, beam_name='Tromso Space Radar Fence Beam')
    beam.ar = ar
    beam.br = br

    return beam


def tsr_beam(el0, f = 224.0e6):
    a = 120              # Panel width, metres (30 = 1 panel, 120 = all panels)
    b = 40               # Panel height, metres
    c = 299792458        # Speed of light, m/s
    wavelength = c/f     # Wavelength, metres

    ar = a / wavelength  # Antenna size in wavelengths
    br = b / wavelength  # ditto

    # Make an equirectangular projection mesh (2000 points per axis)
    x = np.linspace(-np.pi/2,np.pi/2,4000)
    y = np.linspace(-np.pi/2,np.pi/2,4000)
    xx,yy = np.meshgrid(x,y)

    # Calclate the beam pattern
    z = unidirectional_broadside_rectangular_array(ar,br,xx,yy)

    # Normalise (4pi steradian * num.pixels / integrated gain / pi^2)
    scale = 4 * np.pi * z.size / np.sum(z)   # Normalise over sphere
    sincint = np.pi*np.pi                    # Integral of the sinc^2()s: -inf:inf

    beam = antenna.BeamPattern(TSR_gain, az0=0.0, el0=el0, I_0=scale/sincint, f=f, beam_name='Tromso Space Radar Beam')
    beam.ar = ar
    beam.br = br

    return beam