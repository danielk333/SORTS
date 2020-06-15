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
    