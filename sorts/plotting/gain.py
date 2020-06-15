#!/usr/bin/env python

'''Plotting helper functions

'''

#Python standard import

#Third party import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#Local import



def plot_gains(beams, res=1000, min_el = 0.0, alpha = 0.5):
    '''Plot the gain of a list of beam patterns as a function of elevation at :math:`0^\circ` degrees azimuth.
    
    :param list beams: List of instances of :class:`antenna.BeamPattern`.
    :param int res: Number of points to devide the set elevation range into.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`.
    '''

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    
    
    theta=n.linspace(min_el,90.0,num=res)
    
    S=n.zeros((res,len(beams)))
    for b,beam in enumerate(beams):
        for i,th in enumerate(theta):
            k=coord.azel_to_cart(0.0, th, 1.0)
            S[i,b]=beam.gain(k)
    for b in range(len(beams)):
        ax.plot(90-theta,n.log10(S[:,b])*10.0,label="Gain " + beams[b].beam_name, alpha=alpha)
    ax.legend()
    bottom, top = plt.ylim()
    plt.ylim((0,top))
    ax.set_xlabel('Zenith angle [deg]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    ax.set_title('Gain patterns',fontsize=28)

    return fig, ax

def plot_gain_heatmap(beam, res=201, min_el = 0.0, title = None, title_size = 28, ax = None):
    '''Creates a heatmap of the beam-patters as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the wave vector x and y component range into, total number of caluclation points is the square of this number.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    '''
    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if ax is None:
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)
    else:
        fig = None


    kx=n.linspace(
        beam.on_axis[0] - n.cos(min_el*n.pi/180.0),
        beam.on_axis[0] + n.cos(min_el*n.pi/180.0),
        num=res,
    )
    ky=n.linspace(
        beam.on_axis[1] - n.cos(min_el*n.pi/180.0),
        beam.on_axis[1] + n.cos(min_el*n.pi/180.0),
        num=res,
    )
    
    S=n.zeros((res,res))
    K=n.zeros((res,res,2))
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            z2_c = (beam.on_axis[0]-x)**2 + (beam.on_axis[1]-y)**2
            z2 = x**2 + y**2
            if z2_c < n.cos(min_el*n.pi/180.0)**2 and z2 <= 1.0:
                k=n.array([x, y, n.sqrt(1.0 - z2)])
                S[i,j]=beam.gain(k)
            else:
                S[i,j] = 0;
            K[i,j,0]=x
            K[i,j,1]=y
    SdB = n.log10(S)*10.0
    SdB[SdB < 0] = 0
    conf = ax.contourf(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=n.max(SdB))
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    if title is not None:
        ax.set_title(title + ': ' + beam.beam_name + ' gain pattern', fontsize=title_size)
    else:
        ax.set_title('Gain pattern ' + beam.beam_name, fontsize=title_size)

    return fig, ax

def plot_gain(beam,res=1000,min_el = 0.0):
    '''Plot the gain of a beam patterns as a function of elevation at :math:`0^\circ` degrees azimuth.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the set elevation range into.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`.
    '''
    #turn on TeX interperter
    plt.rc('text', usetex=True)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    
    
    theta=n.linspace(min_el,90.0,num=res)
    
    S=n.zeros((res,))
    for i,th in enumerate(theta):
        k=coord.azel_ecef(beam.lat, beam.lon, 0.0, 0, th)
        S[i]=beam.gain(k)

    ax.plot(theta,n.log10(S)*10.0)
    bottom, top = plt.ylim()
    plt.ylim((0,top))
    ax.set_xlabel('Elevation [deg]',fontsize=24)
    ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_title('Gain pattern ' + beam.beam_name,\
        fontsize=28)
    
    plt.show()

def plot_gain3d(beam, res=200, min_el = 0.0):
    '''Creates a 3d plot of the beam-patters as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the wave vector x and y component range into, total number of caluclation points is the square of this number.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    '''
    #turn on TeX interperter
    plt.rc('text', usetex=True)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    kx=n.linspace(-n.cos(min_el*n.pi/180.0),n.cos(min_el*n.pi/180.0),num=res)
    ky=n.linspace(-n.cos(min_el*n.pi/180.0),n.cos(min_el*n.pi/180.0),num=res)
    
    S=n.zeros((res,res))
    K=n.zeros((res,res,2))
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            z2 = x**2 + y**2
            if z2 < n.cos(min_el*n.pi/180.0)**2:
                k=n.array([x, y, n.sqrt(1.0 - z2)])
                S[i,j]=beam.gain(k)
            else:
                S[i,j] = 0;
            K[i,j,0]=x
            K[i,j,1]=y
    SdB = n.log10(S)*10.0
    SdB[SdB < 0] = 0
    surf = ax.plot_surface(K[:,:,0],K[:,:,1],SdB,cmap=cm.plasma, linewidth=0, antialiased=False, vmin=0, vmax=n.max(SdB))
    #surf = ax.plot_surface(K[:,:,0],K[:,:,1],S.T,cmap=cm.plasma,linewidth=0)
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    ax.set_zlabel('Gain $G$ [dB]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_title('Gain pattern ' + beam.beam_name,\
        fontsize=28)
    plt.show()



