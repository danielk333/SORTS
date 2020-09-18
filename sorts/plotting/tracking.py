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




def az_el_to_xy(az,el):
    r=np.cos(np.pi*el/180.0)
    x=r*np.cos(-np.pi*az/180.0 + np.pi/2.0)
    y=r*np.sin(-np.pi*az/180.0 + np.pi/2.0)
    return x,y



def local_tracking(azimuth, elevation, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    x,y = az_el_to_xy(azimuth,elevation)
    ax.plot( x, y )

    x,y=az_el_to_xy(np.linspace(0,360,num=360),np.repeat(30.0,360))
    ax.plot( x, y ,color="black")

    x,y=az_el_to_xy(np.linspace(0,360,num=360),np.repeat(60.0,360))
    ax.plot( x, y ,color="black")

    x,y=az_el_to_xy(np.linspace(0,360,num=360),np.repeat(80.0,360))
    ax.plot( x, y ,color="black")

    ax.axis('off')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_aspect('equal', 'datalim')

    return ax